import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from parallel_synthesis.models import ModelWrapper
from parallel_synthesis.cli.train_mixed_tasks import (
    build_method_for_mixed_training,
    load_task_dataset,
)
from parallel_synthesis.utils.checkpoints import resolve_parallel_kv_checkpoint
from parallel_synthesis.utils.common import canonicalize_task_name, parse_csv
from parallel_synthesis.utils.parallel_eval import evaluate_all_tasks
from parallel_synthesis.utils.utils import auto_device, set_seed


SUPPORTED_TASKS = {
    "gsm8k",
    "aime2024",
    "aime2025",
    "gpqa",
    "mbppplus",
    "humanevalplus",
    "medqa",
}


DEFAULT_ARGS: Dict[str, Any] = {
    "method": None,
    "model_name": None,
    "tasks": "",
    "prompt": "hierarchical",
    "device": "cuda",
    "split": "test",
    "generate_bs": 1,
    "max_new_tokens": 4096,
    "temperature": 0.6,
    "top_p": 0.95,
    "parallel_kv_num_parallel_agents": 3,
    "parallel_kv_enable_affine_map": False,
    "parallel_kv_mapper_hidden_dim": 32,
    "parallel_kv_enable_judger_lora": False,
    "parallel_kv_lora_r": 16,
    "parallel_kv_lora_alpha": 32,
    "parallel_kv_lora_dropout": 0.0,
    "fixed_parallel_kv_cache_max_tokens_per_text": -1,
    "fixed_parallel_kv_auto_cap_on_potential_oom": True,
    "fixed_parallel_kv_auto_cap_total_tokens_threshold": 15000,
    "fixed_parallel_kv_auto_cap_tokens_per_text": 1024,
    "seed": 42,
    "think": False,
    "train_parallel_kv": False,
    "use_vllm": False,
}


def _load_json_object(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        loaded = json.load(fh)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected object in {path}, got {type(loaded).__name__}.")
    return loaded


def _resolve_run_args_path(checkpoint_dir: Path) -> Path | None:
    direct = checkpoint_dir / "run_args.json"
    if direct.exists():
        return direct

    meta_path = checkpoint_dir / "checkpoint_meta.json"
    if meta_path.exists():
        meta = _load_json_object(meta_path)
        source_run_dir = str(meta.get("source_run_dir", "")).strip()
        if source_run_dir:
            candidate = Path(source_run_dir).expanduser()
            if not candidate.is_absolute():
                candidate = (Path.cwd() / candidate).resolve()
            else:
                candidate = candidate.resolve()
            candidate = candidate / "run_args.json"
            if candidate.exists():
                return candidate
    return None


def _infer_parallel_kv_components_from_checkpoint(load_dir: Path) -> Tuple[bool, bool, str]:
    mapper_exists = (load_dir / "cache_mapper.pt").is_file()
    lora_exists = (load_dir / "judger_lora").is_dir()
    if mapper_exists or lora_exists:
        return mapper_exists, lora_exists, "checkpoint_artifacts"
    run_args_path = load_dir / "run_args.json"
    if run_args_path.exists():
        loaded = _load_json_object(run_args_path)
        mapper_flag = loaded.get("parallel_kv_enable_affine_map")
        lora_flag = loaded.get("parallel_kv_enable_judger_lora")
        if mapper_flag is not None or lora_flag is not None:
            return bool(mapper_flag), bool(lora_flag), "run_args_json"
    return True, True, "default_both"


def _resolve_output_dirs(
    cli: argparse.Namespace,
    *,
    checkpoint_dir: Path,
    tasks: List[str],
    safe_model: str,
    method: str,
    num_shards: int,
    shard_id: int,
) -> Tuple[str, Path, Path]:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = (
        cli.run_name.strip()
        if cli.run_name.strip()
        else f"post_eval_{method}_{'-'.join(tasks)}_{safe_model}_{timestamp}"
    )
    base_out_dir = (
        Path(cli.output_dir).expanduser().resolve()
        if cli.output_dir
        else checkpoint_dir / run_name
    )
    out_dir = (
        base_out_dir / "shards" / f"shard{shard_id:02d}of{num_shards:02d}"
        if num_shards > 1
        else base_out_dir
    )
    return run_name, base_out_dir, out_dir


def apply_shard(rows: List[Dict[str, Any]], num_shards: int, shard_id: int) -> List[Dict[str, Any]]:
    if num_shards <= 1:
        return rows
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"Invalid shard: shard_id={shard_id}, num_shards={num_shards}")
    return [row for idx, row in enumerate(rows) if (idx % num_shards) == shard_id]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Checkpoint-based post-eval for parallel_kv/fixed_parallel_kv."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Local checkpoint directory or Hugging Face Hub repository ID.",
    )
    parser.add_argument("--method", choices=["parallel_kv", "fixed_parallel_kv"], default=None)
    parser.add_argument("--tasks", type=str, default="")
    parser.add_argument("--split", type=str, default='test')
    parser.add_argument(
        "--eval_samples_per_task",
        type=int,
        default=None,
        help="Use -1 for full split, or a positive integer to cap every task on the chosen split.",
    )
    parser.add_argument(
        "--eval_mode",
        choices=["auto", "fixed_cache", "agent_output"],
        default="auto",
    )
    parser.add_argument(
        "--load_components",
        choices=["auto", "both", "mapper_only", "lora_only", "none"],
        default="auto",
        help=(
            "Which trainable ParallelKV components to load from --checkpoint_dir. "
            "auto infers from checkpoint artifacts; lora_only loads judger_lora/ "
            "without cache_mapper.pt."
        ),
    )
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--generate_bs", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--parallel_kv_num_parallel_agents", type=int, default=None)
    parser.add_argument("--fixed_parallel_kv_cache_max_tokens_per_text", type=int, default=None)
    parser.add_argument("--fixed_parallel_kv_auto_cap_on_potential_oom", action="store_true")
    parser.add_argument("--fixed_parallel_kv_auto_cap_total_tokens_threshold", type=int, default=None)
    parser.add_argument("--fixed_parallel_kv_auto_cap_tokens_per_text", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--think", dest="think", action="store_true")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--run_name", type=str, default="")
    return parser.parse_args()


def build_eval_namespace(cli: argparse.Namespace) -> argparse.Namespace:
    checkpoint_dir = resolve_parallel_kv_checkpoint(cli.checkpoint_dir)
    cfg = dict(DEFAULT_ARGS)

    run_args_path = _resolve_run_args_path(checkpoint_dir)
    if run_args_path is not None:
        cfg.update(_load_json_object(run_args_path))

    cfg["train_parallel_kv"] = False
    cfg["use_vllm"] = False
    cfg["parallel_kv_load_dir"] = str(checkpoint_dir)
    cfg["parallel_kv_save_dir"] = str(checkpoint_dir)

    for key, value in {
        "method": cli.method,
        "model_name": cli.model_name,
        "device": cli.device,
        "generate_bs": cli.generate_bs,
        "max_new_tokens": cli.max_new_tokens,
        "temperature": cli.temperature,
        "top_p": cli.top_p,
        "parallel_kv_num_parallel_agents": cli.parallel_kv_num_parallel_agents,
        "fixed_parallel_kv_cache_max_tokens_per_text": cli.fixed_parallel_kv_cache_max_tokens_per_text,
        "fixed_parallel_kv_auto_cap_total_tokens_threshold": cli.fixed_parallel_kv_auto_cap_total_tokens_threshold,
        "fixed_parallel_kv_auto_cap_tokens_per_text": cli.fixed_parallel_kv_auto_cap_tokens_per_text,
        "seed": cli.seed,
        "think": cli.think,
    }.items():
        if value is not None and value != "":
            cfg[key] = value

    if cli.fixed_parallel_kv_auto_cap_on_potential_oom:
        cfg["fixed_parallel_kv_auto_cap_on_potential_oom"] = True

    tasks_csv = cli.tasks.strip() or str(cfg.get("tasks", "")).strip()
    cfg["tasks"] = tasks_csv
    if cli.split is not None and str(cli.split).strip():
        cfg["split"] = str(cli.split).strip()
    else:
        cfg["split"] = str(cfg.get("split", "")).strip() or "test"
    if cli.eval_samples_per_task is not None:
        cfg["eval_samples_per_task"] = int(cli.eval_samples_per_task)
    else:
        cfg["eval_samples_per_task"] = int(cfg.get("eval_samples_per_task", -1) or -1)

    if cli.eval_mode == "fixed_cache":
        cfg["method"] = "fixed_parallel_kv"
    elif cli.eval_mode == "agent_output":
        cfg["method"] = "parallel_kv"

    load_affine, load_lora, load_components_source = _infer_parallel_kv_components_from_checkpoint(
        checkpoint_dir
    )
    if cli.load_components != "auto":
        load_components_source = f"cli_{cli.load_components}"
        if cli.load_components == "both":
            load_affine, load_lora = True, True
        elif cli.load_components == "mapper_only":
            load_affine, load_lora = True, False
        elif cli.load_components == "lora_only":
            load_affine, load_lora = False, True
        elif cli.load_components == "none":
            load_affine, load_lora = False, False

    cfg["parallel_kv_enable_affine_map"] = bool(load_affine)
    cfg["parallel_kv_enable_judger_lora"] = bool(load_lora)

    missing = [key for key in ("method", "model_name", "tasks") if not str(cfg.get(key, "")).strip()]
    if missing:
        raise ValueError(
            "Missing required config fields: "
            + ", ".join(missing)
            + ". Provide them via CLI or run_args.json."
        )
    if cfg["eval_samples_per_task"] == 0 or cfg["eval_samples_per_task"] < -1:
        raise ValueError("--eval_samples_per_task must be -1 or a positive integer.")
    cfg["resolved_run_args_path"] = str(run_args_path) if run_args_path is not None else ""
    cfg["load_components"] = cli.load_components
    cfg["load_components_source"] = load_components_source
    cfg["checkpoint_dir"] = str(checkpoint_dir)
    cfg["num_shards"] = max(1, int(cli.num_shards))
    cfg["shard_id"] = int(cli.shard_id)
    cfg["run_name"] = str(cli.run_name or "")
    cfg["output_dir"] = str(cli.output_dir or "")
    cfg["eval_mode"] = cli.eval_mode
    return argparse.Namespace(**cfg)


def main() -> None:
    cli = parse_args()
    args = build_eval_namespace(cli)

    tasks = [canonicalize_task_name(task) for task in parse_csv(args.tasks)]
    if not tasks:
        raise ValueError("No tasks resolved for post-eval.")
    unsupported = [task for task in tasks if task not in SUPPORTED_TASKS]
    if unsupported:
        raise ValueError(
            "Unsupported task(s) for post-eval: " + ", ".join(sorted(unsupported))
        )
    split = str(args.split).strip()
    eval_sample_cap = int(args.eval_samples_per_task)
    checkpoint_dir = Path(args.checkpoint_dir)
    args.num_shards = max(1, int(args.num_shards))
    args.shard_id = int(args.shard_id)
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError(
            f"Expected 0 <= --shard_id < --num_shards, got {args.shard_id}/{args.num_shards}"
        )

    safe_model = args.model_name.replace("/", "-")
    run_name, base_output_dir, output_dir = _resolve_output_dirs(
        cli,
        checkpoint_dir=checkpoint_dir,
        tasks=tasks,
        safe_model=safe_model,
        method=str(args.method),
        num_shards=args.num_shards,
        shard_id=args.shard_id,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(args.seed))
    rows_by_task: Dict[str, List[Dict[str, Any]]] = {}
    total_rows_before_shard = 0
    total_rows_after_shard = 0
    rows_before_shard_by_task: Dict[str, int] = {}
    rows_after_shard_by_task: Dict[str, int] = {}
    for task in tasks:
        rows = load_task_dataset(
            task,
            split=split,
            args=args,
            max_samples_for_task=eval_sample_cap,
        )
        rows_before_shard_by_task[task] = len(rows)
        sharded_rows = apply_shard(rows, args.num_shards, args.shard_id)
        rows_after_shard_by_task[task] = len(sharded_rows)
        rows_by_task[task] = sharded_rows
        total_rows_before_shard += len(rows)
        total_rows_after_shard += len(sharded_rows)

    print(f"Base run directory: {base_output_dir}")
    print(f"Run directory: {output_dir}")
    print(
        f"Shard selection: shard_id={args.shard_id}/{args.num_shards} "
        f"(kept {total_rows_after_shard} of {total_rows_before_shard} rows across {len(tasks)} task(s))."
    )

    device = auto_device(str(args.device))
    model = ModelWrapper(args.model_name, device=device, use_vllm=False, args=args)
    method = build_method_for_mixed_training(model, args)
    method.load_trainable_modules(str(checkpoint_dir), trainable=False)
    print(f"[eval] loaded ParallelKV checkpoint from {checkpoint_dir}")

    start_time = time.time()
    report = (
        evaluate_all_tasks(
            method,
            args,
            rows_by_task,
            split=split,
            output_dir=str(output_dir),
            prefix="post_eval",
        )
        if rows_by_task
        else {}
    )

    summary: Dict[str, Any] = {
        "mode": "post_eval_parallel_kv",
        "method": args.method,
        "model_name": args.model_name,
        "tasks": tasks,
        "split": split,
        "eval_samples_per_task": eval_sample_cap,
        "checkpoint_dir": str(checkpoint_dir),
        "run_args_path": args.resolved_run_args_path,
        "load_components": args.load_components,
        "load_components_source": args.load_components_source,
        "load_affine_map": bool(args.parallel_kv_enable_affine_map),
        "load_judger_lora": bool(args.parallel_kv_enable_judger_lora),
        "run_name": run_name,
        "base_output_dir": str(base_output_dir),
        "output_dir": str(output_dir),
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "total_rows_before_shard": total_rows_before_shard,
        "total_rows_after_shard": total_rows_after_shard,
        "rows_before_shard_by_task": rows_before_shard_by_task,
        "rows_after_shard_by_task": rows_after_shard_by_task,
        "report": report,
        "time_sec": round(time.time() - start_time, 4),
    }

    with open(output_dir / "run_args.json", "w", encoding="utf-8") as fh:
        json.dump(vars(args), fh, ensure_ascii=False, indent=2, sort_keys=True)
    summary_path = output_dir / "post_eval_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(f"[log] wrote {summary_path}")


if __name__ == "__main__":
    main()
