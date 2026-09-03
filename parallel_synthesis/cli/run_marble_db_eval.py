import argparse
import json
import os
import shlex
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from parallel_synthesis.marble_db import (
    ParallelKVMarbleDBMethod,
    TextMASMarbleDBMethod,
    build_db_session,
    evaluate_marble_db_preds,
    load_marble_db_rows,
)
from parallel_synthesis.models import ModelWrapper
from parallel_synthesis.utils.checkpoints import resolve_parallel_kv_checkpoint
from parallel_synthesis.utils.utils import auto_device, set_seed


SUPPORTED_METHODS = ["text_mas", "parallel_kv"]


def _load_dotenv_if_present() -> str:
    candidates = [
        os.path.join(os.getcwd(), ".env"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
    ]
    seen = set()
    for path in candidates:
        norm = os.path.abspath(path)
        if norm in seen or not os.path.isfile(norm):
            continue
        seen.add(norm)
        with open(norm, "r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export ") :].strip()
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                if not key or key in os.environ:
                    continue
                value = value.strip()
                if value:
                    try:
                        parsed = shlex.split(value, comments=True, posix=True)
                        if len(parsed) == 1:
                            value = parsed[0]
                    except ValueError:
                        pass
                os.environ[key] = value
        return norm
    return ""


def _load_json_object(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        loaded = json.load(fh)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected object in {path}, got {type(loaded).__name__}.")
    return loaded


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


def _write_run_args_json(output_dir: Path, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_args.json").open("w", encoding="utf-8") as fh:
        json.dump(vars(args), fh, ensure_ascii=False, indent=2, sort_keys=True, default=str)


def _resolve_output_dirs(args: argparse.Namespace, *, safe_model: str) -> Tuple[str, Path, Path]:
    if str(getattr(args, "output_dir", "") or "").strip():
        default_run_name = f"{args.method}_marble_db_{safe_model}"
        run_name = args.run_name.strip() or default_run_name
        base_out_dir = Path(args.output_dir).expanduser().resolve()
    elif args.method == "parallel_kv" and args.parallel_kv_load_dir:
        default_run_name = f"post_eval_parallel_kv_marble_db_{safe_model}"
        run_name = args.run_name.strip() or default_run_name
        base_out_dir = Path(args.parallel_kv_load_dir).expanduser().resolve() / run_name
    else:
        default_run_name = f"{args.method}_marble_db_{safe_model}"
        run_name = args.run_name.strip() or default_run_name
        base_out_dir = Path(args.log_dir).expanduser().resolve() / run_name
    out_dir = (
        base_out_dir / "shards" / f"shard{args.shard_id:02d}of{args.num_shards:02d}"
        if args.num_shards > 1
        else base_out_dir
    )
    return run_name, base_out_dir, out_dir


def apply_shard(rows: List[Dict[str, Any]], num_shards: int, shard_id: int) -> List[Dict[str, Any]]:
    if num_shards <= 1:
        return rows
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"Invalid shard: shard_id={shard_id}, num_shards={num_shards}")
    return [row for idx, row in enumerate(rows) if (idx % num_shards) == shard_id]


def build_method(model: ModelWrapper, args: argparse.Namespace):
    if TextMASMarbleDBMethod is None or ParallelKVMarbleDBMethod is None:
        raise RuntimeError(
            "MARBLE DB methods could not be imported. "
            "Make sure the runtime environment has the core benchmark dependencies "
            "installed, including torch/transformers."
        )
    session_factory = lambda item: build_db_session(args, item)
    common = dict(
        temperature=args.temperature,
        top_p=args.top_p,
        generate_bs=args.generate_bs,
        max_tool_steps=args.max_tool_steps,
        args=args,
        session_factory=session_factory,
    )
    if args.method == "text_mas":
        return TextMASMarbleDBMethod(
            model,
            max_new_tokens=args.max_new_tokens,
            **common,
        )
    if args.method == "parallel_kv":
        return ParallelKVMarbleDBMethod(
            model,
            judger_max_new_tokens=args.max_new_tokens,
            **common,
        )
    raise ValueError(f"Unsupported method: {args.method}")


def main() -> None:
    dotenv_path = _load_dotenv_if_present()

    parser = argparse.ArgumentParser(
        description="MARBLE database benchmark runner (isolated from the main pipeline)."
    )
    parser.add_argument("--method", choices=SUPPORTED_METHODS, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument(
        "--start_sample",
        type=int,
        default=1,
        help="1-based sample number, after sharding, at which evaluation starts.",
    )
    parser.add_argument(
        "--end_sample",
        type=int,
        default=-1,
        help="Optional inclusive 1-based sample number, after sharding, at which evaluation ends.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)

    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--generate_bs", type=int, default=1)
    parser.add_argument("--max_tool_steps", type=int, default=5)
    parser.add_argument("--parallel_kv_num_parallel_agents", type=int, default=5)

    parser.add_argument("--planner_max_new_tokens", type=int, default=1024)
    parser.add_argument("--planner_temperature", type=float, default=0.7)
    parser.add_argument("--planner_top_p", type=float, default=1.0)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--load_in_4bit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load the Hugging Face backbone with bitsandbytes 4-bit quantization.",
    )
    parser.add_argument(
        "--bnb_4bit_quant_type",
        choices=["nf4", "fp4"],
        default="nf4",
    )
    parser.add_argument(
        "--bnb_4bit_compute_dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument(
        "--bnb_4bit_use_double_quant",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--kbit_keep_embeddings_in_compute_dtype",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--attn_implementation",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
        default="sdpa",
    )
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--enable_prefix_caching", action="store_true")

    parser.add_argument("--db_backend", choices=["disabled", "marble_docker"], default="disabled")
    parser.add_argument(
        "--marble_repo_dir",
        type=str,
        default="",
        help=(
            "Optional path to a local MARBLE clone. If omitted, the Docker "
            "environment bundled with parallel_synthesis.marble_db is used."
        ),
    )
    parser.add_argument(
        "--db_docker_compose_cmd",
        type=str,
        default="docker compose",
        help='Command prefix used to control the MARBLE Docker environment, e.g. "docker compose" or "sudo docker compose".',
    )
    parser.add_argument(
        "--db_anomaly_python_cmd",
        type=str,
        default="python3",
        help='Python command used for MARBLE anomaly trigger scripts, e.g. "python3".',
    )
    parser.add_argument("--db_host", type=str, default="localhost")
    parser.add_argument("--db_port", type=str, default="5432")
    parser.add_argument("--db_name", type=str, default="sysbench")
    parser.add_argument("--db_user", type=str, default="test")
    parser.add_argument("--db_password", type=str, default="Test123_456")
    parser.add_argument("--db_connect_timeout_sec", type=int, default=5)
    parser.add_argument("--db_ready_timeout_sec", type=int, default=120)
    parser.add_argument("--db_statement_timeout_ms", type=int, default=15000)
    parser.add_argument("--db_max_result_chars", type=int, default=12000)
    parser.add_argument("--db_post_setup_sleep_sec", type=float, default=2.0)
    parser.add_argument(
        "--db_anomaly_duration_sec",
        type=float,
        default=5.0,
        help="Approximate duration used by the local anomaly trigger scripts.",
    )
    parser.add_argument(
        "--db_no_restart_containers_per_sample",
        action="store_false",
        dest="db_restart_containers_per_sample",
        help="Reuse the existing MARBLE DB containers across samples instead of resetting them each time.",
    )
    parser.set_defaults(db_restart_containers_per_sample=True)
    parser.add_argument("--db_shutdown_after_item", action="store_true")

    parser.add_argument("--log_dir", type=str, default="example_logs")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Exact output directory; overrides checkpoint-relative and --log_dir defaults.",
    )
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--print_intermediate", action="store_true")
    parser.add_argument("--print_intermediate_max_chars", type=int, default=1200)

    parser.add_argument("--task", type=str, default="")
    parser.add_argument("--parallel_kv_enable_affine_map", action="store_true")
    parser.add_argument("--parallel_kv_mapper_hidden_dim", type=int, default=32)
    parser.add_argument("--parallel_kv_enable_judger_lora", action="store_true")
    parser.add_argument("--parallel_kv_lora_r", type=int, default=16)
    parser.add_argument("--parallel_kv_lora_alpha", type=int, default=32)
    parser.add_argument("--parallel_kv_lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--parallel_kv_load_dir",
        type=str,
        default="",
        help="Local checkpoint directory or Hugging Face Hub repository ID.",
    )

    args = parser.parse_args()
    args.task = "marble_db"
    args.num_shards = max(1, int(args.num_shards))
    args.shard_id = int(args.shard_id)
    args.start_sample = max(1, int(args.start_sample))
    args.end_sample = int(args.end_sample)
    if args.end_sample != -1 and args.end_sample < args.start_sample:
        raise ValueError(
            f"Expected --end_sample >= --start_sample, got "
            f"{args.end_sample} < {args.start_sample}."
        )
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError(f"Expected 0 <= --shard_id < --num_shards, got {args.shard_id}/{args.num_shards}")
    if args.method == "parallel_kv" and args.use_vllm:
        raise ValueError("parallel_kv MARBLE DB requires HF backend; disable --use_vllm.")
    if int(args.parallel_kv_num_parallel_agents) <= 0:
        raise ValueError("--parallel_kv_num_parallel_agents must be > 0.")
    if args.db_backend != "marble_docker" and args.parallel_kv_num_parallel_agents != 5:
        print("Warning: MARBLE DB uses 5 workers in the benchmark metadata.")

    if dotenv_path:
        print(f"Loaded environment variables from {dotenv_path}")

    if args.parallel_kv_load_dir and args.method != "parallel_kv":
        print(
            f"Warning: --parallel_kv_load_dir is ignored for method={args.method}. "
            "Only parallel_kv can load trainable modules."
        )
    if args.method == "parallel_kv" and args.parallel_kv_load_dir:
        checkpoint_dir = resolve_parallel_kv_checkpoint(args.parallel_kv_load_dir)
        load_affine, load_lora, load_components_source = _infer_parallel_kv_components_from_checkpoint(
            checkpoint_dir
        )
        args.parallel_kv_load_dir = str(checkpoint_dir)
        args.parallel_kv_enable_affine_map = bool(load_affine)
        args.parallel_kv_enable_judger_lora = bool(load_lora)
        args.parallel_kv_load_components_source = load_components_source
        print(
            "[parallel_kv] resolved load dir to "
            f"{args.parallel_kv_load_dir} "
            f"(mapper={args.parallel_kv_enable_affine_map}, "
            f"judger_lora={args.parallel_kv_enable_judger_lora}, "
            f"source={load_components_source})"
        )

    set_seed(args.seed)
    all_rows = list(
        load_marble_db_rows(
            max_samples=args.max_samples,
        )
    )
    if not all_rows:
        raise RuntimeError("No MARBLE DB samples were downloaded.")
    rows_after_shard = apply_shard(all_rows, args.num_shards, args.shard_id)
    total_rows = len(all_rows)
    shard_rows = len(rows_after_shard)
    start_index = args.start_sample - 1
    end_index = None if args.end_sample == -1 else args.end_sample
    rows = rows_after_shard[start_index:end_index]
    kept_rows = len(rows)

    safe_model = args.model_name.replace("/", "-")
    run_name, base_out_dir, out_dir = _resolve_output_dirs(args, safe_model=safe_model)
    _write_run_args_json(out_dir, args)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / "preds.jsonl"
    summary_path = out_dir / "summary.json"

    if args.num_shards > 1:
        print(f"Base run directory: {base_out_dir}")
    print(f"Run directory: {out_dir}")
    print(
        f"Shard selection: shard_id={args.shard_id}/{args.num_shards} "
        f"(kept {shard_rows} of {total_rows} rows before sample range)."
    )
    print(
        f"Sample range: start_sample={args.start_sample}, "
        f"end_sample={args.end_sample} (evaluating {kept_rows} rows)."
    )

    if not rows:
        metrics = evaluate_marble_db_preds([])
        metrics["benchmark"] = "marble_db"
        metrics["method"] = args.method
        metrics["model_name"] = args.model_name
        metrics["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        metrics["num_shards"] = args.num_shards
        metrics["shard_id"] = args.shard_id
        metrics["total_rows_before_shard"] = total_rows
        metrics["rows_after_shard"] = shard_rows
        metrics["start_sample"] = args.start_sample
        metrics["end_sample"] = args.end_sample
        metrics["rows_after_sample_range"] = kept_rows
        metrics["db_backend"] = args.db_backend
        with preds_path.open("w", encoding="utf-8"):
            pass
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(metrics, fh, ensure_ascii=False, indent=2)
        print(f"Saved empty predictions: {preds_path}")
        print(f"Saved summary: {summary_path}")
        return

    device = auto_device(args.device)
    model = ModelWrapper(args.model_name, device, use_vllm=args.use_vllm, args=args)
    method = build_method(model, args)

    if args.parallel_kv_load_dir and hasattr(method, "load_trainable_modules"):
        method.load_trainable_modules(args.parallel_kv_load_dir, trainable=False)
        print(f"[parallel_kv] loaded trainable modules from {args.parallel_kv_load_dir}")

    preds: List[Dict[str, Any]] = []
    print(f"Streaming predictions to: {preds_path}")
    with preds_path.open("w", encoding="utf-8") as preds_fh:
        progress = tqdm(total=len(rows), desc=f"marble_db:{args.method}")
        for start in range(0, len(rows), args.generate_bs):
            batch = rows[start : start + args.generate_bs]
            out = method.run_batch(batch)
            preds.extend(out)
            for row in out:
                preds_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            preds_fh.flush()
            progress.update(len(out))
        progress.close()

    metrics = evaluate_marble_db_preds(preds)
    metrics["benchmark"] = "marble_db"
    metrics["method"] = args.method
    metrics["model_name"] = args.model_name
    metrics["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    metrics["num_shards"] = args.num_shards
    metrics["shard_id"] = args.shard_id
    metrics["total_rows_before_shard"] = total_rows
    metrics["rows_after_shard"] = shard_rows
    metrics["start_sample"] = args.start_sample
    metrics["end_sample"] = args.end_sample
    metrics["rows_after_sample_range"] = kept_rows
    metrics["db_backend"] = args.db_backend
    if args.method == "parallel_kv":
        metrics["parallel_kv_load_dir"] = str(args.parallel_kv_load_dir or "")
        metrics["parallel_kv_enable_affine_map"] = bool(args.parallel_kv_enable_affine_map)
        metrics["parallel_kv_enable_judger_lora"] = bool(args.parallel_kv_enable_judger_lora)
        metrics["parallel_kv_load_components_source"] = str(
            getattr(args, "parallel_kv_load_components_source", "")
        )
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)

    print("\n=== MARBLE DB Benchmark Summary ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved predictions: {preds_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
