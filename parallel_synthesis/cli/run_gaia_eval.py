import argparse
import json
import os
import shlex
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from parallel_synthesis.models import ModelWrapper
from parallel_synthesis.toolcall import (
    BaselineToolCallingMethod,
    ParallelKVToolCallingMethod,
    TextMASToolCallingMethod,
    build_default_tool_registry,
    load_gaia_rows,
)
from parallel_synthesis.utils.checkpoints import resolve_parallel_kv_checkpoint
from parallel_synthesis.utils.utils import auto_device, set_seed


SUPPORTED_METHODS = ["baseline", "text_mas", "parallel_kv"]


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

        with open(norm, "r", encoding="utf-8") as f:
            for raw_line in f:
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


def evaluate(preds: List[Dict]) -> Dict[str, float]:
    total = len(preds)
    correct = sum(1 for p in preds if bool(p.get("correct", False)))
    ttfts = [
        float(p["ttft_sec"])
        for p in preds
        if p.get("ttft_sec") is not None
    ]
    cache_prepare_times = [
        float(p["cache_prepare_sec"])
        for p in preds
        if p.get("cache_prepare_sec") is not None
    ]
    return {
        "samples": total,
        "correct": correct,
        "accuracy": (correct / total) if total > 0 else 0.0,
        "avg_ttft_sec": round(sum(ttfts) / len(ttfts), 4) if ttfts else None,
        "ttft_measured_samples": len(ttfts),
        "avg_cache_prepare_sec": (
            round(sum(cache_prepare_times) / len(cache_prepare_times), 4)
            if cache_prepare_times
            else None
        ),
        "cache_prepare_measured_samples": len(cache_prepare_times),
    }


def build_method(model: ModelWrapper, args: argparse.Namespace, tools):
    common = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        generate_bs=args.generate_bs,
        max_tool_steps=args.max_tool_steps,
        tools=tools,
        args=args,
    )
    if args.method == "baseline":
        return BaselineToolCallingMethod(model, **common)
    if args.method == "text_mas":
        return TextMASToolCallingMethod(model, **common)
    if args.method == "parallel_kv":
        return ParallelKVToolCallingMethod(
            model,
            judger_max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            generate_bs=args.generate_bs,
            max_tool_steps=args.max_tool_steps,
            tools=tools,
            args=args,
        )
    raise ValueError(f"Unsupported method: {args.method}")


def load_rows(args: argparse.Namespace) -> List[Dict]:
    return list(
        load_gaia_rows(
            split=args.split,
            gaia_config=args.gaia_config,
            max_samples=args.max_samples,
        )
    )


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_run_args_json(output_dir: Path, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_args.json").open("w", encoding="utf-8") as fh:
        json.dump(vars(args), fh, ensure_ascii=False, indent=2, sort_keys=True, default=str)


def _benchmark_name_for_run(args: argparse.Namespace) -> str:
    benchmark_name = "gaia"
    gaia_config = str(getattr(args, "gaia_config", "auto") or "auto").strip()
    if not gaia_config or gaia_config.lower() == "auto":
        return benchmark_name
    return f"{benchmark_name}_{gaia_config.replace('/', '-')}"


def _resolve_output_dirs(
    args: argparse.Namespace,
    *,
    safe_model: str,
) -> Tuple[str, Path, Path]:
    benchmark_name = _benchmark_name_for_run(args)
    if str(getattr(args, "output_dir", "") or "").strip():
        default_run_name = f"{args.method}_toolcall_{benchmark_name}_{safe_model}"
        run_name = args.run_name.strip() or default_run_name
        base_out_dir = Path(args.output_dir).expanduser().resolve()
    elif args.method == "parallel_kv" and args.parallel_kv_load_dir:
        default_run_name = f"post_eval_parallel_kv_{benchmark_name}_{safe_model}"
        run_name = args.run_name.strip() or default_run_name
        base_out_dir = Path(args.parallel_kv_load_dir).expanduser().resolve() / run_name
    else:
        default_run_name = f"{args.method}_toolcall_{benchmark_name}_{safe_model}"
        run_name = args.run_name.strip() or default_run_name
        base_out_dir = Path(args.log_dir).expanduser().resolve() / run_name

    out_dir = (
        base_out_dir / "shards" / f"shard{args.shard_id:02d}of{args.num_shards:02d}"
        if args.num_shards > 1
        else base_out_dir
    )
    return run_name, base_out_dir, out_dir


def apply_shard(rows: List[Dict], num_shards: int, shard_id: int) -> List[Dict]:
    if num_shards <= 1:
        return rows
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"Invalid shard: shard_id={shard_id}, num_shards={num_shards}")
    return [row for idx, row in enumerate(rows) if (idx % num_shards) == shard_id]


def load_preds_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    preds: List[Dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in existing predictions at {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object in existing predictions at {path}:{line_no}.")
            preds.append(obj)
    return preds


def ensure_trailing_newline(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb") as fh:
        fh.seek(-1, os.SEEK_END)
        last = fh.read(1)
    if last != b"\n":
        with path.open("a", encoding="utf-8") as fh:
            fh.write("\n")


def main() -> None:
    dotenv_path = _load_dotenv_if_present()

    parser = argparse.ArgumentParser(description="End-to-end GAIA tool-calling evaluation.")

    parser.add_argument("--method", choices=SUPPORTED_METHODS, required=True)
    parser.add_argument("--model_name", type=str, required=True)

    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="validation",
        help=(
            "Use validation for locally scored evaluation. Test answers are "
            "hidden and require official GAIA submission."
        ),
    )
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)

    parser.add_argument("--max_new_tokens", type=int, default=10000)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--generate_bs", type=int, default=1)
    parser.add_argument("--max_tool_steps", type=int, default=5)
    parser.add_argument("--parallel_kv_num_parallel_agents", type=int, default=3)

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

    parser.add_argument(
        "--gaia_config",
        type=str,
        default="2023_level1",
        help="GAIA Hugging Face config. Default: 2023_level1. "
        "Use 2023_level1/2/3, the shorthands "
        "level1/2/3, or auto.",
    )
    # Tool options
    parser.add_argument(
        "--disable_python_interpreter",
        action="store_true",
        help=(
            "Exclude the local Python interpreter from the model-callable tool registry. "
            "Search, visit, and file parsing remain enabled."
        ),
    )

    # Logging options
    parser.add_argument("--log_dir", type=str, default="example_logs")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Exact output directory; overrides checkpoint-relative and --log_dir defaults.",
    )
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument(
        "--start_sample",
        type=int,
        default=1,
        help=(
            "1-based sample number, after sharding, to start evaluating from. "
            "For example, --start_sample 107 skips the first 106 shard rows."
        ),
    )
    parser.add_argument(
        "--append_preds",
        action="store_true",
        help="Append to an existing preds.jsonl instead of overwriting it.",
    )
    parser.add_argument(
        "--resume_from_preds",
        action="store_true",
        help=(
            "Resume from the existing preds.jsonl length plus one and append new rows. "
            "This is equivalent to --start_sample N+1 --append_preds when N predictions already exist."
        ),
    )
    parser.add_argument(
        "--print_intermediate",
        action="store_true",
        help="Print intermediate model outputs and tool responses during solving.",
    )
    parser.add_argument(
        "--print_intermediate_max_chars",
        type=int,
        default=1200,
        help="Maximum characters per printed intermediate block.",
    )

    # ParallelKV parent compatibility
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
        help=(
            "Optional local checkpoint directory or Hugging Face Hub repository ID "
            "containing Parallel Synthesis modules (cache mapper and/or judger LoRA)."
        ),
    )
    args = parser.parse_args()
    args.benchmark = "gaia"
    args.task = "gaia"
    args.num_shards = max(1, int(args.num_shards))
    args.shard_id = int(args.shard_id)
    args.start_sample = max(1, int(args.start_sample))
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError(f"Expected 0 <= --shard_id < --num_shards, got {args.shard_id}/{args.num_shards}")
    if int(args.parallel_kv_num_parallel_agents) <= 0:
        raise ValueError("--parallel_kv_num_parallel_agents must be > 0.")
    if args.resume_from_preds and args.start_sample != 1:
        raise ValueError("Use either --resume_from_preds or --start_sample, not both.")
    if args.split == "test":
        print(
            "Note: GAIA test answers are hidden. This run can generate "
            "predictions, but results require official submission."
        )

    if dotenv_path:
        print(f"Loaded environment variables from {dotenv_path}")

    if args.method == "parallel_kv" and args.use_vllm:
        raise ValueError("parallel_kv tool-calling requires HF backend; disable --use_vllm.")
    if args.parallel_kv_load_dir and args.method != "parallel_kv":
        print(
            f"Warning: --parallel_kv_load_dir is ignored for method={args.method}. "
            "Only toolcall parallel_kv can load trainable modules."
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
    all_rows = load_rows(args)
    if not all_rows:
        raise RuntimeError(f"No samples found for benchmark={args.benchmark}. Check dataset path/options.")
    rows_after_shard = apply_shard(all_rows, args.num_shards, args.shard_id)
    total_rows = len(all_rows)
    shard_rows_count = len(rows_after_shard)

    safe_model = args.model_name.replace("/", "-")
    run_name, base_out_dir, out_dir = _resolve_output_dirs(
        args,
        safe_model=safe_model,
    )
    ensure_dir(str(out_dir))
    preds_path = out_dir / "preds.jsonl"
    summary_path = out_dir / "summary.json"

    existing_preds: List[Dict] = []
    if args.resume_from_preds:
        existing_preds = load_preds_jsonl(preds_path)
        args.start_sample = len(existing_preds) + 1
        args.append_preds = True
        print(
            f"Resume requested: found {len(existing_preds)} existing predictions; "
            f"starting at sample {args.start_sample}."
        )
    elif args.append_preds:
        existing_preds = load_preds_jsonl(preds_path)
        if existing_preds and len(existing_preds) != args.start_sample - 1:
            print(
                "Warning: --append_preds is set but existing prediction count "
                f"({len(existing_preds)}) does not match --start_sample - 1 "
                f"({args.start_sample - 1}). Appending anyway."
            )

    start_index = args.start_sample - 1
    rows = rows_after_shard[start_index:]
    kept_rows = len(rows)
    _write_run_args_json(base_out_dir, args)

    print(f"Base run directory: {base_out_dir}")
    print(
        f"Shard selection: shard_id={args.shard_id}/{args.num_shards} "
        f"(kept {shard_rows_count} of {total_rows} rows before start offset)."
    )
    print(
        f"Start selection: start_sample={args.start_sample} "
        f"(1-based after sharding; evaluating {kept_rows} rows)."
    )
    if not rows:
        metrics = evaluate(existing_preds if args.append_preds else [])
        metrics["benchmark"] = args.benchmark
        metrics["method"] = args.method
        metrics["model_name"] = args.model_name
        metrics["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        metrics["scoring"] = "normalized_exact_match"
        metrics["num_shards"] = args.num_shards
        metrics["shard_id"] = args.shard_id
        metrics["total_rows_before_shard"] = total_rows
        metrics["rows_after_shard"] = shard_rows_count
        metrics["start_sample"] = args.start_sample
        metrics["start_index"] = start_index
        metrics["rows_after_start"] = kept_rows
        metrics["append_preds"] = bool(args.append_preds)
        metrics["resume_from_preds"] = bool(args.resume_from_preds)
        metrics["existing_preds_loaded"] = len(existing_preds)
        metrics["new_preds_written"] = 0
        if args.method == "parallel_kv":
            metrics["parallel_kv_load_dir"] = str(args.parallel_kv_load_dir or "")
            metrics["parallel_kv_enable_affine_map"] = bool(args.parallel_kv_enable_affine_map)
            metrics["parallel_kv_enable_judger_lora"] = bool(args.parallel_kv_enable_judger_lora)
            metrics["parallel_kv_load_components_source"] = str(
                getattr(args, "parallel_kv_load_components_source", "")
            )
        if not args.append_preds:
            with open(preds_path, "w", encoding="utf-8"):
                pass
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, ensure_ascii=False, indent=2)
        print(f"No rows to evaluate from start_sample={args.start_sample}.")
        print(f"Saved predictions: {preds_path}")
        print(f"Saved summary: {summary_path}")
        return

    device = auto_device(args.device)
    model = ModelWrapper(args.model_name, device, use_vllm=args.use_vllm, args=args)

    tools = build_default_tool_registry(
        enable_python_interpreter=not args.disable_python_interpreter,
    )
    method = build_method(model, args, tools)

    if args.parallel_kv_load_dir and hasattr(method, "load_trainable_modules"):
        method.load_trainable_modules(args.parallel_kv_load_dir, trainable=False)
        print(f"[parallel_kv] loaded trainable modules from {args.parallel_kv_load_dir}")

    print(f"Streaming predictions to: {preds_path}")

    preds: List[Dict] = list(existing_preds) if args.append_preds else []
    new_preds_written = 0
    file_mode = "a" if args.append_preds else "w"
    if args.append_preds:
        ensure_trailing_newline(preds_path)
    with open(preds_path, file_mode, encoding="utf-8") as preds_fh:
        progress = tqdm(total=len(rows), desc=f"{args.benchmark}:{args.method}")
        for start in range(0, len(rows), args.generate_bs):
            batch = rows[start : start + args.generate_bs]
            out = method.run_batch(batch)
            preds.extend(out)
            for row in out:
                preds_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            preds_fh.flush()
            new_preds_written += len(out)
            progress.update(len(out))
        progress.close()

    metrics = evaluate(preds)
    metrics["benchmark"] = args.benchmark
    metrics["method"] = args.method
    metrics["model_name"] = args.model_name
    metrics["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    metrics["scoring"] = "normalized_exact_match"
    metrics["num_shards"] = args.num_shards
    metrics["shard_id"] = args.shard_id
    metrics["total_rows_before_shard"] = total_rows
    metrics["rows_after_shard"] = shard_rows_count
    metrics["start_sample"] = args.start_sample
    metrics["start_index"] = start_index
    metrics["rows_after_start"] = kept_rows
    metrics["append_preds"] = bool(args.append_preds)
    metrics["resume_from_preds"] = bool(args.resume_from_preds)
    metrics["existing_preds_loaded"] = len(existing_preds)
    metrics["new_preds_written"] = new_preds_written
    if args.method == "parallel_kv":
        metrics["parallel_kv_load_dir"] = str(args.parallel_kv_load_dir or "")
        metrics["parallel_kv_enable_affine_map"] = bool(args.parallel_kv_enable_affine_map)
        metrics["parallel_kv_enable_judger_lora"] = bool(args.parallel_kv_enable_judger_lora)
        metrics["parallel_kv_load_components_source"] = str(
            getattr(args, "parallel_kv_load_components_source", "")
        )
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)

    print("\n=== Tool-Calling Benchmark Summary ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved predictions: {preds_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
