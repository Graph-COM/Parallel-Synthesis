import argparse
import json
import os
import re
import shlex
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from parallel_synthesis.models import ModelWrapper
from parallel_synthesis.toolcall import build_default_tool_registry
from parallel_synthesis.toolcall.gaia_sanity_eval import build_sanity_items_from_preds, build_sanity_method
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


def _load_optional_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        return None
    return _load_json_object(path)


def _find_neighbor_json(base_dir: Optional[Path], preds_path: Path, filename: str) -> Optional[Path]:
    candidates: List[Path] = []
    if base_dir is not None:
        candidates.append(base_dir / filename)
    candidates.append(preds_path.parent / filename)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
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


def _parallel_kv_components_from_cli(value: str) -> Tuple[bool, bool, str]:
    mode = str(value or "auto").strip().lower()
    if mode == "both":
        return True, True, "cli_both"
    if mode == "mapper_only":
        return True, False, "cli_mapper_only"
    if mode == "lora_only":
        return False, True, "cli_lora_only"
    if mode == "none":
        return False, False, "cli_none"
    raise ValueError(f"Unsupported --load_components value: {value!r}")


def evaluate(preds: List[Dict[str, Any]]) -> Dict[str, float]:
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


def apply_shard(rows: List[Dict[str, Any]], num_shards: int, shard_id: int) -> List[Dict[str, Any]]:
    if num_shards <= 1:
        return rows
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"Invalid shard: shard_id={shard_id}, num_shards={num_shards}")
    return [row for idx, row in enumerate(rows) if (idx % num_shards) == shard_id]


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip()).strip("._") or "run"


def _infer_source_benchmark(
    source_summary: Optional[Dict[str, Any]],
    source_run_args: Optional[Dict[str, Any]],
    source_preds_path: Path,
) -> str:
    for source in (source_summary, source_run_args):
        if source and str(source.get("benchmark", "")).strip():
            return str(source.get("benchmark")).strip().lower()
    path_text = str(source_preds_path).lower()
    if "gaia" in path_text:
        return "gaia"
    return "gaia" if "gaia" in path_text else ""


def _source_name(source_run_dir: Optional[Path], source_preds_path: Path) -> str:
    if source_run_dir is not None:
        return source_run_dir.name or "source_run"
    return source_preds_path.parent.name or "source_run"


def _benchmark_name_for_run(
    benchmark: str,
    *,
    source_run_args: Optional[Dict[str, Any]],
) -> str:
    benchmark_name = str(benchmark).strip().lower()
    if benchmark_name != "gaia":
        return benchmark_name
    loaded = source_run_args or {}
    gaia_config = str(loaded.get("gaia_config", "auto") or "auto").strip()
    if not gaia_config or gaia_config.lower() == "auto":
        return benchmark_name
    return f"{benchmark_name}_{gaia_config.replace('/', '-')}"


def _resolve_output_dirs(
    args: argparse.Namespace,
    *,
    benchmark: str,
    source_name: str,
    safe_model: str,
) -> Tuple[str, Path, Path]:
    run_bits = [
        "sanity_eval",
        benchmark,
        args.method,
        safe_model,
        f"from_{_safe_name(source_name)}",
    ]
    load_components = str(getattr(args, "load_components", "auto") or "auto").strip().lower()
    if args.method == "parallel_kv" and load_components != "auto":
        run_bits.append(f"components_{_safe_name(load_components)}")
    memory_mode = str(getattr(args, "source_worker_memory_mode", "final_output") or "final_output").strip().lower()
    run_bits.append(f"memory_{_safe_name(memory_mode)}")
    default_run_name = "_".join(run_bits)
    run_name = args.run_name.strip() or default_run_name
    if str(getattr(args, "output_dir", "") or "").strip():
        base_out_dir = Path(args.output_dir).expanduser().resolve()
    elif args.method == "parallel_kv" and args.parallel_kv_load_dir:
        base_out_dir = Path(args.parallel_kv_load_dir).expanduser().resolve() / run_name
    else:
        base_out_dir = Path(args.log_dir).expanduser().resolve() / f"sanity_{benchmark}" / run_name

    out_dir = (
        base_out_dir / "shards" / f"shard{args.shard_id:02d}of{args.num_shards:02d}"
        if args.num_shards > 1
        else base_out_dir
    )
    return run_name, base_out_dir, out_dir


def _write_run_args_json(output_dir: Path, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_args.json").open("w", encoding="utf-8") as fh:
        json.dump(vars(args), fh, ensure_ascii=False, indent=2, sort_keys=True, default=str)


def main() -> None:
    dotenv_path = _load_dotenv_if_present()

    parser = argparse.ArgumentParser(
        description="GAIA sanity evaluation using fixed worker search trajectories from an existing run."
    )
    parser.add_argument("--method", choices=SUPPORTED_METHODS, required=True)
    parser.add_argument("--model_name", type=str, required=True)

    parser.add_argument("--source_run_dir", type=str, default="")
    parser.add_argument("--source_preds_path", type=str, default="")
    parser.add_argument("--source_summary_path", type=str, default="")
    parser.add_argument("--source_run_args_path", type=str, default="")

    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)

    parser.add_argument("--max_new_tokens", type=int, default=4096)
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
        help="Load the frozen Hugging Face backbone in bitsandbytes 4-bit mode.",
    )
    parser.add_argument("--bnb_4bit_quant_type", choices=("nf4", "fp4"), default="nf4")
    parser.add_argument(
        "--bnb_4bit_compute_dtype",
        choices=("bfloat16", "float16", "float32"),
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
        choices=("auto", "flash_attention_2", "sdpa", "eager"),
        default="sdpa",
    )
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--enable_prefix_caching", action="store_true")

    parser.add_argument("--log_dir", type=str, default="sanity_logs")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument(
        "--source_worker_memory_mode",
        choices=(
            "final_output",
            "full_prompt_output",
            "each_turn_output",
            "full_prompt_output_no_system",
        ),
        default="final_output",
        help=(
            "Which source-worker information to pass to the synthesizer. "
            "'final_output' reproduces the current behavior. "
            "'full_prompt_output' uses the final-turn prompt+output as memory/context. "
            "'each_turn_output' uses every assistant turn output; for parallel_kv it reconstructs one "
            "prompt-conditioned output cache per turn using that turn's exact prompt, then concatenates "
            "those per-turn caches back into one cache block per worker. "
            "'full_prompt_output_no_system' uses the final-turn prompt+output with the worker system prompt "
            "removed; for parallel_kv it first encodes the full prompt+output with system included and then "
            "strips the system-prefix cache span and shifts positions back by the system-prompt length."
        ),
    )
    parser.add_argument("--print_intermediate", action="store_true")
    parser.add_argument("--print_intermediate_max_chars", type=int, default=1200)
    parser.add_argument(
        "--disable_python_interpreter",
        action="store_true",
        help="Exclude arbitrary local Python execution from the model-callable tool registry.",
    )

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
    parser.add_argument(
        "--load_components",
        choices=("auto", "both", "mapper_only", "lora_only", "none"),
        default="auto",
        help=(
            "Which ParallelKV trainable components to load for sanity eval. "
            "'auto' infers from checkpoint artifacts when --parallel_kv_load_dir is set. "
            "'lora_only' loads only judger_lora, and 'none' runs the base model with "
            "raw reconstructed source caches."
        ),
    )

    args = parser.parse_args()
    args.num_shards = max(1, int(args.num_shards))
    args.shard_id = int(args.shard_id)
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError(f"Expected 0 <= --shard_id < --num_shards, got {args.shard_id}/{args.num_shards}")
    if int(args.parallel_kv_num_parallel_agents) <= 0:
        raise ValueError("--parallel_kv_num_parallel_agents must be > 0.")
    if not args.source_run_dir and not args.source_preds_path:
        raise ValueError("Provide either --source_run_dir or --source_preds_path.")
    if args.method == "parallel_kv" and args.use_vllm:
        raise ValueError("parallel_kv sanity-eval requires HF backend; disable --use_vllm.")

    source_run_dir: Optional[Path] = None
    if args.source_run_dir:
        source_run_dir = Path(args.source_run_dir).expanduser().resolve()
        if not source_run_dir.exists() or not source_run_dir.is_dir():
            raise FileNotFoundError(f"source run dir not found: {source_run_dir}")
    source_preds_path = (
        Path(args.source_preds_path).expanduser().resolve()
        if args.source_preds_path
        else (source_run_dir / "preds.jsonl").resolve()
    )
    if not source_preds_path.exists():
        raise FileNotFoundError(f"source preds file not found: {source_preds_path}")

    source_summary_path = (
        Path(args.source_summary_path).expanduser().resolve()
        if args.source_summary_path
        else _find_neighbor_json(source_run_dir, source_preds_path, "summary.json")
    )
    source_run_args_path = (
        Path(args.source_run_args_path).expanduser().resolve()
        if args.source_run_args_path
        else _find_neighbor_json(source_run_dir, source_preds_path, "run_args.json")
    )

    source_summary = _load_optional_json(source_summary_path)
    source_run_args = _load_optional_json(source_run_args_path)

    source_benchmark = _infer_source_benchmark(
        source_summary,
        source_run_args,
        source_preds_path,
    )
    if source_benchmark and source_benchmark != "gaia":
        raise ValueError(
            "GAIA sanity evaluation requires source trajectories from a GAIA run; "
            f"found benchmark={source_benchmark!r}."
        )
    args.benchmark = "gaia"
    args.task = "gaia"

    if dotenv_path:
        print(f"Loaded environment variables from {dotenv_path}")

    if args.method != "parallel_kv" and args.parallel_kv_load_dir:
        print(
            f"Warning: --parallel_kv_load_dir is ignored for method={args.method}. "
            "Only sanity parallel_kv can load trainable modules."
        )
    if args.method != "parallel_kv" and args.load_components != "auto":
        print(
            f"Warning: --load_components={args.load_components} is ignored for method={args.method}. "
            "Component selection only applies to sanity parallel_kv."
        )
    if (
        args.method == "parallel_kv"
        and not args.parallel_kv_load_dir
        and args.load_components not in {"auto", "none"}
    ):
        raise ValueError(
            f"--load_components={args.load_components} requires --parallel_kv_load_dir. "
            "Use --load_components none, or omit --load_components, for base-model raw-cache sanity eval."
        )
    if args.method == "parallel_kv" and args.parallel_kv_load_dir:
        checkpoint_dir = resolve_parallel_kv_checkpoint(args.parallel_kv_load_dir)
        if args.load_components == "auto":
            load_affine, load_lora, load_components_source = _infer_parallel_kv_components_from_checkpoint(
                checkpoint_dir
            )
        else:
            load_affine, load_lora, load_components_source = _parallel_kv_components_from_cli(
                args.load_components
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
    elif args.method == "parallel_kv" and args.load_components == "none":
        args.parallel_kv_enable_affine_map = False
        args.parallel_kv_enable_judger_lora = False
        args.parallel_kv_load_components_source = "cli_none"

    source_model_name = ""
    for source in (source_summary, source_run_args):
        if source and str(source.get("model_name", "")).strip():
            source_model_name = str(source.get("model_name")).strip()
            break
    if source_model_name and source_model_name != args.model_name:
        print(
            f"Note: source worker traces came from model={source_model_name}, "
            f"but sanity eval reconstructs worker text/cache with the current model={args.model_name}. "
            "This is intentional for fixed-trajectory synthesis tests."
        )

    set_seed(args.seed)
    all_rows = build_sanity_items_from_preds(
        source_preds_path,
        max_samples=args.max_samples,
    )
    if not all_rows:
        raise RuntimeError("No source rows found for sanity evaluation.")
    rows = apply_shard(all_rows, args.num_shards, args.shard_id)
    total_rows = len(all_rows)
    kept_rows = len(rows)

    safe_model = args.model_name.replace("/", "-")
    benchmark_name = _benchmark_name_for_run(
        args.benchmark,
        source_run_args=source_run_args,
    )
    run_name, base_out_dir, out_dir = _resolve_output_dirs(
        args,
        benchmark=benchmark_name,
        source_name=_source_name(source_run_dir, source_preds_path),
        safe_model=safe_model,
    )
    _write_run_args_json(base_out_dir, args)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / "preds.jsonl"
    summary_path = out_dir / "summary.json"

    print(f"Base run directory: {base_out_dir}")
    print(
        f"Shard selection: shard_id={args.shard_id}/{args.num_shards} "
        f"(kept {kept_rows} of {total_rows} rows)."
    )
    if not rows:
        metrics = evaluate([])
        metrics["benchmark"] = args.benchmark
        metrics["method"] = f"sanity_{args.method}"
        metrics["model_name"] = args.model_name
        metrics["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        metrics["scoring"] = "normalized_exact_match"
        metrics["num_shards"] = args.num_shards
        metrics["shard_id"] = args.shard_id
        metrics["source_preds_path"] = str(source_preds_path)
        metrics["source_run_dir"] = str(source_run_dir or "")
        metrics["source_summary_path"] = str(source_summary_path or "")
        metrics["source_run_args_path"] = str(source_run_args_path or "")
        metrics["source_worker_memory_mode"] = str(args.source_worker_memory_mode)
        metrics["load_components"] = str(args.load_components)
        metrics["total_rows_before_shard"] = total_rows
        metrics["rows_after_shard"] = kept_rows
        with open(preds_path, "w", encoding="utf-8"):
            pass
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, ensure_ascii=False, indent=2)
        print(f"Saved empty predictions: {preds_path}")
        print(f"Saved summary: {summary_path}")
        return

    device = auto_device(args.device)
    model = ModelWrapper(args.model_name, device, use_vllm=args.use_vllm, args=args)

    tools = build_default_tool_registry(
        enable_python_interpreter=not args.disable_python_interpreter,
    )
    method = build_sanity_method(model, args, tools)

    should_load_parallel_kv_components = (
        args.method == "parallel_kv"
        and bool(args.parallel_kv_load_dir)
        and (
            bool(args.parallel_kv_enable_affine_map)
            or bool(args.parallel_kv_enable_judger_lora)
        )
    )
    if should_load_parallel_kv_components and hasattr(method, "load_trainable_modules"):
        method.load_trainable_modules(args.parallel_kv_load_dir, trainable=False)
        print(f"[parallel_kv] loaded trainable modules from {args.parallel_kv_load_dir}")
    elif args.method == "parallel_kv" and args.parallel_kv_load_dir:
        print(
            "[parallel_kv] no trainable components selected; "
            "using base model with raw reconstructed source caches."
        )

    print(f"Streaming predictions to: {preds_path}")

    preds: List[Dict[str, Any]] = []
    with open(preds_path, "w", encoding="utf-8") as preds_fh:
        progress = tqdm(total=len(rows), desc=f"sanity:{args.benchmark}:{args.method}")
        for start in range(0, len(rows), args.generate_bs):
            batch = rows[start : start + args.generate_bs]
            out = method.run_batch(batch)
            preds.extend(out)
            for row in out:
                preds_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            preds_fh.flush()
            progress.update(len(out))
        progress.close()

    metrics = evaluate(preds)
    metrics["benchmark"] = args.benchmark
    metrics["method"] = f"sanity_{args.method}"
    metrics["model_name"] = args.model_name
    metrics["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    metrics["scoring"] = "normalized_exact_match"
    metrics["num_shards"] = args.num_shards
    metrics["shard_id"] = args.shard_id
    metrics["total_rows_before_shard"] = total_rows
    metrics["rows_after_shard"] = kept_rows
    metrics["source_preds_path"] = str(source_preds_path)
    metrics["source_run_dir"] = str(source_run_dir or "")
    metrics["source_summary_path"] = str(source_summary_path or "")
    metrics["source_run_args_path"] = str(source_run_args_path or "")
    metrics["source_worker_memory_mode"] = str(args.source_worker_memory_mode)
    metrics["load_components"] = str(args.load_components)
    metrics["source_method"] = str((source_summary or {}).get("method", (source_run_args or {}).get("method", "")))
    metrics["source_model_name"] = str(source_model_name)
    if args.method == "parallel_kv":
        metrics["parallel_kv_load_dir"] = str(args.parallel_kv_load_dir or "")
        metrics["parallel_kv_enable_affine_map"] = bool(args.parallel_kv_enable_affine_map)
        metrics["parallel_kv_enable_judger_lora"] = bool(args.parallel_kv_enable_judger_lora)
        metrics["parallel_kv_load_components_source"] = str(
            getattr(args, "parallel_kv_load_components_source", "")
        )
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)

    print("\n=== GAIA Fixed-Trajectory Sanity Summary ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved predictions: {preds_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
