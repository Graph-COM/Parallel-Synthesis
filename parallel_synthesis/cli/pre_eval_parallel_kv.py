import argparse
import json
import os
import time
from typing import Any, Dict, List

from parallel_synthesis.models import ModelWrapper
from parallel_synthesis.cli.train_mixed_tasks import (
    build_method_for_mixed_training,
    load_task_dataset,
    to_serializable_args,
)
from parallel_synthesis.utils.common import canonicalize_task_name, parse_csv, parse_task_caps
from parallel_synthesis.utils.parallel_eval import evaluate_all_tasks
from parallel_synthesis.utils.utils import auto_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plain pre-eval for parallel_kv/fixed_parallel_kv with mapper and judger LoRA disabled."
    )
    parser.add_argument("--method", choices=["parallel_kv", "fixed_parallel_kv"], required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--tasks", type=str, required=True)
    parser.add_argument("--eval_splits", type=str, default="test")
    parser.add_argument(
        "--eval_samples_per_task",
        type=str,
        default="",
        help="Optional CSV caps aligned with --tasks. Use -1 for full split.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--generate_bs", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--parallel_kv_num_parallel_agents", type=int, default=3)
    parser.add_argument("--fixed_parallel_kv_cache_max_tokens_per_text", type=int, default=-1)
    parser.add_argument(
        "--fixed_parallel_kv_auto_cap_on_potential_oom",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--fixed_parallel_kv_auto_cap_total_tokens_threshold", type=int, default=15000)
    parser.add_argument("--fixed_parallel_kv_auto_cap_tokens_per_text", type=int, default=1024)
    parser.add_argument("--fixed_parallel_kv_debug_print_cache_inputs", action="store_true")
    parser.add_argument("--fixed_parallel_kv_debug_print_limit", type=int, default=-1)
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--output_root", type=str, default="eval_logs")
    args = parser.parse_args()
    if args.generate_bs <= 0:
        raise ValueError("--generate_bs must be > 0.")
    if args.method == "parallel_kv" and args.parallel_kv_num_parallel_agents <= 0:
        raise ValueError("--parallel_kv_num_parallel_agents must be > 0.")
    return args


def main() -> None:
    args = parse_args()
    args.prompt = "hierarchical"
    args.train_parallel_kv = False
    args.use_vllm = False
    args.parallel_kv_enable_affine_map = False
    args.parallel_kv_enable_judger_lora = False

    tasks = [canonicalize_task_name(task) for task in parse_csv(args.tasks)]
    if not tasks:
        raise ValueError("No tasks provided. Use --tasks task1,task2,...")
    eval_splits = parse_csv(args.eval_splits)
    if not eval_splits:
        raise ValueError("No eval splits provided.")
    eval_sample_caps = parse_task_caps(
        tasks,
        args.eval_samples_per_task,
        default_cap_fn=lambda _task: -1,
    )

    set_seed(args.seed)
    device = auto_device(args.device)

    safe_model = args.model_name.replace("/", "-")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = (
        args.run_name.strip()
        if args.run_name.strip()
        else f"pre_eval_{args.method}_{'-'.join(tasks)}_{safe_model}_{timestamp}"
    )
    output_dir = os.path.join(args.output_root, run_name)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "run_args.json"), "w", encoding="utf-8") as fh:
        json.dump(to_serializable_args(args), fh, ensure_ascii=False, indent=2)

    model = ModelWrapper(args.model_name, device=device, use_vllm=False, args=args)
    method = build_method_for_mixed_training(model, args)

    summary: Dict[str, Any] = {
        "mode": "pre_eval_parallel_kv",
        "method": args.method,
        "model_name": args.model_name,
        "tasks": tasks,
        "eval_splits": eval_splits,
        "save_dir": output_dir,
        "parallel_kv_enable_affine_map": False,
        "parallel_kv_enable_judger_lora": False,
    }

    start_time = time.time()
    by_split: Dict[str, Any] = {}
    for split in eval_splits:
        split_rows: Dict[str, List[Dict[str, Any]]] = {}
        for task in tasks:
            rows = load_task_dataset(
                task,
                split=split,
                args=args,
                max_samples_for_task=int(eval_sample_caps.get(task, -1)),
            )
            if rows:
                split_rows[task] = rows
        if split_rows:
            by_split[split] = evaluate_all_tasks(
                method,
                args,
                split_rows,
                split=split,
                output_dir=output_dir,
                prefix="pre_eval",
            )

    summary["reports"] = by_split
    summary["time_sec"] = round(time.time() - start_time, 4)
    summary_path = os.path.join(output_dir, "pre_eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(f"[log] wrote {summary_path}")


if __name__ == "__main__":
    main()
