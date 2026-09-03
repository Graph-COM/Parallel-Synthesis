import argparse
import json
import os
from typing import Optional, TextIO

from parallel_synthesis.data import build_dataset_iter
from parallel_synthesis.methods.baseline import BaselineMethod
from parallel_synthesis.methods.text_mas import TextMASMethod
from parallel_synthesis.models import ModelWrapper
from parallel_synthesis.prompts import PRETRAINING_TASKS
from parallel_synthesis.utils.eval_runner import evaluate, run_eval, summarize_latency_metrics
from parallel_synthesis.utils.utils import auto_device, evaluate_rouge, set_seed
import time

def build_method(model: ModelWrapper, args: argparse.Namespace):
    common_kwargs = dict(
        temperature=args.temperature,
        top_p=args.top_p,
    )
    if args.method == "baseline":
        return BaselineMethod(
            model,
            max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            use_vllm=args.use_vllm,
            args=args,
        )
    if args.method == "text_mas":
        return TextMASMethod(
            model,
            max_new_tokens_each=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            args=args,
        )
    raise ValueError(f"Unsupported method: {args.method}")


def main():
    parser = argparse.ArgumentParser()

    # core args for experiments
    parser.add_argument("--method", choices=["baseline", "text_mas"], required=True,
                        help="Which inference method to run.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Hugging Face model id to use for experiments (e.g. 'Qwen/Qwen3-14B').")
    parser.add_argument("--max_samples", type=int, default=-1, help="Number of questions to evaluate; set -1 to use all samples.")
    parser.add_argument(
        "--task",
        choices=[
            "gsm8k",
            "aime2024",
            "aime2025",
            "gpqa",
            "mbppplus",
            "humanevalplus",
            "medqa",
        ],
        default="gsm8k",
        help="Dataset/task to evaluate. Controls which loader is used.",
    )

    # other args
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
        help="Keep frozen embeddings in the selected 4-bit compute dtype.",
    )
    parser.add_argument(
        "--attn_implementation",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
        default="sdpa",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--generate_bs", type=int, default=20, help="Batch size for generation")
    parser.add_argument("--text_mas_context_length", type=int, default=-1, help="TextMAS context length limit")
    parser.add_argument(
        "--parallel_kv_num_parallel_agents",
        type=int,
        default=3,
        help=(
            "Number of worker agents to launch for text_mas."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_dir", type=str, default="example_logs", help="Optional directory to write JSONL predictions + summary JSON.")
    parser.add_argument("--run_name", type=str, default="", help="Optional run name for log folder.")

    # # vLLM support
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM backend for generation")
    parser.add_argument("--enable_prefix_caching", action="store_true", help="Enable prefix caching in vLLM when supported.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="How many GPUs vLLM should shard the model across")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="Target GPU memory utilization for vLLM")

    args = parser.parse_args()

    if args.method == "text_mas" and args.parallel_kv_num_parallel_agents <= 0:
        raise ValueError("--parallel_kv_num_parallel_agents must be > 0.")

    if args.use_vllm and args.method not in {"baseline", "text_mas"}:
        raise ValueError("--use_vllm is currently only supported for baseline and text_mas inference.")

    set_seed(args.seed)
    device = auto_device(args.device)
    start_time = time.time()

    model = ModelWrapper(args.model_name, device, use_vllm=args.use_vllm, args=args)
    method = build_method(model, args)
    dataset_iter = build_dataset_iter(args.task, split=args.split, seed=args.seed, args=args)

    log_fh: Optional[TextIO] = None
    preds_path: Optional[str] = None
    summary_path: Optional[str] = None
    run_args_path: Optional[str] = None
    if args.log_dir:
        safe_model = args.model_name.replace("/", "-")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_name = args.run_name or f"{args.method}_{args.task}_{safe_model}_max{args.max_new_tokens}_{timestamp}"
        out_dir = os.path.join(args.log_dir, run_name)
        os.makedirs(out_dir, exist_ok=True)
        preds_path = os.path.join(out_dir, "preds.jsonl")
        summary_path = os.path.join(out_dir, "summary.json")
        run_args_path = os.path.join(out_dir, "run_args.json")
        log_fh = open(preds_path, "w", encoding="utf-8")
        with open(run_args_path, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=2, sort_keys=True)

    preds, eval_samples = run_eval(method, dataset_iter, args, log_fh=log_fh)
    total_time = time.time() - start_time
    acc, correct = evaluate(preds)

    summary = {
        "method": args.method,
        "model": args.model_name,
        "split": args.split,
        "seed": args.seed,
        "max_samples": eval_samples,
        "accuracy": acc,
        "correct": correct,
        "total_time_sec": round(total_time, 4),
        "time_per_sample_sec": round(total_time / eval_samples, 4) if eval_samples > 0 else None,
    }
    summary.update(summarize_latency_metrics(preds))
    if args.task in PRETRAINING_TASKS:
        summary.update(evaluate_rouge(preds))

    print(json.dumps(summary, ensure_ascii=False))

    if log_fh is not None:
        log_fh.close()
    if args.log_dir and summary_path is not None and preds_path is not None:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logged_files = [preds_path, summary_path]
        if run_args_path is not None:
            logged_files.append(run_args_path)
        print(f"[log] wrote {', '.join(logged_files)}")



if __name__ == "__main__":
    main()
