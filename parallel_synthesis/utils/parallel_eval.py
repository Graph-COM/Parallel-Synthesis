import json
import os
import time
from typing import Any, Dict, List, Tuple

from parallel_synthesis.prompts import PRETRAINING_TASKS
from parallel_synthesis.utils.eval_runner import evaluate, summarize_latency_metrics
from parallel_synthesis.utils.utils import evaluate_rouge


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl_strict(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                loaded = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSONL at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(loaded, dict):
                raise ValueError(
                    f"Expected a JSON object at {path}:{line_number}, "
                    f"got {type(loaded).__name__}"
                )
            rows.append(loaded)
    return rows


def print_eval_result(
    res: Dict[str, Any],
    *,
    task: str,
    split: str,
    phase: str,
    sample_idx: int,
) -> None:
    def _head_tokens(text: Any, limit: int = 50) -> str:
        rendered = str(text or "").strip()
        if not rendered:
            return ""
        pieces = rendered.split()
        clipped = " ".join(pieces[:limit])
        clipped = clipped.replace("\n", "\\n")
        if len(pieces) <= limit:
            return clipped
        return clipped + " ... [truncated]"

    print(
        f"\n==================== {phase} | split={split} | task={task} | sample={sample_idx} ===================="
    )
    print("Question:")
    print(str(res.get("question", "")).strip())
    agents = res.get("agents", [])
    fixed_agents = [agent for agent in agents if str(agent.get("role", "")).strip() == "fixed_cache"]
    judger_agent = next(
        (agent for agent in reversed(agents) if str(agent.get("role", "")).strip() == "judger"),
        None,
    )

    if fixed_agents:
        print(f"sample={sample_idx} task={task} split={split} phase={phase}")
        for idx, agent in enumerate(fixed_agents, start=1):
            print(
                f"parallel_chunk_{idx}: "
                f"prefill={_head_tokens(agent.get('input', ''), limit=50)} | "
                f"extract={_head_tokens(agent.get('extract', ''), limit=50)}"
            )
        if judger_agent is not None:
            print(f"judger_prompt: {_head_tokens(judger_agent.get('input', ''), limit=50)}")
            print(f"judger_output: {str(judger_agent.get('output', '')).rstrip()}")
        print(
            f"Result: Pred={res.get('prediction')} | Gold={res.get('gold')} | OK={res.get('correct')}"
        )
        return

    for agent in agents:
        name = agent.get("name", "Agent")
        role = agent.get("role", "")
        print(f"----- Agent: {name} ({role}) -----")
        print("[To Tokenize]")
        print(str(agent.get("input", "")).rstrip())
        print("[Output]")
        print(str(agent.get("output", "")).rstrip())
        print("----------------------------------------------")
    print(
        f"Result: Pred={res.get('prediction')} | Gold={res.get('gold')} | OK={res.get('correct')}"
    )


def evaluate_task(
    method: Any,
    args,
    task: str,
    rows: List[Dict[str, Any]],
    *,
    split: str,
    phase: str,
    preds_path: str | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    started_at = time.time()
    if hasattr(method, "task"):
        method.task = task
    if getattr(method, "args", None) is not None:
        method.args.task = task
    if hasattr(method, "model") and hasattr(method.model, "model"):
        method.model.model.eval()
    if hasattr(method, "cache_mapper") and method.cache_mapper is not None:
        method.cache_mapper.eval()

    resume_existing = bool(getattr(args, "resume_existing_preds", False))
    existing_preds: List[Dict[str, Any]] = []
    if preds_path and resume_existing and os.path.isfile(preds_path):
        existing_preds = read_jsonl_strict(preds_path)
    elif preds_path:
        write_jsonl(preds_path, [])

    preds: List[Dict[str, Any]] = list(existing_preds)
    sample_offset = len(existing_preds)
    batch_size = max(1, int(args.generate_bs))
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        if not batch:
            continue
        results = method.run_batch(batch)
        if preds_path:
            append_jsonl(preds_path, results)
        for offset, res in enumerate(results):
            preds.append(res)
            print_eval_result(
                res,
                task=task,
                split=split,
                phase=phase,
                sample_idx=sample_offset + start + offset + 1,
            )

    if task in PRETRAINING_TASKS:
        rouge = evaluate_rouge(preds)
        summary = {
            "task": task,
            "split": split,
            "samples": len(preds),
            "accuracy": None,
            "correct": None,
            "eval_mode": "rouge_only",
            "resumed_existing_samples": sample_offset,
            "new_samples": len(rows),
            **rouge,
        }
        elapsed = time.time() - started_at
        summary.update(summarize_latency_metrics(preds))
        summary["total_time_sec"] = round(elapsed, 4)
        summary["time_per_sample_sec"] = round(elapsed / len(rows), 4) if rows else None
        print(
            f"[{phase}] split={split} task={task} samples={len(rows)} "
            f"rouge1_f1={rouge['rouge1_f1']:.4f} rouge2_f1={rouge['rouge2_f1']:.4f} "
            f"rougel_f1={rouge['rougel_f1']:.4f}"
        )
    else:
        acc, correct = evaluate(preds)
        summary = {
            "task": task,
            "split": split,
            "samples": len(preds),
            "accuracy": acc,
            "correct": correct,
            "resumed_existing_samples": sample_offset,
            "new_samples": len(rows),
        }
        elapsed = time.time() - started_at
        summary.update(summarize_latency_metrics(preds))
        summary["total_time_sec"] = round(elapsed, 4)
        summary["time_per_sample_sec"] = round(elapsed / len(rows), 4) if rows else None
        print(
            f"[{phase}] split={split} task={task} samples={len(rows)} accuracy={acc:.4f} correct={correct}"
        )
    return preds, summary


def evaluate_all_tasks(
    method: Any,
    args,
    rows_by_task: Dict[str, List[Dict[str, Any]]],
    *,
    split: str,
    output_dir: str,
    prefix: str,
) -> Dict[str, Any]:
    started_at = time.time()
    by_task: Dict[str, Any] = {}
    all_preds: List[Dict[str, Any]] = []
    total_samples = 0
    total_correct = 0
    exact_metric_samples = 0
    exact_metric_correct = 0

    for task, rows in rows_by_task.items():
        preds_path = os.path.join(output_dir, f"{prefix}_{split}_{task}_preds.jsonl")
        preds, summary = evaluate_task(
            method,
            args,
            task,
            rows,
            split=split,
            phase=prefix,
            preds_path=preds_path,
        )
        by_task[task] = summary
        all_preds.extend(preds)
        total_samples += int(summary["samples"])
        correct = summary.get("correct")
        if correct is not None:
            total_correct += int(correct)
            exact_metric_samples += int(summary["samples"])
            exact_metric_correct += int(correct)

    report = {
        "split": split,
        "tasks": list(rows_by_task.keys()),
        "total_samples": total_samples,
        "total_correct": total_correct,
        "exact_metric_samples": exact_metric_samples,
        "exact_metric_correct": exact_metric_correct,
        "micro_accuracy": (
            exact_metric_correct / exact_metric_samples
            if exact_metric_samples > 0
            else None
        ),
        "by_task": by_task,
    }
    elapsed = time.time() - started_at
    report.update(summarize_latency_metrics(all_preds))
    report["total_time_sec"] = round(elapsed, 4)
    report["time_per_sample_sec"] = round(elapsed / total_samples, 4) if total_samples > 0 else None
    micro_accuracy = report["micro_accuracy"]
    if micro_accuracy is None:
        print(
            f"[{prefix}] split={split} total_samples={report['total_samples']} "
            "micro_accuracy=n/a (rouge-only tasks)"
        )
    else:
        print(
            f"[{prefix}] split={split} total_samples={report['total_samples']} "
            f"total_correct={report['total_correct']} micro_accuracy={micro_accuracy:.4f}"
        )
    return report
