import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from parallel_synthesis.prompts import PRETRAINING_TASKS
from parallel_synthesis.utils.eval_runner import evaluate as evaluate_exact_match
from parallel_synthesis.utils.eval_runner import summarize_latency_metrics


_ROUGE_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9@_]+")


def _rouge_tokenize(text: str) -> List[str]:
    return _ROUGE_TOKEN_PATTERN.findall(str(text).lower())


def _ngram_counts(tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
    counts: Dict[Tuple[str, ...], int] = {}
    if n <= 0 or len(tokens) < n:
        return counts
    for idx in range(len(tokens) - n + 1):
        key = tuple(tokens[idx : idx + n])
        counts[key] = counts.get(key, 0) + 1
    return counts


def _f1_from_counts(overlap: int, pred_total: int, ref_total: int) -> float:
    if pred_total == 0 or ref_total == 0 or overlap <= 0:
        return 0.0
    precision = overlap / pred_total
    recall = overlap / ref_total
    denom = precision + recall
    return (2.0 * precision * recall / denom) if denom > 0 else 0.0


def _rouge_n_f1(pred_tokens: List[str], ref_tokens: List[str], n: int) -> float:
    pred_counts = _ngram_counts(pred_tokens, n)
    ref_counts = _ngram_counts(ref_tokens, n)
    if not pred_counts or not ref_counts:
        return 0.0
    overlap = 0
    for key, pred_count in pred_counts.items():
        overlap += min(pred_count, ref_counts.get(key, 0))
    return _f1_from_counts(overlap, sum(pred_counts.values()), sum(ref_counts.values()))


def _lcs_len(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    if len(a) < len(b):
        short, long_ = a, b
    else:
        short, long_ = b, a
    prev = [0] * (len(short) + 1)
    for tok in long_:
        curr = [0] * (len(short) + 1)
        for idx in range(1, len(short) + 1):
            if tok == short[idx - 1]:
                curr[idx] = prev[idx - 1] + 1
            else:
                curr[idx] = prev[idx] if prev[idx] >= curr[idx - 1] else curr[idx - 1]
        prev = curr
    return prev[-1]


def _rouge_l_f1(pred_tokens: List[str], ref_tokens: List[str]) -> float:
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_len(pred_tokens, ref_tokens)
    return _f1_from_counts(lcs, len(pred_tokens), len(ref_tokens))


def evaluate_rouge(preds: List[Dict[str, Any]]) -> Dict[str, float]:
    rouge1_scores: List[float] = []
    rouge2_scores: List[float] = []
    rougel_scores: List[float] = []

    for row in preds:
        gold = str(row.get("gold", "")).strip()
        pred_text = str(row.get("prediction", "")).strip()
        if not pred_text:
            pred_text = str(row.get("raw_prediction", "")).strip()
        if not gold:
            continue
        pred_tokens = _rouge_tokenize(pred_text)
        ref_tokens = _rouge_tokenize(gold)
        rouge1_scores.append(_rouge_n_f1(pred_tokens, ref_tokens, 1))
        rouge2_scores.append(_rouge_n_f1(pred_tokens, ref_tokens, 2))
        rougel_scores.append(_rouge_l_f1(pred_tokens, ref_tokens))

    denom = len(rouge1_scores)
    if denom == 0:
        return {
            "rouge_samples": 0,
            "rouge1_f1": 0.0,
            "rouge2_f1": 0.0,
            "rougel_f1": 0.0,
        }
    return {
        "rouge_samples": denom,
        "rouge1_f1": sum(rouge1_scores) / denom,
        "rouge2_f1": sum(rouge2_scores) / denom,
        "rougel_f1": sum(rougel_scores) / denom,
    }


def _load_json_object(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        loaded = json.load(fh)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(loaded).__name__}.")
    return loaded


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            loaded = json.loads(line)
            if not isinstance(loaded, dict):
                raise ValueError(f"Expected JSON object in {path}, got {type(loaded).__name__}.")
            rows.append(loaded)
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_post_eval_preds_name(name: str) -> Optional[Tuple[str, str]]:
    prefix = "post_eval_"
    suffix = "_preds.jsonl"
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    middle = name[len(prefix) : -len(suffix)]
    if "_" not in middle:
        return None
    split, task = middle.split("_", 1)
    split = str(split).strip()
    task = str(task).strip()
    if not split or not task:
        return None
    return split, task


def _append_unique(items: List[str], seen: set[str], value: str) -> None:
    normalized = str(value).strip()
    if not normalized or normalized in seen:
        return
    seen.add(normalized)
    items.append(normalized)


def _summarize_task_preds(
    task: str,
    split: str,
    preds: List[Dict[str, Any]],
    *,
    total_time_sec: Optional[float],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "task": task,
        "split": split,
        "samples": len(preds),
    }
    if task in PRETRAINING_TASKS:
        summary["accuracy"] = None
        summary["correct"] = None
        summary["eval_mode"] = "rouge_only"
        summary.update(evaluate_rouge(preds))
    else:
        accuracy, correct = evaluate_exact_match(preds)
        summary["accuracy"] = accuracy
        summary["correct"] = correct
    summary.update(summarize_latency_metrics(preds))
    summary["total_time_sec"] = round(total_time_sec, 4) if total_time_sec is not None else None
    summary["time_per_sample_sec"] = (
        round(total_time_sec / len(preds), 4)
        if total_time_sec is not None and preds
        else None
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge sharded post_eval runs.")
    parser.add_argument("--run_dir", type=str, required=True, help="Base post_eval run directory.")
    parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any shard directory or expected preds file is missing.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    num_shards = max(1, int(args.num_shards))
    if num_shards <= 1:
        raise ValueError("num_shards must be > 1 for merging.")

    shards_root = run_dir / "shards"
    if not shards_root.exists():
        raise FileNotFoundError(f"Missing shards directory: {shards_root}")

    shard_summaries: List[Dict[str, Any]] = []
    shard_run_args: List[Dict[str, Any]] = []
    shard_dirs_present: List[Path] = []
    missing_shard_dirs: List[str] = []
    tasks: List[str] = []
    seen_tasks: set[str] = set()
    split: Optional[str] = None

    for shard_id in range(num_shards):
        shard_dir = shards_root / f"shard{shard_id:02d}of{num_shards:02d}"
        if not shard_dir.exists():
            missing_shard_dirs.append(str(shard_dir))
            continue
        shard_dirs_present.append(shard_dir)

        summary = _load_json_object(shard_dir / "post_eval_summary.json")
        run_args = _load_json_object(shard_dir / "run_args.json")
        if summary:
            shard_summaries.append(summary)
        if run_args:
            shard_run_args.append(run_args)

        candidate_split = str(summary.get("split", "") or run_args.get("split", "")).strip()
        if candidate_split:
            if split is None:
                split = candidate_split
            elif split != candidate_split:
                raise ValueError(f"Inconsistent split across shards: {split!r} vs {candidate_split!r}")

        for task in summary.get("tasks", []) or []:
            _append_unique(tasks, seen_tasks, str(task))

        for preds_path in sorted(shard_dir.glob("post_eval_*_preds.jsonl")):
            parsed = _parse_post_eval_preds_name(preds_path.name)
            if parsed is None:
                continue
            parsed_split, parsed_task = parsed
            if split is None:
                split = parsed_split
            elif split != parsed_split:
                raise ValueError(f"Inconsistent split across shard preds: {split!r} vs {parsed_split!r}")
            _append_unique(tasks, seen_tasks, parsed_task)

    if args.strict and missing_shard_dirs:
        missing = "\n".join(missing_shard_dirs)
        raise FileNotFoundError(f"Missing shard directories:\n{missing}")

    if not shard_dirs_present:
        raise FileNotFoundError(f"No shard directories found under {shards_root}")
    if not split:
        raise ValueError("Could not infer split from shard outputs.")
    if not tasks:
        raise ValueError("Could not infer task list from shard outputs.")

    missing_pred_files: List[str] = []
    merged_preds_by_task: Dict[str, List[Dict[str, Any]]] = {task: [] for task in tasks}

    for shard_id in range(num_shards):
        shard_dir = shards_root / f"shard{shard_id:02d}of{num_shards:02d}"
        if not shard_dir.exists():
            continue
        for task in tasks:
            preds_path = shard_dir / f"post_eval_{split}_{task}_preds.jsonl"
            if not preds_path.exists():
                missing_pred_files.append(str(preds_path))
                continue
            merged_preds_by_task[task].extend(read_jsonl(preds_path))

    if args.strict and missing_pred_files:
        missing = "\n".join(missing_pred_files)
        raise FileNotFoundError(f"Missing shard prediction files:\n{missing}")

    first_summary = shard_summaries[0] if shard_summaries else {}
    first_run_args = shard_run_args[0] if shard_run_args else {}

    per_task_time_sums: Dict[str, float] = {}
    rows_before_shard_by_task: Dict[str, int] = {}
    rows_after_merge_by_task: Dict[str, int] = {}
    report_total_time_sec = 0.0
    report_time_found = False
    summary_total_time_sec = 0.0
    summary_time_found = False

    for summary in shard_summaries:
        report = summary.get("report", {}) if isinstance(summary.get("report"), dict) else {}
        by_task = report.get("by_task", {}) if isinstance(report.get("by_task"), dict) else {}
        report_time = report.get("total_time_sec")
        if report_time is not None:
            report_total_time_sec += float(report_time)
            report_time_found = True
        summary_time = summary.get("time_sec")
        if summary_time is not None:
            summary_total_time_sec += float(summary_time)
            summary_time_found = True

        before_counts = summary.get("rows_before_shard_by_task", {})
        if isinstance(before_counts, dict):
            for task, count in before_counts.items():
                rendered_task = str(task).strip()
                if not rendered_task:
                    continue
                rows_before_shard_by_task[rendered_task] = max(
                    int(count),
                    int(rows_before_shard_by_task.get(rendered_task, 0)),
                )

        for task, task_summary in by_task.items():
            if not isinstance(task_summary, dict):
                continue
            total_time = task_summary.get("total_time_sec")
            if total_time is None:
                continue
            per_task_time_sums[str(task)] = per_task_time_sums.get(str(task), 0.0) + float(total_time)

    by_task_summary: Dict[str, Any] = {}
    all_preds: List[Dict[str, Any]] = []
    total_samples = 0
    total_correct = 0
    exact_metric_samples = 0
    exact_metric_correct = 0

    for task in tasks:
        preds = merged_preds_by_task.get(task, [])
        rows_after_merge_by_task[task] = len(preds)
        summary = _summarize_task_preds(
            task,
            split,
            preds,
            total_time_sec=per_task_time_sums.get(task),
        )
        by_task_summary[task] = summary
        all_preds.extend(preds)
        total_samples += int(summary["samples"])
        correct = summary.get("correct")
        if correct is not None:
            total_correct += int(correct)
            exact_metric_samples += int(summary["samples"])
            exact_metric_correct += int(correct)

        out_preds_path = run_dir / f"post_eval_{split}_{task}_preds.jsonl"
        write_jsonl(out_preds_path, preds)

    report: Dict[str, Any] = {
        "split": split,
        "tasks": tasks,
        "total_samples": total_samples,
        "total_correct": total_correct,
        "exact_metric_samples": exact_metric_samples,
        "exact_metric_correct": exact_metric_correct,
        "micro_accuracy": (
            exact_metric_correct / exact_metric_samples
            if exact_metric_samples > 0
            else None
        ),
        "by_task": by_task_summary,
    }
    report.update(summarize_latency_metrics(all_preds))
    report["total_time_sec"] = round(report_total_time_sec, 4) if report_time_found else None
    report["time_per_sample_sec"] = (
        round(report_total_time_sec / total_samples, 4)
        if report_time_found and total_samples > 0
        else None
    )

    total_rows_before_shard = first_summary.get("total_rows_before_shard")
    if total_rows_before_shard is None:
        total_rows_before_shard = sum(rows_before_shard_by_task.values()) or None

    merged_summary: Dict[str, Any] = {
        "mode": str(first_summary.get("mode", "post_eval_parallel_kv")),
        "method": str(first_summary.get("method", first_run_args.get("method", ""))),
        "model_name": str(first_summary.get("model_name", first_run_args.get("model_name", ""))),
        "tasks": tasks,
        "split": split,
        "eval_samples_per_task": first_summary.get(
            "eval_samples_per_task",
            first_run_args.get("eval_samples_per_task"),
        ),
        "checkpoint_dir": str(first_summary.get("checkpoint_dir", first_run_args.get("checkpoint_dir", ""))),
        "run_args_path": str(first_summary.get("run_args_path", first_run_args.get("resolved_run_args_path", ""))),
        "load_components": str(first_summary.get("load_components", first_run_args.get("load_components", ""))),
        "load_components_source": str(
            first_summary.get(
                "load_components_source",
                first_run_args.get("load_components_source", ""),
            )
        ),
        "load_affine_map": bool(
            first_summary.get(
                "load_affine_map",
                first_run_args.get("parallel_kv_enable_affine_map", False),
            )
        ),
        "load_judger_lora": bool(
            first_summary.get(
                "load_judger_lora",
                first_run_args.get("parallel_kv_enable_judger_lora", False),
            )
        ),
        "run_name": run_dir.name,
        "base_output_dir": str(run_dir),
        "output_dir": str(run_dir),
        "num_shards": num_shards,
        "merged_shards_found": len(shard_dirs_present),
        "missing_shards": missing_shard_dirs,
        "missing_pred_files": missing_pred_files,
        "total_rows_before_shard": total_rows_before_shard,
        "total_rows_after_shard": total_samples,
        "rows_before_shard_by_task": rows_before_shard_by_task,
        "rows_after_shard_by_task": rows_after_merge_by_task,
        "rows_after_merge_by_task": rows_after_merge_by_task,
        "report": report,
        "merged_from_shards": True,
        "merge_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "time_sec": round(summary_total_time_sec, 4) if summary_time_found else None,
    }

    summary_path = run_dir / "post_eval_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(merged_summary, fh, ensure_ascii=False, indent=2)

    print(f"Merged summary: {summary_path}")
    print(f"Tasks merged: {', '.join(tasks)}")
    print(json.dumps(merged_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
