#!/usr/bin/env python3
"""
Analyze MARBLE DB benchmark logs and generate an HTML report.

This reuses the tool-calling step timeline view so MARBLE DB runs can be
inspected the same way as toolcall benchmark runs: exact prompts, exact
assistant outputs, per-step tool requests/responses, and final judger traces.
"""

import argparse
import importlib.util
import json
import re
import sys
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_toolcall_analyzer_module():
    module_path = REPO_ROOT / "toolcall" / "analyze_logs.py"
    spec = importlib.util.spec_from_file_location("toolcall_analyze_logs_standalone", str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load analyzer module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TOOLCALL_ANALYZER = _load_toolcall_analyzer_module()
_build_toolcall_html = _TOOLCALL_ANALYZER._build_html
_build_token_counter = _TOOLCALL_ANALYZER._build_token_counter
_compute_stats = _TOOLCALL_ANALYZER._compute_stats
_load_json = _TOOLCALL_ANALYZER._load_json
_load_preds = _TOOLCALL_ANALYZER._load_preds
_sample_tool_count = _TOOLCALL_ANALYZER._sample_tool_count


ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
THINK_BLOCK_RE = re.compile(r"<think>[\s\S]*?</think>", flags=re.IGNORECASE)


def _detect_metadata_path(explicit_path: str, candidate: Path) -> Optional[Path]:
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    if candidate.exists():
        return candidate
    return None


def _summary_and_run_args_block(
    *,
    summary: Optional[Dict[str, Any]],
    run_args: Optional[Dict[str, Any]],
    max_chars: int,
) -> str:
    blocks = []
    if summary:
        rendered = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        if max_chars > 0 and len(rendered) > max_chars:
            rendered = rendered[:max_chars] + "\n... [truncated]"
        blocks.append(
            "<section class='card'>"
            "<h2>summary.json</h2>"
            f"<pre>{escape(rendered)}</pre>"
            "</section>"
        )
    if run_args:
        rendered = json.dumps(run_args, ensure_ascii=False, indent=2, sort_keys=True)
        if max_chars > 0 and len(rendered) > max_chars:
            rendered = rendered[:max_chars] + "\n... [truncated]"
        blocks.append(
            "<section class='card'>"
            "<h2>run_args.json</h2>"
            f"<pre>{escape(rendered)}</pre>"
            "</section>"
        )
    return "\n".join(blocks)


def _normalize_final_answer_text(value: Any) -> str:
    text = str(value or "")
    text = ZERO_WIDTH_RE.sub("", text)
    text = THINK_BLOCK_RE.sub("", text)
    return text.strip()


def _has_meaningful_final_answer(record: Dict[str, Any]) -> bool:
    predicted = record.get("predicted_root_causes", []) or record.get("prediction", []) or []
    if isinstance(predicted, list) and len(predicted) > 0:
        return True

    raw_prediction = _normalize_final_answer_text(record.get("raw_prediction", ""))
    if raw_prediction:
        return True

    for agent in reversed(record.get("agents", []) or []):
        role = str(agent.get("role", "")).strip().lower()
        name = str(agent.get("name", "")).strip().lower()
        if role == "judger" or name == "judger":
            judger_text = _normalize_final_answer_text(
                agent.get("final_answer", "") or agent.get("output", "")
            )
            return bool(judger_text)
    return False


def _bucket_name(record: Dict[str, Any]) -> str:
    if bool(record.get("correct", False)):
        return "correct"
    if _has_meaningful_final_answer(record):
        return "wrong_with_answer"
    return "no_meaningful_answer"


def _bucket_rows(rows: List[Tuple[int, Dict[str, Any]]]) -> Dict[str, List[Tuple[int, Dict[str, Any]]]]:
    buckets: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {
        "correct": [],
        "no_meaningful_answer": [],
        "wrong_with_answer": [],
    }
    for row in rows:
        _, rec = row
        buckets[_bucket_name(rec)].append(row)
    return buckets


def _retitle_html(
    html_doc: str,
    *,
    source_path: Path,
    summary: Optional[Dict[str, Any]],
    run_args: Optional[Dict[str, Any]],
    max_chars: int,
) -> str:
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_header = (
        "<h1>MARBLE DB Log Analysis</h1>\n"
        f"  <p><strong>Source:</strong> {escape(str(source_path))}</p>\n"
        f"  <p><strong>Generated:</strong> {escape(generated)}</p>\n"
        "  <p class=\"muted\">Each worker and judger block below shows the exact saved prompts, "
        "outputs, and tool interactions from <code>preds.jsonl</code>.</p>\n"
    )
    html_doc = html_doc.replace("<title>Tool-Call Log Analysis</title>", "<title>MARBLE DB Log Analysis</title>")
    html_doc = re.sub(
        r"<h1>Tool-Call Benchmark Log Analysis</h1>\s*"
        r"<p><strong>Source:</strong>.*?</p>\s*"
        r"<p><strong>Generated:</strong>.*?</p>\s*"
        r"<p class=\"muted\">.*?</p>\s*",
        new_header,
        html_doc,
        count=1,
        flags=re.DOTALL,
    )

    extra_block = _summary_and_run_args_block(
        summary=summary,
        run_args=run_args,
        max_chars=max_chars,
    )
    if extra_block:
        html_doc = html_doc.replace("{summary_block_placeholder}", extra_block)
    else:
        html_doc = html_doc.replace("{summary_block_placeholder}", "")
    return html_doc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze MARBLE DB logs and generate an HTML report."
    )
    parser.add_argument("--run_dir", type=str, default="", help="Run directory containing preds.jsonl.")
    parser.add_argument("--preds_path", type=str, default="", help="Optional path to preds.jsonl.")
    parser.add_argument("--summary_path", type=str, default="", help="Optional path to summary.json.")
    parser.add_argument("--run_args_path", type=str, default="", help="Optional path to run_args.json.")
    parser.add_argument("--out_html", type=str, default="", help="Optional output HTML path.")
    parser.add_argument("--max_chars", type=int, default=-1, help="Max chars per displayed text block. -1 means no truncation.")
    parser.add_argument("--limit", type=int, default=-1, help="Optional max number of samples to include.")
    parser.add_argument("--only_with_calls", action="store_true", help="Only include samples with at least one tool call.")
    parser.add_argument("--only_wrong", action="store_true", help="Only include incorrect samples.")
    parser.add_argument("--only_correct", action="store_true", help="Only include correct samples.")
    parser.add_argument(
        "--only_no_meaningful_answer",
        action="store_true",
        help="Only include samples whose final judger output does not contain a meaningful answer.",
    )
    parser.add_argument(
        "--only_wrong_with_answer",
        action="store_true",
        help="Only include incorrect samples that still emit a meaningful final answer.",
    )
    parser.add_argument("--include_prompts", action="store_true", help="Include full saved agent prompts.")
    parser.add_argument("--tokenizer_name", type=str, default="", help="Optional tokenizer name/path for token metrics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bucket_flags = [
        bool(args.only_wrong),
        bool(args.only_correct),
        bool(args.only_no_meaningful_answer),
        bool(args.only_wrong_with_answer),
    ]
    if sum(bucket_flags) > 1:
        raise ValueError(
            "--only_wrong, --only_correct, --only_no_meaningful_answer, and "
            "--only_wrong_with_answer are mutually exclusive."
        )

    run_dir: Optional[Path] = None
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
        if not run_dir.exists() or not run_dir.is_dir():
            raise FileNotFoundError(f"run dir not found: {run_dir}")

    if args.preds_path:
        preds_path = Path(args.preds_path).expanduser().resolve()
    elif run_dir is not None:
        preds_path = run_dir / "preds.jsonl"
    else:
        raise ValueError("Provide either --run_dir or --preds_path.")

    if not preds_path.exists():
        raise FileNotFoundError(f"preds file not found: {preds_path}")

    summary_path = _detect_metadata_path(args.summary_path, preds_path.parent / "summary.json")
    run_args_path = _detect_metadata_path(args.run_args_path, preds_path.parent / "run_args.json")
    summary = _load_json(summary_path) if summary_path else None
    run_args = _load_json(run_args_path) if run_args_path else None

    token_counter, token_counter_mode = _build_token_counter(args.tokenizer_name)
    rows = _load_preds(preds_path)
    full_stats = _compute_stats(rows, token_counter=token_counter)
    full_stats["token_counter_mode"] = token_counter_mode
    bucketed_rows = _bucket_rows(rows)

    displayed = rows
    if args.only_with_calls:
        displayed = [(rid, rec) for rid, rec in displayed if _sample_tool_count(rec) > 0]
    if args.only_wrong:
        displayed = [(rid, rec) for rid, rec in displayed if not bool(rec.get("correct", False))]
    if args.only_correct:
        displayed = [(rid, rec) for rid, rec in displayed if bool(rec.get("correct", False))]
    if args.only_no_meaningful_answer:
        displayed = [(rid, rec) for rid, rec in displayed if _bucket_name(rec) == "no_meaningful_answer"]
    if args.only_wrong_with_answer:
        displayed = [(rid, rec) for rid, rec in displayed if _bucket_name(rec) == "wrong_with_answer"]
    if args.limit is not None and args.limit >= 0:
        displayed = displayed[: args.limit]

    displayed_stats = _compute_stats(displayed, token_counter=token_counter)
    displayed_stats["token_counter_mode"] = token_counter_mode

    out_html = (
        Path(args.out_html).expanduser().resolve()
        if args.out_html
        else preds_path.parent / "marble_db_analysis.html"
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)

    html_doc = _build_toolcall_html(
        source_path=preds_path,
        displayed_rows=displayed,
        full_stats=full_stats,
        displayed_stats=displayed_stats,
        summary=None,
        max_chars=int(args.max_chars),
        include_prompts=bool(args.include_prompts),
    )
    html_doc = html_doc.replace(
        "<section class='card'>\n    <h2>Samples</h2>",
        "{summary_block_placeholder}\n\n  <section class='card'>\n    <h2>Samples</h2>",
        1,
    )
    html_doc = _retitle_html(
        html_doc,
        source_path=preds_path,
        summary=summary,
        run_args=run_args,
        max_chars=int(args.max_chars),
    )
    out_html.write_text(html_doc, encoding="utf-8")

    print(f"[marble_db_analysis] wrote {out_html}")
    print("Loaded samples:", full_stats["samples"])
    print("Bucket counts:", {key: len(value) for key, value in bucketed_rows.items()})
    print("Displayed samples:", displayed_stats["samples"])
    print("Displayed total tool calls:", displayed_stats["total_tool_calls"])
    print("Displayed accuracy:", f"{displayed_stats['accuracy']:.4f}")


if __name__ == "__main__":
    main()
