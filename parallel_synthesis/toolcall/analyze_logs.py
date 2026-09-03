#!/usr/bin/env python3
"""
Render saved GAIA tool-calling results as an HTML report.

The report focuses on:
- Tool call requests (tool name + arguments)
- Tool call responses (stored preview in logs)
- Final answer per sample
"""

import argparse
import ast
import json
import re
from collections import Counter
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _build_token_counter(tokenizer_name: str) -> Tuple[Any, str]:
    name = str(tokenizer_name or "").strip()
    if name:
        try:
            from transformers import AutoTokenizer  # type: ignore

            tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)

            def _count(text: str) -> int:
                ids = tokenizer.encode(str(text), add_special_tokens=False)
                return int(len(ids))

            return _count, f"hf:{name}"
        except Exception:
            pass

    def _fallback_count(text: str) -> int:
        return len(re.findall(r"\S+", str(text)))

    return _fallback_count, "whitespace_fallback"


def _clip_text(text: Any, max_chars: int) -> str:
    s = "" if text is None else str(text)
    if max_chars > 0 and len(s) > max_chars:
        return s[:max_chars] + "\n... [truncated]"
    return s


def _pretty_json(obj: Any, max_chars: int) -> str:
    try:
        rendered = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)
    except TypeError:
        rendered = str(obj)
    return _clip_text(rendered, max_chars)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_preds(path: Path) -> List[Tuple[int, Dict[str, Any]]]:
    rows: List[Tuple[int, Dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            rows.append((i, json.loads(text)))
    return rows


def _iter_tool_calls(record: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for agent_idx, agent in enumerate(record.get("agents", []) or [], start=1):
        agent_name = str(agent.get("name", "") or f"agent_{agent_idx}")
        agent_role = str(agent.get("role", "") or "")
        for call_idx, call in enumerate(agent.get("tool_calls", []) or [], start=1):
            call_name = str(call.get("name", "") or "unknown")
            yield {
                "agent_name": agent_name,
                "agent_role": agent_role,
                "call_idx": call_idx,
                "name": call_name,
                "arguments": call.get("arguments", {}),
                "result_preview": call.get("result_preview", ""),
            }


def _sample_tool_count(record: Dict[str, Any]) -> int:
    count = 0
    for _ in _iter_tool_calls(record):
        count += 1
    return count


def _compute_stats(rows: List[Tuple[int, Dict[str, Any]]], token_counter: Any) -> Dict[str, Any]:
    total = len(rows)
    correct = sum(1 for _, rec in rows if bool(rec.get("correct", False)))
    tool_counter = Counter()
    role_counter = Counter()
    calls_per_sample: List[int] = []
    agent_output_token_lens: List[int] = []
    prompt_plus_output_token_lens: List[int] = []

    for _, rec in rows:
        per_sample = 0
        for call in _iter_tool_calls(rec):
            per_sample += 1
            tool_counter[call["name"]] += 1
            role_key = call["agent_role"] or call["agent_name"]
            role_counter[role_key] += 1
        calls_per_sample.append(per_sample)

        for agent in rec.get("agents", []) or []:
            output_text = str(agent.get("output", "")).strip()
            if not output_text:
                continue
            agent_output_token_lens.append(int(token_counter(output_text)))

            step_prompts = agent.get("step_prompts", []) or []
            if step_prompts:
                final_step_prompt = str(step_prompts[-1])
            else:
                final_step_prompt = str(agent.get("input", ""))
            combined_text = (final_step_prompt + "\n" + output_text).strip()
            prompt_plus_output_token_lens.append(int(token_counter(combined_text)))

    total_calls = sum(calls_per_sample)
    avg_calls = (total_calls / total) if total > 0 else 0.0
    max_calls = max(calls_per_sample) if calls_per_sample else 0
    samples_with_calls = sum(1 for x in calls_per_sample if x > 0)

    return {
        "samples": total,
        "correct": correct,
        "accuracy": (correct / total) if total > 0 else 0.0,
        "total_tool_calls": total_calls,
        "samples_with_tool_calls": samples_with_calls,
        "avg_tool_calls_per_sample": avg_calls,
        "max_tool_calls_in_sample": max_calls,
        "tool_counter": tool_counter,
        "role_counter": role_counter,
        "agent_outputs_measured": len(agent_output_token_lens),
        "avg_agent_final_output_tokens": (
            sum(agent_output_token_lens) / len(agent_output_token_lens)
            if agent_output_token_lens
            else 0.0
        ),
        "prompt_plus_output_measured": len(prompt_plus_output_token_lens),
        "avg_final_step_prompt_plus_output_tokens": (
            sum(prompt_plus_output_token_lens) / len(prompt_plus_output_token_lens)
            if prompt_plus_output_token_lens
            else 0.0
        ),
    }


def _render_summary_table(title: str, stats: Dict[str, Any]) -> str:
    return f"""
<section class="card">
  <h2>{escape(title)}</h2>
  <table class="kv">
    <tr><th>Samples</th><td>{stats['samples']}</td></tr>
    <tr><th>Correct</th><td>{stats['correct']}</td></tr>
    <tr><th>Accuracy</th><td>{stats['accuracy']:.4f}</td></tr>
    <tr><th>Total Tool Calls</th><td>{stats['total_tool_calls']}</td></tr>
    <tr><th>Samples With Tool Calls</th><td>{stats['samples_with_tool_calls']}</td></tr>
    <tr><th>Avg Calls / Sample</th><td>{stats['avg_tool_calls_per_sample']:.2f}</td></tr>
    <tr><th>Max Calls In One Sample</th><td>{stats['max_tool_calls_in_sample']}</td></tr>
    <tr><th>Measured Agent Outputs</th><td>{stats['agent_outputs_measured']}</td></tr>
    <tr><th>Avg Agent Final Output Tokens</th><td>{stats['avg_agent_final_output_tokens']:.2f}</td></tr>
    <tr><th>Measured Prompt+Output Pairs</th><td>{stats['prompt_plus_output_measured']}</td></tr>
    <tr><th>Avg Final-Step Prompt+Output Tokens</th><td>{stats['avg_final_step_prompt_plus_output_tokens']:.2f}</td></tr>
    <tr><th>Token Counter</th><td>{escape(str(stats.get('token_counter_mode', 'unknown')))}</td></tr>
  </table>
</section>
"""


def _render_counter_table(title: str, counter: Counter) -> str:
    rows = []
    for key, value in counter.most_common():
        rows.append(f"<tr><td>{escape(key)}</td><td>{value}</td></tr>")
    body = "\n".join(rows) if rows else '<tr><td colspan="2">None</td></tr>'
    return f"""
<section class="card">
  <h2>{escape(title)}</h2>
  <table class="simple">
    <tr><th>Name</th><th>Count</th></tr>
    {body}
  </table>
</section>
"""


TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
TOOL_RESPONSE_OPEN = "<tool_response>"
TOOL_RESPONSE_CLOSE = "</tool_response>"
CODE_OPEN = "<code>"
CODE_CLOSE = "</code>"


def _parse_json_relaxed(text: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    try:
        obj = ast.literal_eval(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    return None


def _extract_tag_content(text: str, open_tag: str, close_tag: str) -> Optional[str]:
    m = re.search(re.escape(open_tag) + r"([\s\S]*?)" + re.escape(close_tag), str(text), flags=re.IGNORECASE)
    if not m:
        return None
    return m.group(1).strip()


def _extract_tool_calls_from_assistant(text: str) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for m in re.finditer(
        re.escape(TOOL_CALL_OPEN) + r"([\s\S]*?)" + re.escape(TOOL_CALL_CLOSE),
        str(text),
        flags=re.IGNORECASE,
    ):
        payload = m.group(1).strip()
        payload_core = payload
        code = None
        cm = re.search(re.escape(CODE_OPEN) + r"([\s\S]*?)" + re.escape(CODE_CLOSE), payload, flags=re.IGNORECASE)
        if cm:
            code = cm.group(1).strip()
            payload_core = (payload[: cm.start()] + payload[cm.end() :]).strip()

        req = _parse_json_relaxed(payload_core)
        if req is None:
            calls.append({"name": "unparsed", "arguments": {}, "raw": payload})
            continue

        name = str(req.get("name", "")).strip() or "unknown"
        arguments = req.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {}
        if code:
            arguments = dict(arguments)
            arguments["code"] = code
        calls.append({"name": name, "arguments": arguments, "raw": payload})
    return calls


def _extract_tool_response_from_user(text: str) -> Optional[str]:
    content = _extract_tag_content(text, TOOL_RESPONSE_OPEN, TOOL_RESPONSE_CLOSE)
    if content is not None:
        return content
    if TOOL_RESPONSE_OPEN in str(text):
        return str(text).strip()
    return None


def _collect_agent_steps(agent: Dict[str, Any]) -> List[Dict[str, Any]]:
    messages = agent.get("messages", []) or []
    step_prompts = [str(x) for x in (agent.get("step_prompts", []) or [])]
    tool_logs = agent.get("tool_calls", []) or []

    steps: List[Dict[str, Any]] = []
    step_idx = 0
    tool_idx = 0

    for msg_idx, msg in enumerate(messages):
        if str(msg.get("role", "")) != "assistant":
            continue
        step_idx += 1
        assistant_text = str(msg.get("content", ""))
        reqs = _extract_tool_calls_from_assistant(assistant_text)

        item: Dict[str, Any] = {
            "step": step_idx,
            "prompt": step_prompts[step_idx - 1] if step_idx - 1 < len(step_prompts) else "",
            "assistant": assistant_text,
            "tool_requests": reqs,
            "tool_response_messages": [],
            "tool_logs": [],
            "next_prompt": step_prompts[step_idx] if step_idx < len(step_prompts) else "",
        }

        if reqs:
            for _ in reqs:
                if tool_idx < len(tool_logs):
                    item["tool_logs"].append(tool_logs[tool_idx])
                    tool_idx += 1

            j = msg_idx + 1
            while j < len(messages):
                nxt = messages[j]
                if str(nxt.get("role", "")) != "user":
                    break
                resp = _extract_tool_response_from_user(nxt.get("content", ""))
                if resp is not None:
                    item["tool_response_messages"].append(resp)
                j += 1

        steps.append(item)

    if not steps and step_prompts:
        for idx, prompt in enumerate(step_prompts, start=1):
            steps.append(
                {
                    "step": idx,
                    "prompt": prompt,
                    "assistant": "",
                    "tool_requests": [],
                    "tool_response_messages": [],
                    "tool_logs": [],
                    "next_prompt": step_prompts[idx] if idx < len(step_prompts) else "",
                }
            )

    return steps


def _render_step_timeline(agent: Dict[str, Any], max_chars: int) -> str:
    steps = _collect_agent_steps(agent)
    if not steps:
        return "<p class='muted'>No step timeline available (messages/step_prompts missing).</p>"

    parts: List[str] = []
    parts.append("<details open><summary>Step Timeline (Prompt -> Output -> Tool -> Next Prompt)</summary>")
    parts.append("<div class='steps'>")
    for step in steps:
        step_no = int(step.get("step", 0))
        prompt = _clip_text(step.get("prompt", ""), max_chars)
        assistant_text = _clip_text(step.get("assistant", ""), max_chars)
        reqs = step.get("tool_requests", []) or []
        responses = step.get("tool_response_messages", []) or []
        tool_logs = step.get("tool_logs", []) or []
        next_prompt = _clip_text(step.get("next_prompt", ""), max_chars)

        parts.append("<section class='step'>")
        parts.append(f"<h5>Step {step_no}</h5>")
        if prompt:
            parts.append("<details open><summary>Model Prompt (exact)</summary>" f"<pre>{escape(prompt)}</pre></details>")
        else:
            parts.append("<p class='muted'>No captured prompt for this step.</p>")

        if assistant_text:
            parts.append("<details><summary>Assistant Output</summary>" f"<pre>{escape(assistant_text)}</pre></details>")

        if reqs:
            parts.append(f"<p><strong>Tool Calls This Step:</strong> {len(reqs)}</p>")
            for idx, req in enumerate(reqs, start=1):
                req_brief = {"name": req.get("name", "unknown"), "arguments": req.get("arguments", {})}
                req_text = _pretty_json(req_brief, max_chars)
                parts.append(
                    "<details open><summary>Tool Call Request "
                    f"#{idx}</summary><pre>{escape(req_text)}</pre></details>"
                )

                if idx - 1 < len(responses):
                    response_in_msg = _clip_text(responses[idx - 1], max_chars)
                    parts.append(
                        "<details open><summary>Tool Response (from messages) "
                        f"#{idx}</summary><pre>{escape(response_in_msg)}</pre></details>"
                    )
                else:
                    parts.append("<p class='muted'>Tool response message missing for this call.</p>")

                if idx - 1 < len(tool_logs) and isinstance(tool_logs[idx - 1], dict):
                    log_preview = _clip_text(tool_logs[idx - 1].get("result_preview", ""), max_chars)
                    parts.append(
                        "<details><summary>Tool Response Preview (from tool_calls) "
                        f"#{idx}</summary><pre>{escape(log_preview)}</pre></details>"
                    )

            if next_prompt:
                parts.append(
                    "<details><summary>Next-Step Prompt (after tool responses)</summary>"
                    f"<pre>{escape(next_prompt)}</pre></details>"
                )
        parts.append("</section>")
    parts.append("</div>")
    parts.append("</details>")
    return "\n".join(parts)


def _render_agent_block(agent: Dict[str, Any], max_chars: int, include_prompts: bool) -> str:
    name = str(agent.get("name", "agent"))
    role = str(agent.get("role", ""))
    output = _clip_text(agent.get("output", ""), max_chars)
    tool_calls = agent.get("tool_calls", []) or []
    parts: List[str] = []
    parts.append(
        f'<section class="agent"><h4>{escape(name)} <span class="muted">({escape(role)})</span></h4>'
    )
    parts.append(f"<p><strong>Tool Calls:</strong> {len(tool_calls)}</p>")

    if include_prompts:
        prompt = _clip_text(agent.get("input", ""), max_chars)
        parts.append(
            "<details><summary>Agent Prompt</summary>"
            f"<pre>{escape(prompt)}</pre></details>"
        )

    parts.append("<details><summary>Agent Output</summary>" f"<pre>{escape(output)}</pre></details>")
    parts.append(_render_step_timeline(agent, max_chars=max_chars))

    if tool_calls:
        parts.append("<details><summary>Aggregated Tool Calls</summary>")
        for i, call in enumerate(tool_calls, start=1):
            args_text = _pretty_json(call.get("arguments", {}), max_chars)
            result_text = _clip_text(call.get("result_preview", ""), max_chars)
            tool_name = str(call.get("name", "unknown"))
            parts.append(
                "<div class='toolcall'>"
                f"<h5>Call {i}: {escape(tool_name)}</h5>"
                "<details open><summary>Request Arguments</summary>"
                f"<pre>{escape(args_text)}</pre></details>"
                "<details open><summary>Tool Response Preview</summary>"
                f"<pre>{escape(result_text)}</pre></details>"
                "</div>"
            )
        parts.append("</details>")
    else:
        parts.append("<p class='muted'>No tool calls from this agent.</p>")

    raw_messages = agent.get("messages", []) or []
    if raw_messages:
        parts.append(
            "<details><summary>Raw Messages (JSON)</summary>"
            f"<pre>{escape(_pretty_json(raw_messages, max_chars))}</pre></details>"
        )

    parts.append("</section>")
    return "\n".join(parts)


def _render_samples(
    rows: List[Tuple[int, Dict[str, Any]]], max_chars: int, include_prompts: bool
) -> str:
    blocks: List[str] = []
    for row_id, rec in rows:
        correct = bool(rec.get("correct", False))
        status = "correct" if correct else "wrong"
        status_text = "Correct" if correct else "Wrong"

        question = _clip_text(rec.get("question", ""), max_chars)
        gold = _clip_text(rec.get("gold", ""), max_chars)
        pred = _clip_text(rec.get("prediction", ""), max_chars)
        raw = _clip_text(rec.get("raw_prediction", ""), max_chars)
        tool_count = _sample_tool_count(rec)

        block = [
            f"<article class='sample {status}'>",
            f"<h3>Sample #{row_id} - {status_text}</h3>",
            f"<p><strong>Tool Calls In Sample:</strong> {tool_count}</p>",
            f"<details open><summary>Question</summary><pre>{escape(question)}</pre></details>",
            f"<details><summary>Gold Answer</summary><pre>{escape(gold)}</pre></details>",
            f"<details open><summary>Final Answer</summary><pre>{escape(pred)}</pre></details>",
            f"<details><summary>Raw Final Output</summary><pre>{escape(raw)}</pre></details>",
        ]

        agents = rec.get("agents", []) or []
        if agents:
            block.append("<div class='agents'>")
            for agent in agents:
                block.append(_render_agent_block(agent, max_chars=max_chars, include_prompts=include_prompts))
            block.append("</div>")
        else:
            block.append("<p class='muted'>No agent traces found in this sample.</p>")

        block.append("</article>")
        blocks.append("\n".join(block))
    return "\n".join(blocks)


def _build_html(
    *,
    source_path: Path,
    displayed_rows: List[Tuple[int, Dict[str, Any]]],
    full_stats: Dict[str, Any],
    displayed_stats: Dict[str, Any],
    summary: Optional[Dict[str, Any]],
    max_chars: int,
    include_prompts: bool,
) -> str:
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_block = ""
    if summary:
        summary_block = (
            "<section class='card'>"
            "<h2>summary.json (if available)</h2>"
            f"<pre>{escape(_pretty_json(summary, max_chars))}</pre>"
            "</section>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Tool-Call Log Analysis</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; margin: 24px; color: #111; }}
    h1, h2, h3, h4, h5 {{ margin: 0.3rem 0; }}
    pre {{ white-space: pre-wrap; word-wrap: break-word; background: #f6f8fa; padding: 10px; border-radius: 6px; border: 1px solid #e5e7eb; }}
    .muted {{ color: #666; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; margin-bottom: 12px; }}
    .card {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; background: #fff; }}
    .kv th {{ text-align: left; padding-right: 12px; }}
    .simple {{ width: 100%; border-collapse: collapse; }}
    .simple th, .simple td {{ border-bottom: 1px solid #e5e7eb; text-align: left; padding: 6px 4px; }}
    .sample {{ border: 1px solid #d1d5db; border-radius: 8px; padding: 12px; margin: 12px 0; background: #fff; }}
    .sample.correct {{ border-left: 6px solid #16a34a; }}
    .sample.wrong {{ border-left: 6px solid #dc2626; }}
    .agent {{ border: 1px solid #e5e7eb; border-radius: 6px; padding: 10px; margin: 10px 0; background: #fcfcfd; }}
    .toolcall {{ border-left: 4px solid #2563eb; padding-left: 10px; margin: 10px 0; }}
    .steps {{ margin-top: 8px; }}
    .step {{ border: 1px solid #dbe5f0; border-radius: 6px; padding: 8px; margin: 8px 0; background: #f8fbff; }}
    details > summary {{ cursor: pointer; font-weight: 600; }}
  </style>
</head>
<body>
  <h1>Tool-Call Benchmark Log Analysis</h1>
  <p><strong>Source:</strong> {escape(str(source_path))}</p>
  <p><strong>Generated:</strong> {escape(generated)}</p>
  <p class="muted">Each tool response shown below is the saved <code>result_preview</code> from logs.</p>

  <div class="grid">
    {_render_summary_table("All Loaded Samples", full_stats)}
    {_render_summary_table("Displayed Samples", displayed_stats)}
    {_render_counter_table("Tool Usage Counts", displayed_stats["tool_counter"])}
    {_render_counter_table("Tool Calls By Agent Role", displayed_stats["role_counter"])}
  </div>

  {summary_block}

  <section class="card">
    <h2>Samples</h2>
    {_render_samples(displayed_rows, max_chars=max_chars, include_prompts=include_prompts)}
  </section>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render saved GAIA tool-calling results as an HTML report."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default="",
        help="Run directory containing preds.jsonl and optional summary.json.",
    )
    parser.add_argument(
        "--preds_path",
        type=str,
        default="",
        help="Path to preds.jsonl (optional if --run_dir is provided).",
    )
    parser.add_argument(
        "--summary_path",
        type=str,
        default="",
        help="Optional path to summary.json (if omitted, auto-detect next to preds file).",
    )
    parser.add_argument(
        "--out_html",
        type=str,
        default="",
        help="Output HTML path (default: sibling file named toolcall_analysis.html).",
    )
    parser.add_argument(
        "--max_chars",
        type=int,
        default=-1,
        help="Max characters per displayed text block. Use -1 for no truncation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Optional max number of samples to include in report (-1 means all).",
    )
    parser.add_argument(
        "--only_with_calls",
        action="store_true",
        help="If set, include only samples with at least one tool call.",
    )
    parser.add_argument(
        "--include_prompts",
        action="store_true",
        help="If set, include full agent prompts in report.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="",
        help="Optional HF tokenizer name/path for token-length metrics. Fallback is whitespace count.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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

    summary_path: Optional[Path] = None
    if args.summary_path:
        candidate = Path(args.summary_path).expanduser().resolve()
        if candidate.exists():
            summary_path = candidate
    elif run_dir is not None:
        candidate = run_dir / "summary.json"
        if candidate.exists():
            summary_path = candidate
    else:
        candidate = preds_path.parent / "summary.json"
        if candidate.exists():
            summary_path = candidate

    out_html = (
        Path(args.out_html).expanduser().resolve()
        if args.out_html
        else (
            (run_dir / "toolcall_analysis.html").resolve()
            if run_dir is not None
            else (preds_path.parent / "toolcall_analysis.html").resolve()
        )
    )

    token_counter, token_counter_mode = _build_token_counter(args.tokenizer_name)

    rows = _load_preds(preds_path)
    full_stats = _compute_stats(rows, token_counter=token_counter)
    full_stats["token_counter_mode"] = token_counter_mode

    displayed = rows
    if args.only_with_calls:
        displayed = [(rid, rec) for rid, rec in displayed if _sample_tool_count(rec) > 0]
    if args.limit is not None and args.limit >= 0:
        displayed = displayed[: args.limit]
    displayed_stats = _compute_stats(displayed, token_counter=token_counter)
    displayed_stats["token_counter_mode"] = token_counter_mode

    summary = _load_json(summary_path) if summary_path else None

    max_chars = int(args.max_chars)
    if max_chars != -1 and max_chars <= 0:
        raise ValueError("--max_chars must be -1 (no truncation) or a positive integer.")

    html_doc = _build_html(
        source_path=preds_path,
        displayed_rows=displayed,
        full_stats=full_stats,
        displayed_stats=displayed_stats,
        summary=summary,
        max_chars=max_chars,
        include_prompts=bool(args.include_prompts),
    )
    out_html.write_text(html_doc, encoding="utf-8")

    print("Saved report:", out_html)
    print("Loaded samples:", full_stats["samples"])
    print("Displayed samples:", displayed_stats["samples"])
    print("Displayed total tool calls:", displayed_stats["total_tool_calls"])
    print("Displayed accuracy:", f"{displayed_stats['accuracy']:.4f}")


if __name__ == "__main__":
    main()
