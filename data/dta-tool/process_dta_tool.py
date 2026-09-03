import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parallel_synthesis.processor_imports import load_context_dataset_utils, load_dialogue_dataset_utils


_context_dataset_utils = load_context_dataset_utils()
write_jsonl = _context_dataset_utils.write_jsonl

_dialogue_dataset_utils = load_dialogue_dataset_utils()
build_text_references = _dialogue_dataset_utils.build_text_references
extract_balanced_block = _dialogue_dataset_utils.extract_balanced_block
parse_json_or_python_literal = _dialogue_dataset_utils.parse_json_or_python_literal
strip_text = _dialogue_dataset_utils.strip_text

DTA_TOOL_DATASET = "dongsheng/DTA-Tool"
DTA_TOOL_REVISION = "76e4d52f630aaba4721abd56a7666058664068c6"


def _load_dataset(*args, **kwargs):
    from datasets import load_dataset
    return load_dataset(*args, **kwargs)


def _extract_function_call_array(text: str) -> Optional[List[Dict[str, Any]]]:
    block = extract_balanced_block(text, "Function Call:")
    if block is None:
        return None
    parsed = parse_json_or_python_literal(block)
    if not isinstance(parsed, list):
        return None
    normalized: List[Dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "name": strip_text(item.get("name", "")),
                "arguments": item.get("arguments", {}),
            }
        )
    return normalized


def _extract_finish_final_answer(text: str) -> str:
    calls = _extract_function_call_array(text)
    if not calls:
        return ""
    for call in calls:
        if strip_text(call.get("name", "")) != "Finish":
            continue
        arguments = call.get("arguments", {})
        if isinstance(arguments, dict):
            return strip_text(arguments.get("final_answer", ""))
    return ""


def _extract_text_before_function_call(text: str) -> str:
    rendered = str(text or "")
    marker = "Function Call:"
    marker_idx = rendered.find(marker)
    prefix = rendered[:marker_idx] if marker_idx >= 0 else rendered
    prefix = strip_text(prefix)
    if prefix.lower().startswith("thought:"):
        prefix = strip_text(prefix[len("Thought:") :])
    return prefix


def _format_gold_answer(thought: str, final_answer: str) -> str:
    rendered_thought = strip_text(thought)
    rendered_final_answer = strip_text(final_answer)
    if not rendered_final_answer:
        return ""
    return f"<think> {rendered_thought} </think> \\box{{{rendered_final_answer}}}"


def _build_question(user_question: str) -> str:
    rendered_question = strip_text(user_question)
    return (
        "You are given a user question and several tool-call traces that were executed to help answer it.\n\n"
        "Each trace contains a tool call and its corresponding tool response. Read the tool-call contents "
        "and tool responses carefully, reason step by step, and produce the final assistant response. "
        "Put the final answer in the format \\box{...}.\n\n"
        f"Original question:\n{rendered_question}"
    )


def _nearest_previous(conversation: List[Dict[str, Any]], idx: int, role: str) -> str:
    target_role = strip_text(role).lower()
    for prev_idx in range(idx - 1, -1, -1):
        message = conversation[prev_idx]
        if strip_text(message.get("from", "")).lower() == target_role:
            return strip_text(message.get("value", ""))
    return ""


def _render_conversation_prefix(conversation: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for msg_idx, message in enumerate(conversation, start=1):
        role = strip_text(message.get("from", "")).lower().replace(" ", "_")
        content = strip_text(message.get("value", ""))
        if not role or not content:
            continue
        parts.append(f"[{role.upper()}_{msg_idx}]\n{content}")
    return "\n\n".join(parts).strip()


def _pair_parallel_calls_with_results(
    external_calls: List[Dict[str, Any]],
    function_messages: List[Dict[str, Any]],
    *,
    history_prefix: str,
    assistant_precall_thought: str,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    contexts: List[str] = []
    metadata: List[Dict[str, Any]] = []
    for idx, call in enumerate(external_calls):
        response_raw = function_messages[idx] if idx < len(function_messages) else {}
        response_text = strip_text(response_raw.get("value", ""))
        parsed_response = parse_json_or_python_literal(response_text)
        parts = []
        if history_prefix:
            parts.append(f"[PREVIOUS_CONTEXT]\n{history_prefix}")
        if assistant_precall_thought:
            parts.append(f"[ASSISTANT_PRECALL_THOUGHT]\n{assistant_precall_thought}")
        parts.append(f"[PARALLEL_TOOL_CALL_{idx + 1}]\n{json.dumps(call, ensure_ascii=False)}")
        parts.append(f"[PARALLEL_TOOL_RESPONSE_{idx + 1}]\n{response_text}")
        contexts.append("\n\n".join(part for part in parts if part.strip()).strip())
        response_name = ""
        if isinstance(parsed_response, dict):
            response_name = strip_text(parsed_response.get("name", ""))
        metadata.append(
            {
                "tool_name": strip_text(call.get("name", "")),
                "response_tool_name": response_name,
            }
        )
    return contexts, metadata


def _render_extract_contexts(
    external_calls: List[Dict[str, Any]],
    function_messages: List[Dict[str, Any]],
) -> List[str]:
    extract_contexts: List[str] = []
    for idx, call in enumerate(external_calls):
        response_raw = function_messages[idx] if idx < len(function_messages) else {}
        response_text = strip_text(response_raw.get("value", ""))
        parts = [
            f"[PARALLEL_TOOL_CALL_{idx + 1}]\n{json.dumps(call, ensure_ascii=False)}",
            f"[PARALLEL_TOOL_RESPONSE_{idx + 1}]\n{response_text}",
        ]
        extract_contexts.append("\n\n".join(part for part in parts if part.strip()).strip())
    return extract_contexts


def iter_dta_tool_rows(*, split: str, max_rows: int) -> Iterable[Dict[str, Any]]:
    ds = _load_dataset(
        DTA_TOOL_DATASET,
        split=split,
        streaming=True,
        revision=DTA_TOOL_REVISION,
    )
    written = 0
    for row in ds:
        conversations = row.get("conversations", [])
        if not isinstance(conversations, list):
            continue
        event_idx = 0
        for idx, message in enumerate(conversations):
            if strip_text(message.get("from", "")).lower() != "assistant":
                continue
            assistant_before = strip_text(message.get("value", ""))
            calls = _extract_function_call_array(assistant_before)
            if not calls:
                continue
            external_calls = [call for call in calls if strip_text(call.get("name", "")) != "Finish"]
            if len(external_calls) < 2:
                continue

            function_messages: List[Dict[str, Any]] = []
            next_idx = idx + 1
            while next_idx < len(conversations) and strip_text(conversations[next_idx].get("from", "")).lower() == "function":
                function_messages.append(conversations[next_idx])
                next_idx += 1
            if len(function_messages) < len(external_calls):
                continue

            assistant_after = ""
            if next_idx < len(conversations) and strip_text(conversations[next_idx].get("from", "")).lower() == "assistant":
                assistant_after = strip_text(conversations[next_idx].get("value", ""))
            assistant_after_thought = _extract_text_before_function_call(assistant_after)
            final_answer = _extract_finish_final_answer(assistant_after)
            gold = _format_gold_answer(assistant_after_thought, final_answer)
            if not gold:
                continue

            history_before_parallel_call = _render_conversation_prefix(conversations[:idx])
            assistant_before_thought = _extract_text_before_function_call(assistant_before)
            contexts, reference_meta = _pair_parallel_calls_with_results(
                external_calls,
                function_messages,
                history_prefix=history_before_parallel_call,
                assistant_precall_thought=assistant_before_thought,
            )
            extract_contexts = _render_extract_contexts(external_calls, function_messages)
            if not contexts:
                continue

            preceding_user = _nearest_previous(conversations, idx, "user") or strip_text(row.get("id", ""))
            question = _build_question(preceding_user)
            system_prompt = _nearest_previous(conversations, idx, "system")
            event_idx += 1
            yield {
                "id": f"dta_tool_{written}_{event_idx}",
                "dataset": "dta_tool",
                "question": question,
                "query": question,
                "solution": gold,
                "gold": gold,
                "references": build_text_references(
                    contexts,
                    prefix="parallel_call",
                    extra_by_index=reference_meta,
                ),
                "agent_reference_contexts": contexts,
                "agent_extract_contexts": extract_contexts,
                "sample_id": strip_text(row.get("id", "")),
                "system_prompt": system_prompt,
                "current_turn_query": preceding_user,
                "preceding_user_instruction": preceding_user,
                "history_before_parallel_call": history_before_parallel_call,
                "assistant_before_thought": assistant_before_thought,
                "assistant_before_parallel_call": assistant_before,
                "assistant_after_thought": assistant_after_thought,
                "assistant_after_parallel_call": assistant_after,
                "final_answer": final_answer,
                "parallel_function_calls": external_calls,
                "num_parallel_function_calls": len(external_calls),
                "parallel_function_names": [strip_text(call.get("name", "")) for call in external_calls],
                "function_results": [strip_text(message.get("value", "")) for message in function_messages[: len(external_calls)]],
                "parallel_call_turn_index": idx,
                "turn_before": {
                    "user_instruction": preceding_user,
                    "assistant_output": assistant_before,
                },
                "turn_after": {
                    "assistant_output": assistant_after,
                    "final_answer": final_answer,
                },
            }
            written += 1
            if max_rows > 0 and written >= max_rows:
                return


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract DTA-Tool events with parallel external function calls."
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, default="data/dta-tool/processed")
    parser.add_argument("--output_file", type=str, default="train_parallel_function_calls.jsonl")
    parser.add_argument("--max_rows", type=int, default=-1)
    args = parser.parse_args()

    output_path = Path(args.output_dir) / args.output_file
    count = write_jsonl(
        str(output_path),
        iter_dta_tool_rows(split=args.split, max_rows=args.max_rows),
    )
    print(f"[dta-tool] wrote {count} rows to {output_path}")


if __name__ == "__main__":
    main()
