import argparse
import json
import os
import re
import sys
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, TextIO, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parallel_synthesis.processor_imports import load_dialogue_dataset_utils


_dialogue_dataset_utils = load_dialogue_dataset_utils()
build_text_references = _dialogue_dataset_utils.build_text_references
normalize_role_content_messages = _dialogue_dataset_utils.normalize_role_content_messages
parse_json_or_python_literal = _dialogue_dataset_utils.parse_json_or_python_literal
render_messages = _dialogue_dataset_utils.render_messages
strip_text = _dialogue_dataset_utils.strip_text

TOUCAN_DATASET = "Agent-Ark/Toucan-1.5M"
TOUCAN_REVISION = "0df3cf37f2abefb380370cfb02eabea2a35ae782"


def _load_dataset(*args, **kwargs):
    from datasets import load_dataset

    return load_dataset(*args, **kwargs)


def _build_tool_call_message(function_call: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(function_call, dict):
        return None
    tool_name = strip_text(function_call.get("name", ""))
    if not tool_name:
        return None
    arguments = function_call.get("arguments", {})
    payload = {
        "name": tool_name,
        "arguments": arguments,
    }
    return {
        "role": "tool_call",
        "content": json.dumps(payload, ensure_ascii=False),
    }


def _append_normalized_assistant(
    normalized: List[Dict[str, Any]],
    *,
    content: str,
) -> bool:
    rendered = strip_text(content)
    if not rendered:
        return False
    normalized.append({"role": "assistant", "content": rendered})
    return True


def _parse_messages(raw_messages: str) -> List[Dict[str, Any]]:
    parsed = json.loads(raw_messages)
    if not isinstance(parsed, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for message in parsed:
        if not isinstance(message, dict):
            continue

        role = strip_text(message.get("role", "")).lower()
        content = strip_text(message.get("content", ""))
        reasoning_content = strip_text(message.get("reasoning_content", ""))
        function_call = message.get("function_call")

        if role in {"user", "system"}:
            if content:
                normalized.append({"role": role, "content": content})
            continue

        if role == "assistant":
            planning_text = content or reasoning_content
            has_planning = _append_normalized_assistant(normalized, content=planning_text)

            tool_call_message = _build_tool_call_message(function_call) if isinstance(function_call, dict) else None
            if tool_call_message is not None:
                if not has_planning and (
                    not normalized
                    or strip_text(normalized[-1].get("role", "")) not in {"assistant", "tool_call", "tool_response"}
                ):
                    normalized.append({"role": "assistant", "content": ""})
                normalized.append(tool_call_message)
            continue

        if role == "function":
            response_message: Dict[str, Any] = {
                "role": "tool_response",
                "content": content,
            }
            tool_name = strip_text(message.get("name", ""))
            if tool_name:
                response_message["name"] = tool_name
            normalized.append(response_message)
            continue

        if role in {"tool_call", "tool_response"}:
            normalized_message: Dict[str, Any] = {
                "role": role,
                "content": content,
            }
            tool_name = strip_text(message.get("name", ""))
            if tool_name:
                normalized_message["name"] = tool_name
            normalized.append(normalized_message)

    return normalized


def _find_final_assistant_index(messages: List[Dict[str, str]]) -> int:
    for idx in range(len(messages) - 1, -1, -1):
        if strip_text(messages[idx].get("role", "")).lower() == "assistant":
            return idx
    return -1


def _count_user_turns(messages: List[Dict[str, str]], end_idx: int) -> int:
    count = 0
    for message in messages[: end_idx + 1]:
        if strip_text(message.get("role", "")).lower() == "user":
            count += 1
    return count


def _nearest_user(messages: List[Dict[str, str]], idx: int) -> Tuple[int, str]:
    for user_idx in range(idx - 1, -1, -1):
        if strip_text(messages[user_idx].get("role", "")).lower() == "user":
            return user_idx, strip_text(messages[user_idx].get("content", ""))
    return -1, ""


def _parse_structured_message(content: str) -> Dict[str, Any]:
    try:
        parsed = parse_json_or_python_literal(content)
    except (ValueError, SyntaxError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _render_tool_event(call_text: str, response_text: str, *, event_idx: Optional[int] = None) -> str:
    parts: List[str] = []
    call = strip_text(call_text)
    response = strip_text(response_text)
    call_tag = f"[TOOL_CALL {event_idx}]" if event_idx is not None else "[TOOL_CALL]"
    response_tag = f"[TOOL_RESPONSE {event_idx}]" if event_idx is not None else "[TOOL_RESPONSE]"
    if call:
        parts.append(f"{call_tag}\n{call}")
    if response:
        parts.append(f"{response_tag}\n{response}")
    return "\n\n".join(parts).strip()


def _render_user_task_tool_trace(tool_events: List[Dict[str, str]]) -> str:
    rendered = [
        _render_tool_event(event.get("call_text", ""), event.get("response_text", ""))
        for event in tool_events
    ]
    return "\n\n".join(part for part in rendered if part.strip()).strip()


def _render_turn_history(
    *,
    user_text: str,
    planning_text: str,
    tool_events: List[Dict[str, Any]],
    assistant_response_text: str,
) -> str:
    parts: List[str] = []
    rendered_user = strip_text(user_text)
    rendered_planning = strip_text(planning_text)
    rendered_response = strip_text(assistant_response_text)
    if rendered_user:
        parts.append(f"[USER_QUERY]\n{rendered_user}")
    if rendered_planning:
        parts.append(f"[ASSISTANT_PLANNING]\n{rendered_planning}")
    for event_idx, event in enumerate(tool_events, start=1):
        call_text = strip_text(event.get("call_text", ""))
        response_text = strip_text(event.get("response_text", ""))
        if call_text:
            parts.append(f"[TOOL_CALL_{event_idx}]\n{call_text}")
        if response_text:
            parts.append(f"[TOOL_RESPONSE_{event_idx}]\n{response_text}")
    if rendered_response:
        parts.append(f"[ASSISTANT_RESPONSE]\n{rendered_response}")
    return "\n\n".join(parts).strip()


def _render_prefill_context(messages: List[Dict[str, str]], end_idx_inclusive: int) -> str:
    prefix_messages = [
        message
        for message in messages[: end_idx_inclusive + 1]
        if strip_text(message.get("role", "")).lower() != "system"
    ]
    return render_messages(prefix_messages)


def _format_planned_tool_call(call_text: str) -> str:
    rendered = strip_text(call_text)
    if not rendered:
        return ""
    payload = _parse_structured_message(rendered)
    tool_name = strip_text(payload.get("name", ""))
    raw_arguments = payload.get("arguments", {})
    arguments = parse_json_or_python_literal(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
    if not isinstance(arguments, dict):
        arguments = {}

    normalized_name = re.sub(r"[^a-z0-9]+", "_", tool_name.lower())
    tokens = [token for token in normalized_name.split("_") if token]
    noise = {
        "mcp",
        "server",
        "service",
        "simple",
        "tool",
        "tools",
        "resource",
        "resources",
        "component",
        "components",
        "library",
        "libraries",
        "fe",
        "byted",
    }
    filtered_tokens = [token for token in tokens if token not in noise and not token.startswith("mcp")]
    if not filtered_tokens:
        filtered_tokens = tokens

    def first_arg(*keys: str) -> str:
        for key in keys:
            value = arguments.get(key)
            if isinstance(value, list):
                joined = ", ".join(strip_text(item) for item in value if strip_text(item))
                if joined:
                    return joined
            text = strip_text(value)
            if text:
                return text
        return ""

    if "alert" in filtered_tokens or "alerts" in filtered_tokens:
        place = first_arg("state", "city", "location", "place", "query")
        if place:
            return f"check for any active weather alerts in {place}."
        return "check for any active weather alerts."

    if "weather" in filtered_tokens:
        place = first_arg("city", "location", "place", "query")
        if place:
            return f"get the current conditions in {place} right now."
        return "get the current weather conditions."

    if "rhyme" in filtered_tokens or "rhymes" in filtered_tokens:
        word = first_arg("input_word", "word", "query")
        if word:
            return f"find rhyming options for \"{word}\"."
        return "find rhyming options."

    if "syllable" in filtered_tokens or "syllables" in filtered_tokens:
        text = first_arg("text", "line", "sentence", "query", "input")
        if text:
            return f"count the syllables in \"{text}\"."
        return "count the syllables."

    if "hello" in filtered_tokens or ("say" in filtered_tokens and "hello" in filtered_tokens):
        target = first_arg("name", "target", "query")
        if target:
            return f"say hello to {target}."
        return "say hello."

    if "search" in filtered_tokens or "find" in filtered_tokens or "query" in filtered_tokens:
        subject = first_arg("query", "queries", "keyword", "keywords", "name", "topic")
        if subject:
            return f"search for {subject}."

    if "list" in filtered_tokens:
        subject = first_arg("library", "name", "query")
        if subject:
            return f"list the available items for {subject}."

    action_map = {
        "get": "get",
        "query": "query",
        "search": "search for",
        "find": "find",
        "list": "list",
        "check": "check",
        "count": "count",
        "say": "say",
        "lookup": "look up",
        "look": "look up",
        "fetch": "fetch",
        "retrieve": "retrieve",
    }
    action = ""
    for token in filtered_tokens:
        if token in action_map:
            action = action_map[token]
            break
    object_tokens = [token for token in filtered_tokens if token not in action_map]
    object_text = " ".join(object_tokens).strip() or "information"
    target = first_arg("query", "name", "city", "state", "location", "word", "text", "input")
    if action and target:
        return f"{action} {object_text} for {target}."
    if action:
        return f"{action} {object_text}."
    if tool_name:
        return f"use the tool {tool_name}."
    return rendered


def _dedupe_nonempty(texts: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for text in texts:
        rendered = strip_text(text)
        if not rendered or rendered in seen:
            continue
        seen.add(rendered)
        out.append(rendered)
    return out


def _collect_execution_blocks(messages: List[Dict[str, str]], end_idx: int) -> Tuple[List[Dict[str, Any]], bool]:
    blocks: List[Dict[str, Any]] = []
    has_parallel = False
    idx = 0
    while idx < end_idx:
        message = messages[idx]
        role = strip_text(message.get("role", "")).lower()
        if role != "assistant":
            idx += 1
            continue

        next_idx = idx + 1
        if next_idx >= end_idx or strip_text(messages[next_idx].get("role", "")).lower() != "tool_call":
            idx += 1
            continue

        user_index, user_text = _nearest_user(messages, idx)
        call_messages: List[Dict[str, str]] = []
        response_messages: List[Dict[str, str]] = []
        tool_call_count = 0
        max_call_burst = 0
        has_parallel_burst = False

        while next_idx < end_idx:
            next_message = messages[next_idx]
            next_role = strip_text(next_message.get("role", "")).lower()
            if next_role not in {"tool_call", "tool_response"}:
                break

            call_burst = 0
            while next_idx < end_idx:
                next_message = messages[next_idx]
                next_role = strip_text(next_message.get("role", "")).lower()
                if next_role != "tool_call":
                    break
                call_messages.append(next_message)
                tool_call_count += 1
                call_burst += 1
                next_idx += 1

            if call_burst > 0:
                max_call_burst = max(max_call_burst, call_burst)
                if call_burst >= 2:
                    has_parallel_burst = True

            while next_idx < end_idx:
                next_message = messages[next_idx]
                next_role = strip_text(next_message.get("role", "")).lower()
                if next_role != "tool_response":
                    break
                response_messages.append(next_message)
                next_idx += 1

        if tool_call_count <= 0:
            idx += 1
            continue

        tool_events: List[Dict[str, Any]] = []
        max_events = max(len(call_messages), len(response_messages))
        for event_idx in range(max_events):
            call_message = call_messages[event_idx] if event_idx < len(call_messages) else {}
            response_message = response_messages[event_idx] if event_idx < len(response_messages) else {}
            call_text = strip_text(call_message.get("content", ""))
            response_text = strip_text(response_message.get("content", ""))
            if not call_text and not response_text:
                continue
            call_payload = _parse_structured_message(call_text)
            response_payload = _parse_structured_message(response_text)
            tool_events.append(
                {
                    "call_text": call_text,
                    "response_text": response_text,
                    "tool_name": strip_text(call_payload.get("name", "")),
                    "response_tool_name": strip_text(response_payload.get("name", "")) or strip_text(response_message.get("name", "")),
                }
            )

        blocks.append(
            {
                "assistant_index": idx,
                "assistant_text": strip_text(message.get("content", "")),
                "assistant_response_text": (
                    strip_text(messages[next_idx].get("content", ""))
                    if next_idx < end_idx
                    and strip_text(messages[next_idx].get("role", "")).lower() == "assistant"
                    else ""
                ),
                "user_index": user_index,
                "user_text": user_text,
                "tool_call_count": tool_call_count,
                "max_call_burst": max_call_burst,
                "parallel": has_parallel_burst,
                "tool_events": tool_events,
            }
        )
        if has_parallel_burst:
            has_parallel = True
        idx = next_idx
    return blocks, has_parallel


def _select_target_parallel_block(blocks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for block in reversed(blocks):
        if bool(block.get("parallel", False)) and block.get("tool_events"):
            return block
    return None


def _group_blocks_by_user(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: List[Dict[str, Any]] = []
    for block in blocks:
        user_index = int(block.get("user_index", -1))
        user_text = strip_text(block.get("user_text", ""))
        tool_events = block.get("tool_events", [])
        if user_index < 0 or not user_text or not tool_events:
            continue

        if grouped and grouped[-1]["user_index"] == user_index:
            group = grouped[-1]
        else:
            group = {
                "user_index": user_index,
                "user_text": user_text,
                "planning_texts": [],
                "assistant_response_texts": [],
                "tool_events": [],
                "tool_count": 0,
                "max_call_burst": 0,
                "parallel": False,
            }
            grouped.append(group)

        assistant_text = strip_text(block.get("assistant_text", ""))
        if assistant_text:
            group["planning_texts"].append(assistant_text)
        assistant_response_text = strip_text(block.get("assistant_response_text", ""))
        if assistant_response_text:
            group["assistant_response_texts"].append(assistant_response_text)
        group["tool_events"].extend(tool_events)
        group["tool_count"] += int(block.get("tool_call_count", 0) or 0)
        group["max_call_burst"] = max(group["max_call_burst"], int(block.get("max_call_burst", 0) or 0))
        group["parallel"] = bool(group["parallel"] or block.get("parallel", False))

    for group in grouped:
        deduped_plans = _dedupe_nonempty(group.get("planning_texts", []))
        deduped_responses = _dedupe_nonempty(group.get("assistant_response_texts", []))
        group["planning_texts"] = deduped_plans
        group["assistant_response_texts"] = deduped_responses
        group["planning_text"] = "\n\n".join(deduped_plans).strip()
        group["assistant_response_text"] = "\n\n".join(deduped_responses).strip()
        group["trace_context"] = _render_user_task_tool_trace(group.get("tool_events", []))
        group["turn_history"] = _render_turn_history(
            user_text=group.get("user_text", ""),
            planning_text=group.get("planning_text", ""),
            tool_events=group.get("tool_events", []),
            assistant_response_text=group.get("assistant_response_text", ""),
        )
        group["context"] = group["turn_history"]
        group["planned_tool_calls"] = [
            planned
            for planned in (
                _format_planned_tool_call(event.get("call_text", ""))
                for event in group.get("tool_events", [])
            )
            if planned
        ]
        group["single_burst_only"] = bool(
            int(group.get("tool_count", 0) or 0) > 0
            and int(group.get("tool_count", 0) or 0) == int(group.get("max_call_burst", 0) or 0)
        )
    return [group for group in grouped if strip_text(group.get("turn_history", ""))]


def _build_multi_turn_question(num_tasks: int) -> str:
    return (
        "You will be provided with several context traces from different sub-agents.\n\n"
        "Each sub-agent trace contains tool calls and tool responses produced while working on a particular "
        "task and query inside a multi-turn conversation. For each sub-agent, infer:\n"
        "1. the underlying task and query;\n"
        "2. the planned tool calls.\n\n"
        "Use only the provided traces and output the results in order using exactly the following format:\n"
        "Sub-agent 1: The task and query is: ... The planned tool calls are 1). ... 2). ...\n"
        "Sub-agent 2: The task and query is: ... The planned tool calls are 1). ... 2). ...\n\n"
        f"There are {num_tasks} sub-agent traces."
    )


def _build_multi_turn_planning_question(current_turn_query: str) -> str:
    rendered_query = strip_text(current_turn_query)
    return (
        "You are given several previous turn histories and the current user query.\n\n"
        "Read the previous turn histories carefully, reason step by step, and write the best next "
        "assistant planning message for the current turn before any tool calls are made.\n\n"
        f"Current turn query:\n{rendered_query}"
    )


def _build_single_turn_question(question: str) -> str:
    rendered_question = strip_text(question)
    return (
        "You are given a user question and several tool-call traces that were executed to help answer it.\n\n"
        "Each trace contains a tool call and its corresponding tool response. Read the tool-call contents "
        "and tool responses carefully, reason over the provided evidence, and write the final assistant "
        "response to the question.\n\n"
        f"Original question:\n{rendered_question}"
    )


def _build_multi_turn_target(group: Dict[str, Any], idx: int) -> str:
    query = strip_text(group.get("user_text", ""))
    planned_tool_calls = group.get("planned_tool_calls", []) or []
    if planned_tool_calls:
        planned_text = " ".join(f"{tool_idx}). {tool_call}" for tool_idx, tool_call in enumerate(planned_tool_calls, start=1))
    else:
        planned_text = "Unavailable."
    return (
        f"Sub-agent {idx}: The task and query is: {query}. "
        f"The planned tool calls are {planned_text}"
    ).strip()


def _prefix_previous_turn_histories(contexts: List[str]) -> List[str]:
    prefixed: List[str] = []
    for idx, context in enumerate(contexts, start=1):
        rendered = strip_text(context)
        if not rendered:
            continue
        prefixed.append(f"This is previous turn history {idx}:\n{rendered}")
    return prefixed


def _build_single_turn_row(
    *,
    row: Dict[str, Any],
    messages: List[Dict[str, str]],
    final_idx: int,
    blocks: List[Dict[str, Any]],
    written: int,
) -> Optional[Dict[str, Any]]:
    target_block = _select_target_parallel_block(blocks)
    if target_block is None:
        return None

    contexts = [
        _render_tool_event(
            event.get("call_text", ""),
            event.get("response_text", ""),
            event_idx=event_idx,
        )
        for event_idx, event in enumerate(target_block.get("tool_events", []), start=1)
    ]
    extract_contexts = [context for context in contexts if strip_text(context)]
    if not extract_contexts:
        return None

    prefill_context = _render_prefill_context(messages, int(target_block.get("assistant_index", 0) or 0))
    contexts = [
        "\n\n".join(part for part in [prefill_context, extract_context] if strip_text(part)).strip()
        for extract_context in extract_contexts
    ]

    gold = strip_text(messages[final_idx].get("content", ""))
    original_question = strip_text(row.get("question", "")) or strip_text(target_block.get("user_text", ""))
    if not gold or not original_question:
        return None
    question = _build_single_turn_question(original_question)

    references = build_text_references(
        contexts,
        prefix="tool_pair",
        extra_by_index=[
            {
                "tool_name": event.get("tool_name", ""),
                "response_tool_name": event.get("response_tool_name", ""),
            }
            for event in target_block.get("tool_events", [])
        ],
    )
    return {
        "id": strip_text(row.get("uuid", "")) or f"toucan_single_{written}",
        "dataset": "toucan_single_parallel",
        "question": question,
        "query": question,
        "solution": gold,
        "gold": gold,
        "references": references,
        "agent_reference_contexts": contexts,
        "agent_extract_contexts": extract_contexts,
        "subset_name": strip_text(row.get("subset_name", "")),
        "target_tools": strip_text(row.get("target_tools", "")),
        "original_question_text": strip_text(row.get("question", "")),
        "tools": strip_text(row.get("tools", "")),
        "messages": messages,
        "assistant_before_parallel_call": strip_text(target_block.get("assistant_text", "")),
        "assistant_after_parallel_call": gold,
        "parallel_call_user_question": strip_text(target_block.get("user_text", "")),
        "current_turn_query": original_question,
        "tool_event_count": len(target_block.get("tool_events", [])),
        "max_parallel_tool_calls": int(target_block.get("max_call_burst", 0) or 0),
        "selection_tags": ["single_turn_parallel_tool_call"],
        "is_multi_turn": False,
        "has_parallel_tool_call": True,
        "is_single_turn_parallel_tool_call": True,
        "is_multi_turn_parallel_tool_call": False,
    }


def _build_multi_turn_row(
    *,
    row: Dict[str, Any],
    messages: List[Dict[str, str]],
    blocks: List[Dict[str, Any]],
    user_turns: int,
    written: int,
) -> Optional[Dict[str, Any]]:
    grouped_tasks = _group_blocks_by_user(blocks)
    if len(grouped_tasks) < 3:
        return None
    if any(not bool(group.get("single_burst_only", False)) for group in grouped_tasks):
        return None
    has_parallel = bool(any(bool(group.get("parallel", False)) for group in grouped_tasks))

    previous_groups = grouped_tasks[:-1]
    current_group = grouped_tasks[-1]
    contexts = _prefix_previous_turn_histories(
        [
            strip_text(group.get("turn_history", ""))
            for group in previous_groups
            if strip_text(group.get("turn_history", ""))
        ]
    )
    if not contexts:
        return None

    current_turn_query = strip_text(current_group.get("user_text", ""))
    gold = strip_text(current_group.get("planning_text", ""))
    if not current_turn_query or not gold:
        return None

    references = build_text_references(
        contexts,
        prefix="task_trace",
        extra_by_index=[
            {
                "tool_call_count": int(group.get("tool_count", 0) or 0),
                "max_call_burst": int(group.get("max_call_burst", 0) or 0),
                "parallel": bool(group.get("parallel", False)),
                "user_turn_index": int(group.get("user_index", -1)),
            }
            for group in previous_groups
            if strip_text(group.get("turn_history", ""))
        ],
    )
    return {
        "id": strip_text(row.get("uuid", "")) or f"toucan_multi_{written}",
        "dataset": "toucan_multi_parallel",
        "question": _build_multi_turn_planning_question(current_turn_query),
        "query": _build_multi_turn_planning_question(current_turn_query),
        "solution": gold,
        "gold": gold,
        "references": references,
        "agent_reference_contexts": contexts,
        "agent_extract_contexts": list(contexts),
        "agent_reference_contexts_truth": [
            strip_text(group.get("planning_text", ""))
            for group in previous_groups
            if strip_text(group.get("turn_history", ""))
        ],
        "subset_name": strip_text(row.get("subset_name", "")),
        "target_tools": strip_text(row.get("target_tools", "")),
        "original_question_text": strip_text(row.get("question", "")),
        "tools": strip_text(row.get("tools", "")),
        "messages": messages,
        "task_user_questions": [strip_text(group.get("user_text", "")) for group in grouped_tasks],
        "task_planning_texts": [strip_text(group.get("planning_text", "")) for group in grouped_tasks],
        "task_planned_tool_calls": [group.get("planned_tool_calls", []) for group in grouped_tasks],
        "task_targets": [strip_text(group.get("planning_text", "")) for group in grouped_tasks],
        "task_assistant_responses": [
            strip_text(group.get("assistant_response_text", ""))
            for group in grouped_tasks
        ],
        "previous_turn_user_questions": [
            strip_text(group.get("user_text", ""))
            for group in previous_groups
            if strip_text(group.get("turn_history", ""))
        ],
        "previous_turn_planning_texts": [
            strip_text(group.get("planning_text", ""))
            for group in previous_groups
            if strip_text(group.get("turn_history", ""))
        ],
        "previous_turn_assistant_responses": [
            strip_text(group.get("assistant_response_text", ""))
            for group in previous_groups
            if strip_text(group.get("turn_history", ""))
        ],
        "current_turn_query": current_turn_query,
        "query_turn": {
            "role": "user",
            "content": current_turn_query,
        },
        "current_turn_planning": gold,
        "current_turn_assistant_response": strip_text(current_group.get("assistant_response_text", "")),
        "user_turn_count": user_turns,
        "task_count": len(grouped_tasks),
        "context_count": len(contexts),
        "selection_tags": (
            ["multi_turn_parallel_tool_call", "single_burst_tool_plan", "min3_total_turns_all_prev_turns"]
            if has_parallel
            else ["single_burst_tool_plan", "min3_total_turns_all_prev_turns"]
        ),
        "is_multi_turn": True,
        "has_parallel_tool_call": has_parallel,
        "is_single_turn_parallel_tool_call": False,
        "is_multi_turn_parallel_tool_call": has_parallel,
        "max_parallel_tool_calls": max(int(group.get("max_call_burst", 0) or 0) for group in grouped_tasks),
    }


def iter_toucan_rows(
    *,
    split: str,
    config_name: str,
    revision: str = TOUCAN_REVISION,
) -> Iterable[Dict[str, Any]]:
    load_kwargs: Dict[str, Any] = {"split": split, "streaming": True}
    if revision:
        load_kwargs["revision"] = revision
    ds = _load_dataset(TOUCAN_DATASET, config_name, **load_kwargs)
    written_single = 0
    written_multi = 0
    for row in ds:
        try:
            messages = _parse_messages(row.get("messages", ""))
        except json.JSONDecodeError:
            continue

        final_idx = _find_final_assistant_index(messages)
        if final_idx <= 0:
            continue

        final_gold = strip_text(messages[final_idx].get("content", ""))
        if not final_gold:
            continue

        blocks, has_parallel = _collect_execution_blocks(messages, final_idx)
        if not blocks:
            continue

        user_turns = _count_user_turns(messages, final_idx - 1)
        is_multi_turn = strip_text(row.get("subset_name", "")) == "multi-turn" or user_turns >= 2

        if is_multi_turn:
            built = _build_multi_turn_row(
                row=row,
                messages=messages,
                blocks=blocks,
                user_turns=user_turns,
                written=written_multi,
            )
            if built is None:
                continue
            if config_name.upper() == "SFT":
                built.pop("config_name", None)
            else:
                built["config_name"] = config_name
            written_multi += 1
            yield built
            continue

        if not has_parallel:
            continue

        built = _build_single_turn_row(
            row=row,
            messages=messages,
            final_idx=final_idx,
            blocks=blocks,
            written=written_single,
        )
        if built is None:
            continue
        if config_name.upper() == "SFT":
            built.pop("config_name", None)
        else:
            built["config_name"] = config_name
        written_single += 1
        yield built


def _parse_config_names(raw: str) -> List[str]:
    names = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not names:
        raise ValueError("--config_names must contain at least one Toucan configuration.")
    return names


def _write_row(fh: TextIO, row: Dict[str, Any]) -> None:
    fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_with_validation_tail(
    row: Dict[str, Any],
    *,
    tail: Deque[Dict[str, Any]],
    validation_rows: int,
    train_fh: TextIO,
) -> bool:
    """Keep the final N rows for validation and stream older rows to train."""
    tail.append(row)
    if len(tail) <= validation_rows:
        return False
    _write_row(train_fh, tail.popleft())
    return True


def write_unified_toucan(
    *,
    split: str,
    config_names: List[str],
    multi_turn_config: str,
    output_dir: Path,
    validation_rows: int,
    max_rows: int,
    revision: str,
) -> Dict[str, int]:
    """Build the four Toucan files consumed by the release training recipe."""
    if validation_rows < 0:
        raise ValueError("--validation_rows must be non-negative.")
    if multi_turn_config not in config_names:
        raise ValueError("--multi_turn_config must be present in --config_names.")

    output_dir.mkdir(parents=True, exist_ok=True)
    single_train_path = output_dir / "train_single_turn_parallel_tool_call_unified.jsonl"
    single_validation_path = output_dir / "validation_single_turn_parallel_tool_call_unified.jsonl"
    multi_train_path = output_dir / "train_multi_turn_parallel_tool_call.jsonl"
    multi_validation_path = output_dir / "validation_multi_turn_parallel_tool_call.jsonl"

    counts = {
        "single_train": 0,
        "single_validation": 0,
        "multi_train": 0,
        "multi_validation": 0,
    }
    final_config = config_names[-1]
    single_validation_tail: Deque[Dict[str, Any]] = deque()
    multi_validation_tail: Deque[Dict[str, Any]] = deque()

    with (
        single_train_path.open("w", encoding="utf-8") as single_train_fh,
        single_validation_path.open("w", encoding="utf-8") as single_validation_fh,
        multi_train_path.open("w", encoding="utf-8") as multi_train_fh,
        multi_validation_path.open("w", encoding="utf-8") as multi_validation_fh,
    ):
        for config_name in config_names:
            config_single_rows = 0
            config_multi_rows = 0
            for row in iter_toucan_rows(
                split=split,
                config_name=config_name,
                revision=revision,
            ):
                is_single = bool(row.get("is_single_turn_parallel_tool_call", False))
                is_multi = bool(row.get("is_multi_turn", False))

                if is_single:
                    if max_rows > 0 and config_single_rows >= max_rows:
                        continue
                    config_single_rows += 1
                    if config_name == final_config and validation_rows > 0:
                        if _append_with_validation_tail(
                            row,
                            tail=single_validation_tail,
                            validation_rows=validation_rows,
                            train_fh=single_train_fh,
                        ):
                            counts["single_train"] += 1
                    else:
                        _write_row(single_train_fh, row)
                        counts["single_train"] += 1

                if is_multi and config_name == multi_turn_config:
                    if max_rows > 0 and config_multi_rows >= max_rows:
                        continue
                    config_multi_rows += 1
                    if validation_rows > 0:
                        if _append_with_validation_tail(
                            row,
                            tail=multi_validation_tail,
                            validation_rows=validation_rows,
                            train_fh=multi_train_fh,
                        ):
                            counts["multi_train"] += 1
                    else:
                        _write_row(multi_train_fh, row)
                        counts["multi_train"] += 1

                if max_rows > 0:
                    single_done = config_single_rows >= max_rows
                    multi_done = config_name != multi_turn_config or config_multi_rows >= max_rows
                    if single_done and multi_done:
                        break

        for row in single_validation_tail:
            _write_row(single_validation_fh, row)
            counts["single_validation"] += 1
        for row in multi_validation_tail:
            _write_row(multi_validation_fh, row)
            counts["multi_validation"] += 1

    paths = {
        "single_train": single_train_path,
        "single_validation": single_validation_path,
        "multi_train": multi_train_path,
        "multi_validation": multi_validation_path,
    }
    for name, path in paths.items():
        print(f"[toucan] wrote {counts[name]} rows to {path}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the unified Toucan single-turn and multi-turn train/validation "
            "files used by Parallel Synthesis."
        )
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--config_names",
        type=str,
        default="SFT,Kimi-K2,Qwen3",
        help="Comma-separated configurations in concatenation order.",
    )
    parser.add_argument(
        "--multi_turn_config",
        type=str,
        default="SFT",
        help="Configuration used for the multi-turn output.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=TOUCAN_REVISION,
        help="Pinned Hugging Face dataset revision.",
    )
    parser.add_argument("--output_dir", type=str, default="data/toucan/processed")
    parser.add_argument(
        "--validation_rows",
        type=int,
        default=200,
        help="Reserve the final N rows of the relevant source stream for validation.",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=-1,
        help=(
            "If >0, keep at most N single-turn and N multi-turn rows per source "
            "configuration. Intended for processor smoke checks."
        ),
    )
    args = parser.parse_args()
    write_unified_toucan(
        split=args.split,
        config_names=_parse_config_names(args.config_names),
        multi_turn_config=args.multi_turn_config,
        output_dir=Path(args.output_dir),
        validation_rows=args.validation_rows,
        max_rows=args.max_rows,
        revision=args.revision,
    )


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
