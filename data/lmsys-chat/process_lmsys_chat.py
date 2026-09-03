import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parallel_synthesis.processor_imports import load_context_dataset_utils, load_dialogue_dataset_utils


_context_dataset_utils = load_context_dataset_utils()
write_jsonl = _context_dataset_utils.write_jsonl

_dialogue_dataset_utils = load_dialogue_dataset_utils()
build_text_references = _dialogue_dataset_utils.build_text_references
normalize_role_content_messages = _dialogue_dataset_utils.normalize_role_content_messages
render_messages = _dialogue_dataset_utils.render_messages
strip_text = _dialogue_dataset_utils.strip_text

LMSYS_DATASET = "lmsys/lmsys-chat-1m"
LMSYS_REVISION = "200748d9d3cddcc9d782887541057aca0b18c5da"


def _load_dataset(*args, **kwargs):
    from datasets import load_dataset

    return load_dataset(*args, **kwargs)


def _conversation_is_english(
    conversation: List[Dict[str, Any]],
    *,
    top_level_language: str,
    require_all_turns_english: bool,
) -> bool:
    if strip_text(top_level_language) != "English":
        return False
    if not require_all_turns_english:
        return True
    return all(strip_text(message.get("language", "")) in {"", "English"} for message in conversation)


def _build_prior_round_contexts(history: List[Dict[str, str]]) -> List[str]:
    contexts: List[str] = []
    pending_user: Optional[Dict[str, str]] = None
    for message in history:
        role = strip_text(message.get("role", "")).lower()
        if role == "user":
            pending_user = message
            continue
        if role != "assistant":
            continue
        if pending_user is None:
            continue
        contexts.append(render_messages([pending_user, message]))
        pending_user = None
    return [context for context in contexts if strip_text(context)]


def _prefix_previous_chat_histories(contexts: List[str]) -> List[str]:
    prefixed: List[str] = []
    for idx, context in enumerate(contexts, start=1):
        rendered = strip_text(context)
        if not rendered:
            continue
        prefixed.append(f"This is previous chat history {idx}:\n{rendered}")
    return prefixed


def _find_last_user_index(conversation: List[Dict[str, str]]) -> int:
    for idx in range(len(conversation) - 2, -1, -1):
        if strip_text(conversation[idx].get("role", "")).lower() == "user":
            return idx
    return -1


def _build_query(current_turn_query: str) -> str:
    rendered_query = strip_text(current_turn_query)
    return (
        "You are given several previous user-assistant interaction contexts and the current user query.\n\n"
        "Read the previous contexts carefully, reason step by step, and write the best next assistant "
        "response for the current turn.\n\n"
        f"Current turn query:\n{rendered_query}"
    )


def iter_lmsys_rows(
    *,
    split: str,
    min_turns: int,
    require_all_turns_english: bool,
    drop_redacted: bool,
    include_all_languages: bool,
    max_rows: int,
) -> Iterable[Dict[str, Any]]:
    ds = _load_dataset(
        LMSYS_DATASET,
        split=split,
        streaming=True,
        revision=LMSYS_REVISION,
    )
    written = 0
    for row in ds:
        conversation = normalize_role_content_messages(
            row.get("conversation", []),
            role_key="role",
            content_key="content",
        )
        if not conversation:
            continue
        if int(row.get("turn", 0) or 0) < min_turns:
            continue
        if not include_all_languages:
            if strip_text(row.get("language", "")) != "English":
                continue
            if not _conversation_is_english(
                row.get("conversation", []),
                top_level_language=row.get("language", ""),
                require_all_turns_english=require_all_turns_english,
            ):
                continue
        if drop_redacted and bool(row.get("redacted", False)):
            continue
        if any(strip_text(message.get("role", "")).lower() not in {"user", "assistant"} for message in conversation):
            continue
        if not all(
            strip_text(message.get("content", ""))
            for message in conversation
            if strip_text(message.get("role", "")).lower() == "user"
        ):
            continue
        if strip_text(conversation[-1].get("role", "")).lower() != "assistant":
            continue

        gold = strip_text(conversation[-1].get("content", ""))
        last_user_idx = _find_last_user_index(conversation)
        if last_user_idx < 0:
            continue
        current_turn_query = strip_text(conversation[last_user_idx].get("content", ""))
        history_before_question = conversation[:last_user_idx]
        contexts = _prefix_previous_chat_histories(_build_prior_round_contexts(history_before_question))
        if not gold or not current_turn_query or len(contexts) < 2:
            continue

        question = _build_query(current_turn_query)
        simplified_conversation = [
            {"role": strip_text(message.get("role", "")), "content": strip_text(message.get("content", ""))}
            for message in conversation
        ]
        yield {
            "id": strip_text(row.get("conversation_id", "")) or f"lmsys_chat_{written}",
            "dataset": "lmsys-chat-1m",
            "question": question,
            "query": question,
            "solution": gold,
            "gold": gold,
            "references": build_text_references(contexts, prefix="round"),
            "agent_reference_contexts": contexts,
            "agent_extract_contexts": list(contexts),
            "agent_reference_contexts_truth": [gold for _ in contexts],
            "conversation_id": strip_text(row.get("conversation_id", "")),
            "model": strip_text(row.get("model", "")),
            "turn": int(row.get("turn", 0) or 0),
            "language": strip_text(row.get("language", "")),
            "redacted": bool(row.get("redacted", False)),
            "history_messages": history_before_question,
            "current_turn_query": current_turn_query,
            "query_turn": {
                "role": "user",
                "content": current_turn_query,
            },
            "conversation": simplified_conversation,
            "context_count": len(contexts),
            "openai_moderation": row.get("openai_moderation", []),
            "split": split,
        }
        written += 1
        if max_rows > 0 and written >= max_rows:
            break


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract LMSYS-Chat multi-turn samples into a WildChat-style query/context/answer format."
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, default="data/lmsys-chat/processed")
    parser.add_argument("--output_file", type=str, default="train_all_lang_multi_turn_min3.jsonl")
    parser.add_argument("--min_turns", type=int, default=3)
    parser.add_argument("--max_rows", type=int, default=-1)
    parser.add_argument("--drop_redacted", action="store_true")
    parser.add_argument("--no_require_all_turns_english", action="store_true")
    parser.add_argument(
        "--english_only",
        action="store_true",
        help="Keep only English conversations.",
    )
    args = parser.parse_args()

    output_path = Path(args.output_dir) / args.output_file
    count = write_jsonl(
        str(output_path),
        iter_lmsys_rows(
            split=args.split,
            min_turns=args.min_turns,
            require_all_turns_english=not args.no_require_all_turns_english,
            drop_redacted=args.drop_redacted,
            include_all_languages=not args.english_only,
            max_rows=args.max_rows,
        ),
    )
    print(f"[lmsys-chat] wrote {count} rows to {output_path}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
