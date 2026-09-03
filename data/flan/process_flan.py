import argparse
import sys
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parallel_synthesis.processor_imports import load_context_dataset_utils, load_dialogue_dataset_utils


_context_dataset_utils = load_context_dataset_utils()
write_jsonl = _context_dataset_utils.write_jsonl

FLAN_DATASET = "ai2-adapt-dev/flan_v2_converted"
FLAN_REVISION = "20f17e144dfdfe2d20f12f140ace5b95c31577c0"

_dialogue_dataset_utils = load_dialogue_dataset_utils()
build_text_references = _dialogue_dataset_utils.build_text_references
strip_text = _dialogue_dataset_utils.strip_text


QUESTION_START_RE = re.compile(
    r"(?im)^(?:"
    r"The corresponding question:|"
    r"The question and answer are below\.?|"
    r"Question followed by answer:|"
    r"Question for this logic:|"
    r"Question\s*==>\s*|"
    r"Reverse engineering the question:|"
    r"What was the question\?|"
    r"So what could be the question\?|"
    r">Question<|"
    r"\*Question\*|"
    r"\[QUESTION\]|"
    r"\[Question\]|"
    r"QUESTION:|"
    r"Question:|"
    r"Q:"
    r")"
)

QUESTION_META_ONLY_RE = re.compile(
    r"(?i)^(?:"
    r"The corresponding question:|"
    r"The question and answer are below\.?|"
    r"Question followed by answer:|"
    r"Question for this logic:|"
    r"Question\s*==>\s*|"
    r"Reverse engineering the question:|"
    r"What was the question\?|"
    r"So what could be the question\?"
    r")\s*$"
)

INLINE_QUESTION_MARKER_RE = re.compile(
    r"(?i)^(?:"
    r"The corresponding question:|"
    r"Question followed by answer:|"
    r"Question for this logic:|"
    r"Question\s*==>\s*|"
    r">Question<|\*Question\*|\[QUESTION\]|\[Question\]|QUESTION:|Question:|Q:"
    r")\s*(.*)$"
)

TRAILING_PROMPT_LINE_RE = re.compile(
    r"(?i)^(?:"
    r"A:|"
    r"Answer:|"
    r"ANSWER:|"
    r"ANSWER W/ DETAILS:|"
    r"\[A\]:|"
    r"\[Ans\]:|"
    r"\[Answer\]:|"
    r"Student:|"
    r"standard solution:|"
    r"Answer|"
    r"ANSWER|"
    r"A"
    r")\s*$"
)

EXPLICIT_ANSWER_MARKER_RE = re.compile(
    r"(?i)^(?:"
    r">CoT<|"
    r"\*CoT\*|"
    r">Ans<|"
    r"\*Ans\*|"
    r"Answer:|"
    r"ANSWER:|"
    r"A:|"
    r"\[A\]:|"
    r"\[Answer\]:|"
    r"\[Ans\]:|"
    r"Explanation:|"
    r"Explanation and answer:|"
    r"Reasoning and answer:|"
    r"Detailed logic:|"
    r"Chain of thoughts:|"
    r"Let's solve it slowly:|"
    r"Let's think step by step\.?|"
    r"Stream of consciousness:"
    r")"
)

QUESTION_SIGNAL_RE = re.compile(
    r"(?i)(?:"
    r"\boptions:\b|"
    r"\bpremise:\b|"
    r"\bhypothesis:\b|"
    r"\bclaim:\b|"
    r"\bgiven the sentence\b|"
    r"\bcan we conclude\b|"
    r"\bdoes that mean\b|"
    r"\bis the hypothesis entailed\b|"
    r"\bwhich of the following sentences\b|"
    r"\bof the following\b|"
    r"\bdoesn't make sense\b"
    r")"
)

QUESTION_FALLBACK_START_RE = re.compile(
    r"(?im)^(?:"
    r"The corresponding question:|"
    r"The question and answer are below\.?|"
    r"Question followed by answer:|"
    r"Question for this logic:|"
    r"Question\s*==>\s*|"
    r"Reverse engineering the question:|"
    r"What was the question\?|"
    r"So what could be the question\?|"
    r">Question<|"
    r"\*Question\*|"
    r"\[QUESTION\]|\[Question\]|"
    r"QUESTION:|Question:|Q:|"
    r"Claim:|"
    r"Yes / no, is the following|"
    r"Is the following|"
    r"Given the sentence|"
    r"Can we conclude from|"
    r"If \"|"
    r"Based on this premise"
    r")"
)

SEPARATOR_LINE_RE = re.compile(r"\n\s*(?:\*{2,}|-{2,}|_{2,})\s*\n")


def _load_dataset(*args, **kwargs):
    from datasets import load_dataset

    return load_dataset(*args, **kwargs)


def _normalize_text(text: str) -> str:
    return str(text or "").replace("\r\n", "\n").replace("\r", "\n")


def _strip_separator_lines(text: str) -> str:
    lines = _normalize_text(text).splitlines()
    kept: List[str] = []
    for line in lines:
        if re.fullmatch(r"\s*(?:\*{2,}|-{2,}|_{2,})\s*", line):
            continue
        kept.append(line.rstrip())
    return "\n".join(kept).strip()


def _split_raw_blocks(text: str) -> List[str]:
    rendered = _normalize_text(text)
    parts = [strip_text(part) for part in SEPARATOR_LINE_RE.split(rendered) if strip_text(part)]
    if len(parts) > 1:
        base_parts = parts
    else:
        base_parts = [strip_text(part) for part in re.split(r"\n{2,}", rendered) if strip_text(part)]

    merged: List[str] = []
    for part in base_parts:
        if merged and EXPLICIT_ANSWER_MARKER_RE.match(strip_text(part)) and _looks_like_question_block(merged[-1]):
            merged[-1] = "\n\n".join([merged[-1], part]).strip()
            continue
        merged.append(part)
    return merged


def _looks_like_question_block(text: str) -> bool:
    rendered = strip_text(text)
    if not rendered:
        return False
    match = QUESTION_START_RE.search(rendered)
    if match and match.start() == 0:
        return True
    return bool(QUESTION_SIGNAL_RE.search(rendered))


def _strip_trailing_prompt_markers(text: str) -> str:
    lines = [line.rstrip() for line in _normalize_text(text).splitlines()]
    while lines and not strip_text(lines[-1]):
        lines.pop()
    while lines and TRAILING_PROMPT_LINE_RE.match(strip_text(lines[-1])):
        lines.pop()
        while lines and not strip_text(lines[-1]):
            lines.pop()
    return "\n".join(lines).strip()


def _normalize_question_text(text: str) -> str:
    lines = [line.rstrip() for line in _strip_separator_lines(text).splitlines()]
    normalized: List[str] = []
    for line in lines:
        rendered = strip_text(line)
        if not rendered:
            if normalized and normalized[-1]:
                normalized.append("")
            continue
        if QUESTION_META_ONLY_RE.match(rendered):
            continue
        match = INLINE_QUESTION_MARKER_RE.match(rendered)
        if match:
            rendered = strip_text(match.group(1))
            if not rendered:
                continue
        normalized.append(rendered)
    while normalized and not strip_text(normalized[-1]):
        normalized.pop()
    return _strip_trailing_prompt_markers("\n".join(normalized).strip())


def _normalize_answer_text(text: str) -> str:
    rendered = _strip_separator_lines(text)
    if not rendered:
        return ""
    lines = [line.rstrip() for line in rendered.splitlines()]
    normalized: List[str] = []
    for line in lines:
        current = strip_text(line)
        if not current:
            if normalized and normalized[-1]:
                normalized.append("")
            continue
        if QUESTION_META_ONLY_RE.match(current):
            continue
        for suffix in (
            "So what could be the question?",
            "What was the question?",
        ):
            if current.endswith(suffix):
                current = strip_text(current[: -len(suffix)]).rstrip(". ")
                break
        for marker in (
            ">CoT<",
            "*CoT*",
            ">Ans<",
            "*Ans*",
            "Explanation and answer:",
            "Reasoning and answer:",
            "Explanation:",
            "Detailed logic:",
            "Chain of thoughts:",
            "Stream of consciousness:",
            "Answer:",
            "ANSWER:",
            "A:",
            "[A]:",
            "[Ans]:",
            "[Answer]:",
        ):
            if current.startswith(marker):
                current = strip_text(current[len(marker) :])
                break
        normalized.append(current)
    while normalized and not strip_text(normalized[-1]):
        normalized.pop()
    return "\n".join(normalized).strip()


def _is_short_final_answer(text: str) -> bool:
    rendered = strip_text(text)
    if not rendered:
        return False
    if "\n" in rendered:
        return False
    if rendered.endswith(":") or "?" in rendered:
        return False
    if len(rendered) > 80:
        return False
    if len(rendered.split()) > 12:
        return False
    return True


def _looks_like_short_answer_line(text: str) -> bool:
    rendered = strip_text(text)
    if not _is_short_final_answer(rendered):
        return False
    if rendered.startswith("- "):
        return False
    if rendered.startswith('"') or rendered.startswith("'"):
        return False
    return True


def _split_inverse_pair_block(text: str) -> Optional[Tuple[str, str]]:
    rendered = strip_text(text)
    if not rendered:
        return None
    match = QUESTION_START_RE.search(rendered)
    if match and match.start() > 0:
        answer_part = strip_text(rendered[: match.start()])
        question_part = strip_text(rendered[match.start() :])
        if answer_part and question_part and _normalize_question_text(question_part):
            return answer_part, question_part
    return None


def _collect_inverse_pairs_from_option_questions(inputs: str) -> Tuple[List[Tuple[str, str]], str]:
    rendered = _normalize_text(inputs).strip()
    if "Options:" not in rendered:
        return [], strip_text(rendered)

    matches = list(QUESTION_FALLBACK_START_RE.finditer(rendered))
    if not matches:
        return [], strip_text(rendered)

    pairs: List[Tuple[str, str]] = []
    cursor = 0
    for idx, match in enumerate(matches):
        start = match.start()
        answer_part = strip_text(rendered[cursor:start])
        next_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(rendered)
        question_slice = rendered[start:next_start]
        lines = question_slice.splitlines(keepends=True)
        consumed = 0
        saw_options = False
        saw_option_line = False
        for line_idx, line in enumerate(lines):
            current = strip_text(line)
            if current.lower() == "options:":
                saw_options = True
            elif saw_options and current.startswith("- "):
                saw_option_line = True
            elif not saw_options and line_idx > 0 and EXPLICIT_ANSWER_MARKER_RE.match(current):
                break
            elif saw_options and saw_option_line and current:
                break
            consumed = line_idx + 1
        if consumed <= 0:
            continue
        question_part = "".join(lines[:consumed]).strip()
        normalized_question = _normalize_question_text(question_part)
        if answer_part and normalized_question:
            pairs.append((answer_part, question_part))
            cursor = start + len("".join(lines[:consumed]))
            continue
        break
    pending_answer = strip_text(rendered[cursor:])
    return pairs, pending_answer


def _collect_inverse_pairs(inputs: str) -> Tuple[List[Tuple[str, str]], str]:
    pairs: List[Tuple[str, str]] = []
    pending_answer = ""
    for block in _split_raw_blocks(inputs):
        split_pair = _split_inverse_pair_block(block)
        if split_pair is not None:
            answer_part, question_part = split_pair
            if pending_answer:
                answer_part = "\n\n".join([pending_answer, answer_part]).strip()
                pending_answer = ""
            pairs.append((answer_part, question_part))
            continue
        if _looks_like_question_block(block):
            if pending_answer:
                pairs.append((pending_answer, block))
                pending_answer = ""
            continue
        question_match = QUESTION_START_RE.search(block)
        if question_match and question_match.start() > 0:
            answer_prefix = strip_text(block[: question_match.start()])
            if answer_prefix:
                pending_answer = "\n\n".join(
                    part for part in [pending_answer, answer_prefix] if strip_text(part)
                ).strip()
            continue
        pending_answer = "\n\n".join(part for part in [pending_answer, block] if strip_text(part)).strip()
    if len(pairs) >= 2:
        return pairs, pending_answer
    fallback_pairs, fallback_pending = _collect_inverse_pairs_from_option_questions(inputs)
    if len(fallback_pairs) >= 2:
        return fallback_pairs, fallback_pending
    return pairs, pending_answer


def _extract_target_question_and_answer(target: str) -> Tuple[str, str]:
    rendered = _strip_separator_lines(target)
    if not rendered:
        return "", ""
    lines = [line.rstrip() for line in rendered.splitlines()]
    for idx, line in enumerate(lines):
        current = strip_text(line)
        if current.startswith("- "):
            continue
        if EXPLICIT_ANSWER_MARKER_RE.match(current):
            question_part = "\n".join(lines[:idx]).strip()
            answer_part = "\n".join(lines[idx:]).strip()
            return question_part, answer_part
    question_prefix = "\n".join(lines[:-1]).strip()
    if (
        len(lines) >= 2
        and "Options:" in question_prefix
        and _looks_like_question_block(question_prefix)
        and _looks_like_short_answer_line(lines[-1])
    ):
        question_part = "\n".join(lines[:-1]).strip()
        answer_part = lines[-1].strip()
        return question_part, answer_part
    return rendered, ""


def _merge_inverse_answer_parts(input_answer: str, target_answer: str) -> str:
    left = _normalize_answer_text(input_answer)
    right = _normalize_answer_text(target_answer)
    if not left:
        return right
    if not right:
        return left
    if _is_short_final_answer(left) and not _is_short_final_answer(right):
        return "\n".join(part for part in [right, left] if strip_text(part)).strip()
    if _is_short_final_answer(right) and not _is_short_final_answer(left):
        return "\n".join(part for part in [left, right] if strip_text(part)).strip()
    return "\n".join(part for part in [left, right] if strip_text(part)).strip()


def _render_context(idx: int, pair_text: str) -> str:
    rendered = strip_text(pair_text)
    if not rendered:
        return ""
    return f"This is the example question {idx}:\n{rendered}"


def _render_inverse_context(idx: int, question_text: str, answer_text: str) -> str:
    question = strip_text(question_text)
    answer = strip_text(answer_text)
    if not question or not answer:
        return ""
    return (
        f"This is the example question {idx}:\n"
        f"{question}\n\n"
        f"This is the corresponding answer {idx}:\n"
        f"{answer}"
    )


def _build_query(question_text: str) -> str:
    rendered_question = strip_text(question_text)
    return (
        "Given the previous in-context examples, answer the following question.\n\n"
        "Reason step by step using the patterns demonstrated in the examples, and then provide the final answer "
        "clearly and directly.\n\n"
        f"Question:\n{rendered_question}"
    )


def _build_forward_row(row: Dict[str, Any], *, written: int) -> Optional[Dict[str, Any]]:
    blocks = _split_raw_blocks(row.get("inputs", ""))
    if len(blocks) < 3:
        return None
    previous_blocks = blocks[:-1]
    if len(previous_blocks) < 2:
        return None
    query = _normalize_question_text(blocks[-1])
    gold = strip_text(row.get("targets", ""))
    if not query or not gold:
        return None
    contexts = [
        _render_context(idx, block)
        for idx, block in enumerate(previous_blocks, start=1)
        if strip_text(block)
    ]
    contexts = [context for context in contexts if strip_text(context)]
    if len(contexts) < 2:
        return None
    question = _build_query(query)
    return {
        "id": f"flan_{written}",
        "dataset": "flan_v2_converted",
        "question": question,
        "query": question,
        "solution": gold,
        "gold": gold,
        "references": build_text_references(contexts, prefix="example"),
        "agent_reference_contexts": contexts,
        "agent_extract_contexts": list(contexts),
        "_task_name": strip_text(row.get("_task_name", "")),
        "_task_source": strip_text(row.get("_task_source", "")),
        "_template_type": strip_text(row.get("_template_type", "")),
        "_template_idx": row.get("_template_idx", None),
        "previous_example_count": len(contexts),
        "is_inverse_task": False,
        "current_question": query,
        "inputs": strip_text(row.get("inputs", "")),
        "targets": gold,
        "messages": row.get("messages", []),
    }


def _build_inverse_row(row: Dict[str, Any], *, written: int) -> Optional[Dict[str, Any]]:
    pairs, current_answer_prefix = _collect_inverse_pairs(row.get("inputs", ""))
    if len(pairs) < 2:
        return None
    question_part, target_answer_part = _extract_target_question_and_answer(row.get("targets", ""))
    query = _normalize_question_text(question_part)
    gold = _merge_inverse_answer_parts(current_answer_prefix, target_answer_part)
    if not query or not gold:
        return None
    contexts = []
    for idx, (answer_text, question_text) in enumerate(pairs, start=1):
        pair_question_part, pair_target_answer_part = _extract_target_question_and_answer(question_text)
        contexts.append(
            _render_inverse_context(
                idx,
                _normalize_question_text(pair_question_part),
                _merge_inverse_answer_parts(answer_text, pair_target_answer_part),
            )
        )
    contexts = [context for context in contexts if strip_text(context)]
    if len(contexts) < 2:
        return None
    question = _build_query(query)
    return {
        "id": f"flan_{written}",
        "dataset": "flan_v2_converted",
        "question": question,
        "query": question,
        "solution": gold,
        "gold": gold,
        "references": build_text_references(contexts, prefix="example"),
        "agent_reference_contexts": contexts,
        "agent_extract_contexts": list(contexts),
        "_task_name": strip_text(row.get("_task_name", "")),
        "_task_source": strip_text(row.get("_task_source", "")),
        "_template_type": strip_text(row.get("_template_type", "")),
        "_template_idx": row.get("_template_idx", None),
        "previous_example_count": len(contexts),
        "is_inverse_task": True,
        "current_question": query,
        "inverse_input_answer_prefix": _normalize_answer_text(current_answer_prefix),
        "inverse_target_answer_suffix": _normalize_answer_text(target_answer_part),
        "inputs": strip_text(row.get("inputs", "")),
        "targets": strip_text(row.get("targets", "")),
        "messages": row.get("messages", []),
    }


def iter_flan_rows(*, split: str, max_rows: int) -> Iterable[Dict[str, Any]]:
    ds = _load_dataset(
        FLAN_DATASET,
        split=split,
        streaming=True,
        revision=FLAN_REVISION,
    )
    written = 0
    for row in ds:
        task_name = strip_text(row.get("_task_name", ""))
        if task_name.endswith("ii"):
            built = _build_inverse_row(row, written=written)
        else:
            built = _build_forward_row(row, written=written)
        if built is None:
            continue
        yield built
        written += 1
        if max_rows > 0 and written >= max_rows:
            break


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert FLAN v2 converted rows into query/context/answer examples with few-shot contexts."
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, default="data/flan/processed")
    parser.add_argument("--output_file", type=str, default="train_in_context_examples.jsonl")
    parser.add_argument("--max_rows", type=int, default=-1)
    args = parser.parse_args()

    output_path = Path(args.output_dir) / args.output_file
    count = write_jsonl(
        str(output_path),
        iter_flan_rows(split=args.split, max_rows=args.max_rows),
    )
    print(f"[flan] wrote {count} rows to {output_path}")


if __name__ == "__main__":
    main()
