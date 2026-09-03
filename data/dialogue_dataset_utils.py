import ast
import json
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def strip_text(value: Any) -> str:
    return str(value or "").strip()


def split_sentences(text: str) -> List[str]:
    parts = _SENTENCE_SPLIT_RE.split(strip_text(text))
    return [part.strip() for part in parts if part and part.strip()]


def split_truth_into_agent_chunks(text: str, num_agents: int) -> List[str]:
    if num_agents <= 0:
        return []
    sentences = split_sentences(text)
    if not sentences:
        return [""] * num_agents

    total = len(sentences)
    base = total // num_agents
    remainder = total % num_agents
    sizes = [base] * num_agents
    for idx in range(remainder):
        sizes[num_agents - remainder + idx] += 1

    chunks: List[str] = []
    start = 0
    for size in sizes:
        end = start + size
        chunks.append(" ".join(sentences[start:end]).strip())
        start = end
    return chunks


def _normalize_role(role: Any) -> str:
    rendered = strip_text(role).lower().replace(" ", "_")
    if not rendered:
        rendered = "unknown"
    return rendered


def render_messages(
    messages: Sequence[Dict[str, Any]],
    *,
    role_key: str = "role",
    content_key: str = "content",
    start_index: int = 1,
) -> str:
    parts: List[str] = []
    for idx, message in enumerate(messages, start=start_index):
        if not isinstance(message, dict):
            continue
        role = _normalize_role(message.get(role_key, ""))
        content = strip_text(message.get(content_key, ""))
        if not content:
            continue
        parts.append(f"[{role.upper()}_{idx}]\n{content}")
    return "\n\n".join(parts).strip()


def build_text_references(
    texts: Sequence[str],
    *,
    prefix: str,
    mids: Optional[Sequence[str]] = None,
    extra_by_index: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    references: List[Dict[str, Any]] = []
    for idx, text in enumerate(texts, start=1):
        abstract = strip_text(text)
        if not abstract:
            continue
        ref: Dict[str, Any] = {
            "citation": f"{prefix}_{idx}",
            "mid": strip_text(mids[idx - 1]) if mids and idx - 1 < len(mids) else "",
            "abstract": abstract,
        }
        if extra_by_index and idx - 1 < len(extra_by_index):
            extra = extra_by_index[idx - 1] or {}
            for key, value in extra.items():
                if value is None:
                    continue
                if isinstance(value, str) and not value.strip():
                    continue
                ref[key] = value
        references.append(ref)
    return references


def extract_balanced_block(text: str, marker: str, *, open_char: str = "[", close_char: str = "]") -> Optional[str]:
    rendered = str(text or "")
    marker_idx = rendered.find(marker)
    if marker_idx < 0:
        return None
    start = marker_idx + len(marker)
    while start < len(rendered) and rendered[start].isspace():
        start += 1
    if start >= len(rendered) or rendered[start] != open_char:
        return None

    depth = 0
    in_string = False
    escape = False
    quote_char = ""
    for idx in range(start, len(rendered)):
        ch = rendered[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote_char:
                in_string = False
            continue
        if ch in {'"', "'"}:
            in_string = True
            quote_char = ch
            continue
        if ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                return rendered[start : idx + 1]
    return None


def parse_json_or_python_literal(text: str) -> Optional[Any]:
    rendered = strip_text(text)
    if not rendered:
        return None
    try:
        return json.loads(rendered)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(rendered)
    except (SyntaxError, ValueError):
        return None


def normalize_role_content_messages(
    messages: Iterable[Dict[str, Any]],
    *,
    role_key: str,
    content_key: str,
) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = strip_text(message.get(role_key, ""))
        content = strip_text(message.get(content_key, ""))
        if not role or not content:
            continue
        normalized.append({"role": role, "content": content})
    return normalized


def count_tree_depth(node: Dict[str, Any], *, child_key: str = "children") -> int:
    children = node.get(child_key, [])
    if not isinstance(children, list) or not children:
        return 1
    return 1 + max(count_tree_depth(child, child_key=child_key) for child in children)


def count_tree_nodes(node: Dict[str, Any], *, child_key: str = "children") -> int:
    children = node.get(child_key, [])
    total = 1
    if isinstance(children, list):
        for child in children:
            total += count_tree_nodes(child, child_key=child_key)
    return total


def sort_rank_created(item: Dict[str, Any]) -> Tuple[int, str, str]:
    rank = item.get("rank")
    if rank is None:
        rank_value = 10**9
    else:
        try:
            rank_value = int(rank)
        except (TypeError, ValueError):
            rank_value = 10**9
    created = strip_text(item.get("created_date", ""))
    message_id = strip_text(item.get("message_id", item.get("id", "")))
    return rank_value, created, message_id
