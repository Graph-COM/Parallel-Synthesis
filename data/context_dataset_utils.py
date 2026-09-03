import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def build_reference(
    text: str,
    *,
    citation: str = "",
    title: str = "",
    mid: str = "",
    **extra: Any,
) -> Optional[Dict[str, Any]]:
    abstract = str(text or "").strip()
    if not abstract:
        return None
    citation_text = str(citation or title or "").strip()
    title_text = str(title or citation_text).strip()
    ref: Dict[str, Any] = {
        "citation": citation_text,
        "mid": str(mid).strip(),
        "abstract": abstract,
    }
    if title_text:
        ref["title"] = title_text
    for key, value in extra.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        ref[key] = value
    return ref


def render_reference_context(ref: Dict[str, Any]) -> str:
    title = str(ref.get("title", "") or ref.get("citation", "")).strip()
    abstract = str(ref.get("abstract", "")).strip()
    if title and abstract:
        return f"[{title}]\n{abstract}"
    return abstract or title


def finalize_references(
    references: Iterable[Optional[Dict[str, Any]]],
    *,
    max_contexts: int = -1,
    exact_contexts: int = -1,
) -> Optional[Tuple[List[Dict[str, Any]], List[str]]]:
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for ref in references:
        if not ref:
            continue
        key = (
            str(ref.get("citation", "")).strip(),
            str(ref.get("title", "")).strip(),
            str(ref.get("abstract", "")).strip(),
        )
        if not key[2] or key in seen:
            continue
        seen.add(key)
        deduped.append(ref)

    if exact_contexts > 0 and len(deduped) != exact_contexts:
        return None
    if max_contexts > 0:
        deduped = deduped[:max_contexts]

    contexts = [render_reference_context(ref) for ref in deduped]
    return deduped, contexts


def build_context_qa_example(
    *,
    dataset: str,
    raw_split: str,
    example_id: str,
    question: str,
    answer: str,
    references: Iterable[Optional[Dict[str, Any]]],
    max_contexts: int = -1,
    exact_contexts: int = -1,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    finalized = finalize_references(
        references,
        max_contexts=max_contexts,
        exact_contexts=exact_contexts,
    )
    if finalized is None:
        return None
    refs, contexts = finalized
    row: Dict[str, Any] = {
        "id": str(example_id).strip(),
        "dataset": dataset,
        "raw_split": raw_split,
        "question": str(question).strip(),
        "solution": str(answer).strip(),
        "gold": str(answer).strip(),
        "answer": str(answer).strip(),
        "references": refs,
        "contexts": contexts,
        "agent_reference_contexts": contexts,
        "num_contexts": len(contexts),
    }
    if extra:
        row.update(extra)
    return row


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count
