import argparse
import json
import sys
from pathlib import Path
import shutil
from typing import Any, Dict, Iterable, List, Optional
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parallel_synthesis.processor_imports import load_context_dataset_utils


_context_dataset_utils = load_context_dataset_utils()
finalize_references = _context_dataset_utils.finalize_references
build_reference = _context_dataset_utils.build_reference
write_jsonl = _context_dataset_utils.write_jsonl

TWOWIKI_REVISION = "612bc5039a457880d9e7d84c3b0a4cf154b70e4f"

def _load_parquet_rows(path: Path):
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    return table.to_pylist()


PARQUET_URLS = {
    split: (
        "https://huggingface.co/datasets/xanhho/2WikiMultihopQA/resolve/"
        f"{TWOWIKI_REVISION}/{split}.parquet"
    )
    for split in ("train", "dev", "test")
}


def _find_cached_parquet(split_norm: str) -> Optional[Path]:
    parquet_name = f"{split_norm}.parquet"
    search_roots = [
        PROJECT_ROOT / "data/2wiki-multihopqa/raw" / TWOWIKI_REVISION,
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    for root in search_roots:
        direct_candidate = root / parquet_name
        if direct_candidate.exists():
            return direct_candidate
        dataset_root = root / "datasets--xanhho--2WikiMultihopQA" / "snapshots"
        if not dataset_root.exists():
            continue
        candidate = dataset_root / TWOWIKI_REVISION / parquet_name
        if candidate.exists():
            return candidate
    return None


def _download_parquet(split_norm: str) -> Path:
    output_dir = PROJECT_ROOT / "data/2wiki-multihopqa/raw" / TWOWIKI_REVISION
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split_norm}.parquet"
    if output_path.exists():
        return output_path

    with urlopen(PARQUET_URLS[split_norm], timeout=300) as response, output_path.open("wb") as fh:
        shutil.copyfileobj(response, fh)
    return output_path


def _normalize_split(split: str) -> str:
    split_norm = split.lower().strip()
    if split_norm in {"validation", "valid", "val"}:
        return "dev"
    if split_norm not in {"train", "dev", "test"}:
        raise ValueError(f"Unsupported 2Wiki split '{split}'. Use train, dev, or test.")
    return split_norm


def _normalize_sentences(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(part).strip() for part in value if str(part).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _maybe_json_loads(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _load_2wiki_split(split: str):
    split_norm = _normalize_split(split)
    cached_parquet = _find_cached_parquet(split_norm)
    if cached_parquet is not None:
        return _load_parquet_rows(cached_parquet)
    return _load_parquet_rows(_download_parquet(split_norm))


def _iter_context_references(item: Dict[str, Any]) -> Iterable[Optional[Dict[str, Any]]]:
    context = _maybe_json_loads(item.get("context"))
    if isinstance(context, dict):
        titles = list(context.get("title", []) or [])
        sentences_by_title = list(
            context.get("sentences", context.get("content", context.get("text", []))) or []
        )
        for idx, title in enumerate(titles):
            sentences = _normalize_sentences(sentences_by_title[idx] if idx < len(sentences_by_title) else [])
            yield build_reference(
                " ".join(sentences),
                citation=str(title).strip() or f"context_{idx + 1}",
                title=str(title).strip(),
                sentences=sentences,
                context_index=idx,
            )
        return

    if not isinstance(context, list):
        return
    for idx, entry in enumerate(context):
        title = ""
        sentences: List[str] = []
        if isinstance(entry, dict):
            title = str(entry.get("title", "")).strip()
            sentences = _normalize_sentences(
                entry.get("sentences", entry.get("content", entry.get("text", [])))
            )
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            title = str(entry[0]).strip()
            sentences = _normalize_sentences(entry[1])
        else:
            continue
        yield build_reference(
            " ".join(sentences),
            citation=title or f"context_{idx + 1}",
            title=title,
            sentences=sentences,
            context_index=idx,
        )


def _normalize_supporting_facts(value: Any) -> List[Dict[str, Any]]:
    value = _maybe_json_loads(value)
    if isinstance(value, dict):
        titles = list(value.get("title", []) or [])
        sent_ids = list(value.get("sent_id", value.get("sent_ids", [])) or [])
        return [
            {"title": str(title).strip(), "sent_id": int(sent_ids[idx]) if idx < len(sent_ids) else -1}
            for idx, title in enumerate(titles)
        ]
    if isinstance(value, list):
        normalized: List[Dict[str, Any]] = []
        for entry in value:
            if isinstance(entry, dict):
                normalized.append(
                    {
                        "title": str(entry.get("title", "")).strip(),
                        "sent_id": int(entry.get("sent_id", entry.get("sentence_id", -1))),
                    }
                )
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                normalized.append({"title": str(entry[0]).strip(), "sent_id": int(entry[1])})
        return normalized
    return []


def _format_query(question: str) -> str:
    question_text = str(question).strip()
    return (
        "You are given an original multi-hop question and a set of contexts retrieved by different explorers.\n\n"
        f"Original question: {question_text}\n\n"
        "Instructions:\n"
        "1. Briefly summarize each retrieved context.\n"
        "2. Decide which contexts support answering the original question.\n"
        "3. Answer the original question.\n"
        "4. Present the final answer in the format \\box{...}."
    )


def _ordered_supporting_titles(
    references: List[Dict[str, Any]],
    supporting_facts: List[Dict[str, Any]],
) -> List[str]:
    support_set = {str(fact.get("title", "")).strip() for fact in supporting_facts if str(fact.get("title", "")).strip()}
    ordered: List[str] = []
    seen = set()
    for ref in references:
        title = str(ref.get("title", "") or ref.get("citation", "")).strip()
        if title and title in support_set and title not in seen:
            seen.add(title)
            ordered.append(title)
    for title in sorted(support_set):
        if title not in seen:
            ordered.append(title)
    return ordered


def _format_gold(
    answer: str,
    references: List[Dict[str, Any]],
    supporting_titles: List[str],
) -> str:
    explorer_titles = []
    for idx, ref in enumerate(references, start=1):
        title = str(ref.get("title", "") or ref.get("citation", "")).strip() or f"context {idx}"
        explorer_titles.append(f"{idx}). {title}")
    supporting_text = ", ".join(supporting_titles) if supporting_titles else "none"
    return (
        "<think>\n"
        f"The explorer-provided contexts are: {', '.join(explorer_titles)}.\n"
        f"The contexts that support the answer are: {supporting_text}.\n"
        "</think>\n"
        f"\\box{{{str(answer).strip()}}}"
    )


def _build_row(
    *,
    split_norm: str,
    item: Dict[str, Any],
    row_idx: int,
    question: str,
    answer: str,
    supporting_facts: List[Dict[str, Any]],
    evidences: Any,
    max_contexts: int,
    exact_contexts: int,
) -> Optional[Dict[str, Any]]:
    finalized = finalize_references(
        _iter_context_references(item),
        max_contexts=max_contexts,
        exact_contexts=exact_contexts,
    )
    if finalized is None:
        return None
    refs, contexts = finalized
    paragraph_contexts = [str(ref.get("abstract", "")).strip() for ref in refs]
    supporting_titles = _ordered_supporting_titles(refs, supporting_facts)
    formatted_query = _format_query(question)
    formatted_gold = _format_gold(answer, refs, supporting_titles)
    return {
        "id": str(item.get("_id", item.get("id", row_idx))),
        "dataset": "2wiki_multihopqa",
        "raw_split": split_norm,
        "question": formatted_query,
        "query": formatted_query,
        "original_question": question,
        "solution": answer,
        "final_answer": answer,
        "gold": formatted_gold,
        "answer": formatted_gold,
        "references": refs,
        "contexts": contexts,
        "agent_reference_contexts": paragraph_contexts,
        "agent_extract_contexts": list(paragraph_contexts),
        "num_contexts": len(paragraph_contexts),
        "supporting_facts": supporting_facts,
        "supporting_context_titles": supporting_titles,
        "context_titles": [
            str(ref.get("title", "") or ref.get("citation", "")).strip() or f"context_{idx + 1}"
            for idx, ref in enumerate(refs)
        ],
        "evidences": evidences if isinstance(evidences, list) else [],
        "type": str(item.get("type", "")).strip(),
        "level": str(item.get("level", "")).strip(),
    }


def iter_2wiki_rows(
    split: str,
    *,
    max_contexts: int,
    exact_contexts: int,
    max_rows: int = -1,
) -> Iterable[Dict[str, Any]]:
    split_norm = _normalize_split(split)
    ds = _load_2wiki_split(split_norm)
    written = 0
    for row_idx, item in enumerate(ds):
        if max_rows > 0 and written >= max_rows:
            break
        answer = str(item.get("answer", "")).strip()
        question = str(item.get("question", "")).strip()
        if not question or not answer:
            continue
        supporting_facts = _normalize_supporting_facts(item.get("supporting_facts"))
        evidences = _maybe_json_loads(item.get("evidences", []))
        example = _build_row(
            split_norm=split_norm,
            item=item,
            row_idx=row_idx,
            question=question,
            answer=answer,
            supporting_facts=supporting_facts,
            evidences=evidences,
            max_contexts=max_contexts,
            exact_contexts=exact_contexts,
        )
        if example is not None:
            written += 1
            yield example


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process HF 2WikiMultihopQA into context-QA JSONL."
    )
    parser.add_argument("--output_dir", type=str, default="data/2wiki-multihopqa/processed")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "dev", "test"],
        choices=["train", "dev", "test", "validation", "valid", "val"],
    )
    parser.add_argument(
        "--max_contexts",
        type=int,
        default=-1,
        help="Maximum number of context documents to keep per sample. Use -1 for all.",
    )
    parser.add_argument(
        "--exact_contexts",
        type=int,
        default=-1,
        help="Keep only samples with exactly this many context documents. Use -1 to disable.",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=-1,
        help="Maximum number of processed rows to write per split. Use -1 for all.",
    )
    args = parser.parse_args()

    if args.exact_contexts > 0 and args.max_contexts > 0 and args.exact_contexts > args.max_contexts:
        raise ValueError("--exact_contexts cannot be greater than --max_contexts.")

    total = 0
    for split in args.splits:
        split_norm = _normalize_split(split)
        output_path = Path(args.output_dir) / f"{split_norm}.jsonl"
        count = write_jsonl(
            str(output_path),
            iter_2wiki_rows(
                split_norm,
                max_contexts=args.max_contexts,
                exact_contexts=args.exact_contexts,
                max_rows=args.max_rows,
            ),
        )
        print(f"[2wiki-multihopqa] wrote {count} rows to {output_path}")
        total += count
    print(f"[2wiki-multihopqa] done. Total rows written: {total}")


if __name__ == "__main__":
    main()
