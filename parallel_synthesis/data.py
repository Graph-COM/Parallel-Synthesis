import argparse
import gzip
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from parallel_synthesis.utils.utils import extract_gold, normalize_answer


def _load_dataset(*args, **kwargs):
    from datasets import load_dataset

    return load_dataset(*args, **kwargs)


def _open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _read_jsonl(path: Path) -> Iterable[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find JSONL file: {path}")
    with _open_text(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _render_context_list(contexts: List[str]) -> str:
    return "\n\n".join(str(ctx).strip() for ctx in contexts if str(ctx).strip()).strip()


def _normalize_context_list(raw: Optional[List[str]]) -> List[str]:
    if not isinstance(raw, list):
        return []
    return [str(ctx).strip() for ctx in raw if str(ctx).strip()]


def _normalize_context_qa_split(task: str, split: str) -> str:
    if task != "2wiki_multihopqa":
        raise ValueError(f"Unsupported context-QA task: {task}")
    split_norm = str(split or "").strip().lower()
    if split_norm in {"validation", "valid", "val"}:
        return "validation"
    if split_norm not in {"train", "validation", "dev", "test"}:
        raise ValueError(f"Unsupported 2Wiki split: {split!r}")
    return split_norm


def _context_qa_processed_path(task: str, split: str) -> Path:
    split_norm = _normalize_context_qa_split(task, split)
    return Path("data/2wiki-multihopqa/processed") / f"{split_norm}.jsonl"


def _prepare_context_qa_row(task: str, row: Dict) -> Optional[Dict]:
    question = str(row.get("question", row.get("query", ""))).strip()
    gold = str(row.get("gold", row.get("answer", row.get("solution", "")))).strip()
    contexts_raw = row.get("contexts")
    if not isinstance(contexts_raw, list):
        contexts_raw = row.get("agent_reference_contexts")
    contexts = _normalize_context_list(contexts_raw)
    if not question or not gold or not contexts:
        return None

    prepared = dict(row)
    prepared["query"] = str(row.get("query", question)).strip() or question
    prepared["question"] = question
    final_answer = str(
        row.get("final_answer", row.get("raw_answer", row.get("solution", "")))
    ).strip()
    prepared["solution"] = str(row.get("solution", final_answer or gold)).strip() or gold
    prepared["gold"] = gold
    prepared["answer"] = str(row.get("answer", gold)).strip() or gold
    if final_answer:
        prepared["final_answer"] = final_answer
    prepared["gold_norm"] = normalize_answer(final_answer or prepared["solution"] or gold)
    prepared["contexts"] = contexts
    prepared["full_context"] = _render_context_list(contexts)
    reference_contexts = _normalize_context_list(row.get("agent_reference_contexts"))
    if not reference_contexts:
        reference_contexts = list(contexts)
    extract_contexts = _normalize_context_list(row.get("agent_extract_contexts"))
    if not extract_contexts:
        extract_contexts = list(reference_contexts)
    prepared["agent_reference_contexts"] = reference_contexts
    prepared["agent_extract_contexts"] = extract_contexts

    aliases = row.get("normalized_aliases") or row.get("answer_aliases") or []
    alias_norms = []
    if isinstance(aliases, list):
        for alias in aliases:
            normalized = normalize_answer(str(alias))
            if normalized:
                alias_norms.append(normalized)
    if prepared["gold_norm"]:
        alias_norms.append(prepared["gold_norm"])
    prepared["answer_aliases_norm"] = sorted(set(alias_norms))
    return prepared


def _load_context_qa_processed(task: str, split: str) -> Iterable[Dict]:
    path = _context_qa_processed_path(task, split)
    for row in _read_jsonl(path):
        prepared = _prepare_context_qa_row(task, row)
        if prepared is not None:
            yield prepared


def _validate_aime_full_test_split(split: str) -> str:
    split_norm = str(split or "").strip().lower()
    if split_norm in {"validation", "val"}:
        split_norm = "valid"
    if split_norm not in {"train", "valid", "test"}:
        raise ValueError(
            f"Unsupported split '{split}' for AIME. Use train, valid/validation, or test."
        )
    return "test"


def _canonicalize_task_name(task: str) -> str:
    key = str(task or "").strip().lower().replace("-", "_")
    aliases = {
        "gsmk8": "gsm8k",
        "lmsys-chat": "lmsys_chat",
        "lmsyschat": "lmsys_chat",
        "lmsys": "lmsys_chat",
        "ultra_chat": "ultrachat",
        "aime_2024": "aime2024",
        "aime_2025": "aime2025",
        "gpqa_diamond": "gpqa",
        "2wiki": "2wiki_multihopqa",
        "2wikimultihopqa": "2wiki_multihopqa",
        "wild_chat": "wildchat",
        "toucan_single": "toucan_single_parallel",
        "toucan_multi": "toucan_multi_parallel",
        "dta": "dta_tool",
        "browsecomp_textmas": "browsecomp_textmas_toolcall",
        "browsecomp_textmas_toolcall": "browsecomp_textmas_toolcall",
        "browsecomp-textmas": "browsecomp_textmas_toolcall",
        "browsecomp-textmas-toolcall": "browsecomp_textmas_toolcall",
    }
    return aliases.get(key, key)


def _require_train_split(task: str, split: str) -> None:
    split_norm = str(split or "").strip().lower()
    if split_norm != "train":
        raise ValueError(
            f"clean_parallel_kv uses the full original train split for {task}. "
            f"Unsupported split requested: {split!r}."
        )


def load_gsm8k(split: str = "test", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = _load_dataset("gsm8k", "main", split=split, cache_dir=cache_dir)
    for item in ds:
        question = item["question"].strip()
        solution = item["answer"]
        gold = normalize_answer(extract_gold(solution))
        yield {
            "question": question,
            "solution": solution,
            "gold": gold,
        }


def load_2wiki_multihopqa(split: str = "train") -> Iterable[Dict]:
    return _load_context_qa_processed("2wiki_multihopqa", split)


def load_aime2025(
    split: str = "test",
    cache_dir: Optional[str] = None,
    seed: int = 42,
) -> Iterable[Dict]:
    _ = seed
    _validate_aime_full_test_split(split)
    ds = _load_dataset("yentinglin/aime_2025", split="train", cache_dir=cache_dir)
    for item in ds:
        problem = item["problem"].strip()
        answer = str(item["answer"]).strip()
        gold = normalize_answer(answer)
        yield {
            "question": problem,
            "solution": answer,
            "gold": gold,
        }


def load_aime2024(
    split: str = "test",
    cache_dir: Optional[str] = None,
    seed: int = 42,
) -> Iterable[Dict]:
    _ = seed
    _validate_aime_full_test_split(split)
    ds = _load_dataset("HuggingFaceH4/aime_2024", split="train", cache_dir=cache_dir)
    for item in ds:
        problem = item["problem"].strip()
        answer = str(item["answer"]).strip()
        gold = normalize_answer(answer)
        yield {
            "question": problem,
            "solution": answer,
            "gold": gold,
        }


def load_gpqa_diamond(split: str = "test", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = _load_dataset("fingertap/GPQA-Diamond", split=split, cache_dir=cache_dir)
    for item in ds:
        question = item["question"].strip()
        answer = item["answer"].strip()
        gold = normalize_answer(answer)
        yield {
            "question": question,
            "solution": answer,
            "gold": gold,
        }


def load_mbppplus(
    split: str = "test",
    subset: str = None,
    cache_dir: Optional[str] = None,
) -> Iterable[Dict]:
    ds = _load_dataset("evalplus/mbppplus", subset, split=split, cache_dir=cache_dir)
    for item in ds:
        question = f"""Please provide a self-contained Python script that solves the following problem in a markdown code block:\n```python\nYOUR_PYTHON_CODE\n```:
{item["prompt"]}
Your answer will be tested on test cases like:
{item["test_list"][0]}
{item["test_list"][1]}
{item["test_list"][2]}
"""

        answer = str(item["test"])
        gold = answer
        yield {
            "question": question,
            "solution": answer,
            "gold": gold,
        }


def load_humanevalplus(
    split: str = "test",
    subset: str = None,
    cache_dir: Optional[str] = None,
) -> Iterable[Dict]:
    ds = _load_dataset("evalplus/humanevalplus", subset, split=split, cache_dir=cache_dir)
    for item in ds:
        question = f"""Please provide a self-contained Python script that solves the following problem in a markdown code block:\n```python\nYOUR_PYTHON_CODE\n```:
{item["prompt"]}
"""
        raw_answer = str(item["test"])
        answer = raw_answer.replace("candidate", item["entry_point"])
        answer += f'\n\ncheck({item["entry_point"]})'
        gold = answer
        yield {
            "question": question,
            "solution": answer,
            "gold": gold,
        }


def _format_medqa_question(question: str, options: Dict[str, str]) -> str:
    rendered_question = str(question).strip()
    option_lines = []
    for label in ("A", "B", "C", "D"):
        text = str(options.get(label, "")).strip()
        if text:
            option_lines.append(f"{label.lower()}: {text}")
    if not option_lines:
        raise ValueError("MedQA sample is missing options A-D.")
    return rendered_question + "\n" + "\n".join(option_lines)


def _normalize_medqa_options(raw_options: object) -> Dict[str, str]:
    if isinstance(raw_options, dict):
        return {
            label: str(raw_options.get(label, "")).strip()
            for label in ("A", "B", "C", "D")
        }
    return {}


def load_medqa_source(
    split: str = "train",
    *,
    cache_dir: Optional[str] = None,
) -> Iterable[Dict]:
    ds = _load_dataset("araag2/MedQA", "source", split=split, cache_dir=cache_dir)
    for item in ds:
        question = str(item.get("question", "")).strip()
        answer_idx = str(item.get("answer_idx", "")).strip().upper()
        options = _normalize_medqa_options(item.get("options", {}))
        if answer_idx not in {"A", "B", "C", "D"}:
            raise ValueError(f"Invalid MedQA answer_idx from source split {split!r}: {answer_idx!r}")
        yield {
            "question": _format_medqa_question(question, options),
            "solution": answer_idx.lower(),
            "gold": normalize_answer(answer_idx),
            "answer_text": str(item.get("answer", "")).strip(),
            "meta_info": str(item.get("meta_info", "")).strip(),
            "metamap_phrases": item.get("metamap_phrases", []),
        }


def load_medqa(
    split: str = "train",
    subset: str = None,
    cache_dir: Optional[str] = None,
    seed: int = 42,
) -> Iterable[Dict]:
    _ = subset
    _ = seed
    return load_medqa_source(split=split, cache_dir=cache_dir)


def _load_processed_parallel_text_rows(
    path: Path,
    *,
    task_name: str,
    max_contexts: int = -1,
) -> Iterable[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing processed {task_name} file: {path}.")

    required = ["question", "gold", "references", "agent_reference_contexts"]
    with _open_text(path) as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Invalid row at {path}:{line_no}; expected JSON object.")
            missing = [key for key in required if key not in row]
            if missing:
                raise ValueError(f"Missing keys at {path}:{line_no}: {missing}")

            references = row.get("references", [])
            contexts = row.get("agent_reference_contexts", [])
            truth_contexts = row.get("agent_reference_contexts_truth")
            if not isinstance(references, list) or not isinstance(contexts, list):
                raise ValueError(
                    f"Invalid references/context types at {path}:{line_no}; both must be lists."
                )
            if truth_contexts is not None and not isinstance(truth_contexts, list):
                raise ValueError(
                    f"Invalid agent_reference_contexts_truth type at {path}:{line_no}; "
                    "must be a list when provided."
                )

            if max_contexts > 0:
                references = references[:max_contexts]
                contexts = contexts[:max_contexts]
                if isinstance(truth_contexts, list):
                    truth_contexts = truth_contexts[:max_contexts]
            extract_contexts = row.get("agent_extract_contexts")
            if isinstance(extract_contexts, list) and max_contexts > 0:
                extract_contexts = extract_contexts[:max_contexts]

            out = dict(row)
            out["references"] = references
            out["agent_reference_contexts"] = contexts
            if isinstance(extract_contexts, list):
                out["agent_extract_contexts"] = extract_contexts
            if isinstance(truth_contexts, list):
                out["agent_reference_contexts_truth"] = truth_contexts
            if "solution" not in out:
                out["solution"] = str(out.get("gold", ""))
            yield out


def _normalize_processed_parallel_text_split(task_name: str, split: str) -> str:
    split_norm = str(split or "").strip().lower()
    if split_norm in {"val", "valid"}:
        split_norm = "validation"
    if split_norm not in {"train", "validation"}:
        raise ValueError(f"{task_name} processing supports train and validation splits only.")
    return split_norm


def _processed_parallel_text_split_path(
    *,
    data_dir: str,
    split: str,
    task_name: str,
    train_file: str,
    validation_file: str,
) -> Path:
    split_norm = _normalize_processed_parallel_text_split(task_name, split)
    filename = train_file if split_norm == "train" else validation_file
    return Path(data_dir) / filename


def load_wildchat(
    split: str = "train",
    data_dir: str = "./data/wildchat/processed",
    *,
    max_contexts: int = -1,
) -> Iterable[Dict]:
    path = _processed_parallel_text_split_path(
        data_dir=data_dir,
        split=split,
        task_name="WildChat",
        train_file="train_all_lang_multi_turn_min3.jsonl",
        validation_file="validation_all_lang_multi_turn_min3.jsonl",
    )
    return _load_processed_parallel_text_rows(path, task_name="wildchat", max_contexts=max_contexts)


def load_lmsys_chat(
    split: str = "train",
    data_dir: str = "./data/lmsys-chat/processed",
    *,
    max_contexts: int = -1,
) -> Iterable[Dict]:
    path = _processed_parallel_text_split_path(
        data_dir=data_dir,
        split=split,
        task_name="LMSYS-Chat",
        train_file="train_all_lang_multi_turn_min3.jsonl",
        validation_file="validation_all_lang_multi_turn_min3.jsonl",
    )
    return _load_processed_parallel_text_rows(path, task_name="lmsys-chat", max_contexts=max_contexts)


def load_ultrachat(
    split: str = "train",
    data_dir: str = "./data/ultrachat/processed",
    *,
    max_contexts: int = -1,
) -> Iterable[Dict]:
    path = _processed_parallel_text_split_path(
        data_dir=data_dir,
        split=split,
        task_name="UltraChat",
        train_file="train_sft_multi_turn_min3.jsonl",
        validation_file="validation_sft_multi_turn_min3.jsonl",
    )
    return _load_processed_parallel_text_rows(path, task_name="ultrachat", max_contexts=max_contexts)


def load_flan(
    split: str = "train",
    data_dir: str = "./data/flan/processed",
    *,
    max_contexts: int = -1,
) -> Iterable[Dict]:
    path = _processed_parallel_text_split_path(
        data_dir=data_dir,
        split=split,
        task_name="FLAN",
        train_file="train_in_context_examples.jsonl",
        validation_file="validation_in_context_examples.jsonl",
    )
    return _load_processed_parallel_text_rows(path, task_name="flan", max_contexts=max_contexts)


def load_toucan_single_parallel(
    split: str = "train",
    data_dir: str = "./data/toucan/processed",
    *,
    max_contexts: int = -1,
) -> Iterable[Dict]:
    path = _processed_parallel_text_split_path(
        data_dir=data_dir,
        split=split,
        task_name="Toucan single-turn",
        train_file="train_single_turn_parallel_tool_call_unified.jsonl",
        validation_file="validation_single_turn_parallel_tool_call_unified.jsonl",
    )
    return _load_processed_parallel_text_rows(
        path,
        task_name="toucan-single-turn",
        max_contexts=max_contexts,
    )


def load_toucan_multi_parallel(
    split: str = "train",
    data_dir: str = "./data/toucan/processed",
    *,
    max_contexts: int = -1,
) -> Iterable[Dict]:
    path = _processed_parallel_text_split_path(
        data_dir=data_dir,
        split=split,
        task_name="Toucan multi-turn",
        train_file="train_multi_turn_parallel_tool_call.jsonl",
        validation_file="validation_multi_turn_parallel_tool_call.jsonl",
    )
    return _load_processed_parallel_text_rows(
        path,
        task_name="toucan-multi-turn",
        max_contexts=max_contexts,
    )


def load_dta_tool(
    split: str = "train",
    data_dir: str = "./data/dta-tool/processed",
    *,
    max_contexts: int = -1,
) -> Iterable[Dict]:
    path = _processed_parallel_text_split_path(
        data_dir=data_dir,
        split=split,
        task_name="DTA-Tool",
        train_file="train_parallel_function_calls.jsonl",
        validation_file="validation_parallel_function_calls.jsonl",
    )
    return _load_processed_parallel_text_rows(path, task_name="dta-tool", max_contexts=max_contexts)


def load_browsecomp_textmas_toolcall(
    split: str = "train",
    data_dir: str = "./data/browsecomp-textmas/processed",
    *,
    data_file: Optional[str] = None,
    max_contexts: int = -1,
) -> Iterable[Dict]:
    _require_train_split("browsecomp_textmas_toolcall", split)
    path = Path(data_file) if data_file else Path(data_dir) / "filtered_train.jsonl.gz"
    return _load_processed_parallel_text_rows(
        path,
        task_name="browsecomp-textmas-toolcall",
        max_contexts=max_contexts,
    )


def build_dataset_iter(
    task: str,
    split: str,
    seed: int,
    args: Optional[argparse.Namespace] = None,
):
    task = _canonicalize_task_name(task)
    if task == "gsm8k":
        return load_gsm8k(split=split)
    if task == "2wiki_multihopqa":
        return load_2wiki_multihopqa(split=split)
    if task == "aime2024":
        return load_aime2024(split=split, seed=seed)
    if task == "aime2025":
        return load_aime2025(split=split, seed=seed)
    if task == "gpqa":
        return load_gpqa_diamond(split=split)
    if task == "mbppplus":
        return load_mbppplus(split=split)
    if task == "humanevalplus":
        return load_humanevalplus(split=split)
    if task == "medqa":
        return load_medqa(split=split, seed=seed)
    if task == "wildchat":
        return load_wildchat(split=split)
    if task == "lmsys_chat":
        return load_lmsys_chat(split=split)
    if task == "ultrachat":
        return load_ultrachat(split=split)
    if task == "flan":
        return load_flan(split=split)
    if task == "toucan_single_parallel":
        return load_toucan_single_parallel(split=split)
    if task == "toucan_multi_parallel":
        return load_toucan_multi_parallel(split=split)
    if task == "dta_tool":
        return load_dta_tool(split=split)
    if task == "browsecomp_textmas_toolcall":
        return load_browsecomp_textmas_toolcall(
            split=split,
            data_file=(
                str(getattr(args, "browsecomp_textmas_data_file", "")).strip()
                if args is not None
                else None
            )
            or None,
        )
    raise ValueError(f"no {task} support")


__all__ = [
    "_load_dataset",
    "build_dataset_iter",
    "load_2wiki_multihopqa",
    "load_aime2024",
    "load_aime2025",
    "load_browsecomp_textmas_toolcall",
    "load_dta_tool",
    "load_flan",
    "load_gsm8k",
    "load_gpqa_diamond",
    "load_humanevalplus",
    "load_lmsys_chat",
    "load_mbppplus",
    "load_medqa",
    "load_medqa_source",
    "load_toucan_multi_parallel",
    "load_toucan_single_parallel",
    "load_ultrachat",
    "load_wildchat",
]
