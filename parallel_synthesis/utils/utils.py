import os
import random
import re
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def auto_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# this is to extract answer in \boxed{}
def extract_gsm8k_answer(text: str) -> Optional[str]:
    boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxes:
        content = boxes[-1]
        number = re.search(r"[-+]?\d+(?:\.\d+)?", content)
        return number.group(0) if number else content.strip()

    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1]
    return None


def extract_gold(text: str) -> Optional[str]:
    match = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", text)
    return match.group(1) if match else None


def normalize_answer(ans: Optional[str]) -> Optional[str]:
    if ans is None:
        return None
    return ans.strip().lower()


def extract_after_think(text: str) -> str:
    s = str(text)
    tag = "</think>"
    idx = s.rfind(tag)
    if idx == -1:
        return s.strip()
    return s[idx + len(tag) :].strip()


_FINAL_ANSWER_PREFIX_RE = re.compile(r"^(?:final answer|answer|final)\s*:\s*(.+)$", re.IGNORECASE)


def extract_context_qa_prediction(text: str) -> Optional[str]:
    final_text = extract_after_think(text).strip()
    lines = [line.strip() for line in final_text.splitlines() if line.strip()]

    def _extract_last_boxed_content_local(source: str) -> Optional[str]:
        rendered = str(source or "")
        token = "\\boxed{"
        start = rendered.rfind(token)
        if start < 0:
            return None
        idx = start + len(token)
        depth = 1
        content = []
        while idx < len(rendered):
            ch = rendered[idx]
            if ch == "{":
                depth += 1
                content.append(ch)
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return "".join(content).strip()
                content.append(ch)
            else:
                content.append(ch)
            idx += 1
        return None

    for source in (text, final_text):
        boxed = _extract_last_boxed_content_local(source)
        if boxed:
            return boxed
    for line in reversed(lines):
        match = _FINAL_ANSWER_PREFIX_RE.match(line)
        if match:
            candidate = match.group(1).strip()
            if candidate:
                return candidate
    if lines:
        return lines[-1]
    return final_text or None


def context_qa_match(item: Dict[str, Any], prediction: Optional[str]) -> bool:
    pred_norm = normalize_answer(prediction)
    if not pred_norm:
        return False
    aliases = item.get("answer_aliases_norm", [])
    if isinstance(aliases, list) and aliases:
        return pred_norm in {str(alias).strip() for alias in aliases if str(alias).strip()}
    gold_source = item.get("final_answer", item.get("gold", ""))
    gold_norm = normalize_answer(gold_source)
    return bool(gold_norm and pred_norm == gold_norm)


_ROUGE_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9@_]+")


def _rouge_tokenize(text: str) -> List[str]:
    return _ROUGE_TOKEN_PATTERN.findall(str(text).lower())


def _ngram_counts(tokens: List[str], n: int) -> Counter:
    if n <= 0 or len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _f1_from_counts(overlap: int, pred_total: int, ref_total: int) -> float:
    if pred_total == 0 or ref_total == 0 or overlap <= 0:
        return 0.0
    precision = overlap / pred_total
    recall = overlap / ref_total
    denom = precision + recall
    return (2.0 * precision * recall / denom) if denom > 0 else 0.0


def _rouge_n_f1(pred_tokens: List[str], ref_tokens: List[str], n: int) -> float:
    pred_counts = _ngram_counts(pred_tokens, n)
    ref_counts = _ngram_counts(ref_tokens, n)
    if not pred_counts or not ref_counts:
        return 0.0
    overlap = sum((pred_counts & ref_counts).values())
    return _f1_from_counts(overlap, sum(pred_counts.values()), sum(ref_counts.values()))


def _lcs_len(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    if len(a) < len(b):
        short, long_ = a, b
    else:
        short, long_ = b, a
    prev = [0] * (len(short) + 1)
    for tok in long_:
        curr = [0] * (len(short) + 1)
        for j in range(1, len(short) + 1):
            if tok == short[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = prev[j] if prev[j] >= curr[j - 1] else curr[j - 1]
        prev = curr
    return prev[-1]


def _rouge_l_f1(pred_tokens: List[str], ref_tokens: List[str]) -> float:
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_len(pred_tokens, ref_tokens)
    return _f1_from_counts(lcs, len(pred_tokens), len(ref_tokens))


def evaluate_rouge(preds: List[Dict]) -> Dict[str, float]:
    rouge1_scores: List[float] = []
    rouge2_scores: List[float] = []
    rougel_scores: List[float] = []

    for row in preds:
        gold = str(row.get("gold", "")).strip()
        pred_text = str(row.get("prediction", "")).strip()
        if not pred_text:
            pred_text = str(row.get("raw_prediction", "")).strip()
        if not gold:
            continue
        pred_tokens = _rouge_tokenize(pred_text)
        ref_tokens = _rouge_tokenize(gold)
        rouge1_scores.append(_rouge_n_f1(pred_tokens, ref_tokens, 1))
        rouge2_scores.append(_rouge_n_f1(pred_tokens, ref_tokens, 2))
        rougel_scores.append(_rouge_l_f1(pred_tokens, ref_tokens))

    denom = len(rouge1_scores)
    if denom == 0:
        return {
            "rouge_samples": 0,
            "rouge1_f1": 0.0,
            "rouge2_f1": 0.0,
            "rougel_f1": 0.0,
        }
    return {
        "rouge_samples": denom,
        "rouge1_f1": sum(rouge1_scores) / denom,
        "rouge2_f1": sum(rouge2_scores) / denom,
        "rougel_f1": sum(rougel_scores) / denom,
    }


def extract_markdown_python_block(text: str) -> Optional[str]:
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return None


# to run python
import traceback
from multiprocessing import Process, Manager
def run_with_timeout(code, timeout):
    def worker(ns, code):
        try:
            local_ns = {}
            exec(code, local_ns)
            ns['ok'] = True
            ns['error'] = None
        except Exception:
            ns['ok'] = False
            ns['error'] = traceback.format_exc()
    with Manager() as manager:
        ns = manager.dict()
        p = Process(target=worker, args=(ns, code))
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate()
            ns['ok'] = False
            ns['error'] = f"TimeoutError: Execution exceeded {timeout} seconds"
        return ns.get('ok', False), ns.get('error', None)
