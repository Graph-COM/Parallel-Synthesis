import json
import re
from typing import Any, Dict, Iterable, List


ANSWER_RE = re.compile(r"<answer>([\s\S]*?)</answer>", flags=re.IGNORECASE)


def _extract_after_think(text: str) -> str:
    rendered = str(text)
    tag = "</think>"
    idx = rendered.rfind(tag)
    if idx == -1:
        return rendered.strip()
    return rendered[idx + len(tag) :].strip()


def _extract_answer_payload(text: str) -> str:
    rendered = str(text)
    matches = [m.group(1).strip() for m in ANSWER_RE.finditer(rendered) if m.group(1).strip()]
    if matches:
        return matches[-1]
    return _extract_after_think(rendered).strip()


def _normalize_label(text: str) -> str:
    return str(text).strip().upper()


def _collect_string_values(value: Any, out: List[str]) -> None:
    if isinstance(value, str):
        out.append(value)
        return
    if isinstance(value, list):
        for item in value:
            _collect_string_values(item, out)
        return
    if isinstance(value, dict):
        for item in value.values():
            _collect_string_values(item, out)


def _collect_preferred_label_strings(value: Any, out: List[str]) -> bool:
    if not isinstance(value, dict):
        return False
    found = False
    for key in ("root_causes", "root_cause", "predicted_root_causes"):
        if key not in value:
            continue
        found = True
        _collect_string_values(value.get(key), out)
    return found


def _json_candidates(text: str) -> Iterable[Any]:
    payload = str(text).strip()
    candidates: List[str] = [payload]
    fence_match = re.finditer(r"```(?:json)?\s*([\s\S]*?)```", payload, flags=re.IGNORECASE)
    for match in fence_match:
        candidates.append(match.group(1).strip())

    for candidate in candidates:
        try:
            yield json.loads(candidate)
            continue
        except Exception:
            pass
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end > start:
            try:
                yield json.loads(candidate[start : end + 1])
            except Exception:
                pass
        start = candidate.find("[")
        end = candidate.rfind("]")
        if start != -1 and end > start:
            try:
                yield json.loads(candidate[start : end + 1])
            except Exception:
                pass


def extract_predicted_root_causes(
    raw_text: str,
    labels: List[str],
    *,
    max_labels: int = 2,
) -> List[str]:
    text = _extract_answer_payload(str(raw_text)).strip()
    normalized_to_label = {_normalize_label(label): str(label).strip() for label in labels}
    ordered: List[str] = []
    seen = set()

    def maybe_add(label_text: str) -> None:
        normalized = _normalize_label(label_text)
        label = normalized_to_label.get(normalized)
        if not label or label in seen:
            return
        seen.add(label)
        ordered.append(label)

    for parsed in _json_candidates(text):
        preferred_strings: List[str] = []
        if _collect_preferred_label_strings(parsed, preferred_strings):
            for value in preferred_strings:
                maybe_add(value)
                normalized_value = _normalize_label(value)
                for normalized_label, label in normalized_to_label.items():
                    if normalized_label in normalized_value and label not in seen:
                        seen.add(label)
                        ordered.append(label)
            continue

        strings: List[str] = []
        _collect_string_values(parsed, strings)
        for value in strings:
            maybe_add(value)
            normalized_value = _normalize_label(value)
            for normalized_label, label in normalized_to_label.items():
                if normalized_label in normalized_value and label not in seen:
                    seen.add(label)
                    ordered.append(label)

    if len(ordered) < max(1, max_labels):
        positions: List[tuple[int, str]] = []
        for label in labels:
            match = re.search(re.escape(label), text, flags=re.IGNORECASE)
            if not match:
                continue
            positions.append((match.start(), label))
        for _, label in sorted(positions, key=lambda item: item[0]):
            if label in seen:
                continue
            seen.add(label)
            ordered.append(label)

    if max_labels > 0:
        return ordered[:max_labels]
    return ordered


def score_marble_db_prediction(raw_text: str, item: Dict[str, Any]) -> Dict[str, Any]:
    gold = [str(x).strip() for x in (item.get("gold_root_causes", []) or []) if str(x).strip()]
    pred_num = int(item.get("number_of_labels_pred", 2) or 2)
    predicted = extract_predicted_root_causes(
        raw_text,
        [str(x).strip() for x in (item.get("labels", []) or []) if str(x).strip()],
        max_labels=pred_num,
    )
    hit_count = sum(1 for label in gold if label in predicted)
    task_score = (hit_count / len(gold)) if gold else 0.0
    return {
        "predicted_root_causes": predicted,
        "gold_root_causes": gold,
        "hit_count": hit_count,
        "task_score": float(task_score),
        "correct": bool(task_score >= 0.999999),
    }


def evaluate_marble_db_preds(preds: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(preds)
    correct = sum(1 for row in preds if bool(row.get("correct", False)))
    task_scores = [float(row.get("task_score", 0.0) or 0.0) for row in preds]
    ttfts = [
        float(row["ttft_sec"])
        for row in preds
        if row.get("ttft_sec") is not None
    ]
    cache_prepare_times = [
        float(row["cache_prepare_sec"])
        for row in preds
        if row.get("cache_prepare_sec") is not None
    ]
    avg_predicted_count = (
        round(
            sum(len(row.get("predicted_root_causes", []) or []) for row in preds) / total,
            4,
        )
        if total > 0
        else 0.0
    )
    return {
        "samples": total,
        "correct": correct,
        "accuracy": (correct / total) if total > 0 else 0.0,
        "avg_task_score": (sum(task_scores) / len(task_scores)) if task_scores else 0.0,
        "avg_predicted_root_causes": avg_predicted_count,
        "avg_ttft_sec": round(sum(ttfts) / len(ttfts), 4) if ttfts else None,
        "ttft_measured_samples": len(ttfts),
        "avg_cache_prepare_sec": (
            round(sum(cache_prepare_times) / len(cache_prepare_times), 4)
            if cache_prepare_times
            else None
        ),
        "cache_prepare_measured_samples": len(cache_prepare_times),
    }
