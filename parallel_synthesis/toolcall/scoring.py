import re
from typing import Optional

from parallel_synthesis.utils.utils import extract_after_think


def extract_answer_text(raw_text: str) -> str:
    text = extract_after_think(str(raw_text))
    # Peel answer tags repeatedly in case the model emits nested wrappers like
    # <answer><answer>...</answer></answer>.
    while True:
        m = re.search(r"<answer>([\s\S]*?)</answer>", text, flags=re.IGNORECASE)
        if not m:
            break
        extracted = m.group(1).strip()
        if extracted == text.strip():
            break
        text = extracted
    return text.strip()


def normalize_free_form(text: Optional[str]) -> str:
    if text is None:
        return ""
    s = str(text).strip().lower()
    s = re.sub(r"\\s+", " ", s)
    s = s.strip(" \n\t\r\"'`.,;:!?()[]{}")
    return s


def _to_float_if_possible(text: str) -> Optional[float]:
    try:
        cleaned = text.replace(",", "")
        return float(cleaned)
    except Exception:
        return None


def free_form_exact_match(pred: str, gold: str) -> bool:
    p = normalize_free_form(pred)
    g = normalize_free_form(gold)
    if not p or not g:
        return False
    if p == g:
        return True

    pf = _to_float_if_possible(p)
    gf = _to_float_if_possible(g)
    if pf is not None and gf is not None:
        if abs(pf - gf) <= 1e-9:
            return True
    return False
