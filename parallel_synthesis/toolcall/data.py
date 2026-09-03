import os
from functools import lru_cache
from typing import Dict, Iterable, List, Optional


def _pick_first(row: Dict, keys: List[str]) -> str:
    for key in keys:
        if key in row:
            val = row.get(key)
            if val is not None:
                s = str(val).strip()
                if s:
                    return s
    return ""


def _gaia_hf_file_candidates(file_ref: str, split: str) -> List[str]:
    ref = str(file_ref).strip().lstrip("/")
    if not ref:
        return []
    if ref.startswith(("http://", "https://")):
        return [ref]

    candidates: List[str] = []
    if ref.startswith("2023/") or ref.startswith("2024/"):
        candidates.append(ref)
    else:
        candidates.append(f"2023/{split}/{ref}")
        # Keep robust fallbacks across GAIA split naming/layouts.
        candidates.append(f"2023/validation/{ref}")
        candidates.append(f"2023/test/{ref}")
        candidates.append(ref)

    seen = set()
    uniq: List[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


@lru_cache(maxsize=8192)
def _download_gaia_file_from_hf(file_ref: str, split: str) -> str:
    """
    Best-effort attachment resolver for GAIA rows loaded from HF.
    Returns local cached absolute path when found; otherwise returns original ref.
    """
    ref = str(file_ref).strip()
    if not ref:
        return ref
    if os.path.isabs(ref) and os.path.exists(ref):
        return ref

    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        return ref

    for candidate in _gaia_hf_file_candidates(ref, split):
        try:
            return hf_hub_download(
                repo_id="gaia-benchmark/GAIA",
                repo_type="dataset",
                filename=candidate,
            )
        except Exception:
            continue
    return ref


def _maybe_files(
    row: Dict,
    *,
    split: str,
) -> List[str]:
    out: List[str] = []
    for key in [
        "file_path",
        "file_name",
        "file",
        "files",
        "attachment",
        "attachments",
    ]:
        if key not in row:
            continue
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            out.append(val.strip())
        elif isinstance(val, list):
            for x in val:
                sx = str(x).strip()
                if sx:
                    out.append(sx)

    cleaned = []
    for x in out:
        if os.path.isabs(x) and os.path.exists(x):
            cleaned.append(x)
        else:
            cleaned.append(_download_gaia_file_from_hf(x, split))
    # preserve order + dedupe
    seen = set()
    uniq = []
    for p in cleaned:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq


def _normalize_split_for_gaia(split: str) -> str:
    s = str(split).strip().lower()
    if s in {"val", "validation"}:
        return "validation"
    if s == "test":
        return "test"
    raise ValueError(
        "GAIA supports the validation split for local evaluation and the test "
        "split for official submission; there is no train split."
    )


def _normalize_gaia_config(config: str) -> Optional[str]:
    s = str(config or "").strip().lower()
    if not s or s == "auto":
        return None
    if s in {"level1", "l1"}:
        return "2023_level1"
    if s in {"level2", "l2"}:
        return "2023_level2"
    if s in {"level3", "l3"}:
        return "2023_level3"
    return str(config).strip()


def _try_load_gaia_from_hf(
    split: str,
    *,
    gaia_config: str = "2023_level1",
) -> List[Dict]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            "datasets is required for loading GAIA from Hugging Face. "
            "Install the project dependencies first."
        ) from exc

    normalized_config = _normalize_gaia_config(gaia_config)
    if normalized_config is None:
        candidates = [
            ("gaia-benchmark/GAIA", "2023_level1"),
            ("gaia-benchmark/GAIA", None),
        ]
    else:
        candidates = [("gaia-benchmark/GAIA", normalized_config)]
    last_err = None
    for ds_name, config in candidates:
        try:
            if config is None:
                ds = load_dataset(ds_name, split=split)
            else:
                ds = load_dataset(ds_name, config, split=split)
            return [dict(x) for x in ds]
        except Exception as exc:
            last_err = exc
            continue
    raise RuntimeError(
        "Failed to load GAIA from Hugging Face. Accept the dataset access "
        "conditions and authenticate with `huggingface-cli login` or `HF_TOKEN`."
    ) from last_err


def load_gaia_rows(
    *,
    split: str,
    gaia_config: str = "2023_level1",
    max_samples: int = -1,
) -> Iterable[Dict]:
    split_norm = _normalize_split_for_gaia(split)
    rows = _try_load_gaia_from_hf(split_norm, gaia_config=gaia_config)

    out: List[Dict] = []
    for row in rows:
        question = _pick_first(row, ["Question", "question", "problem", "query"])
        gold = _pick_first(row, ["Final answer", "final_answer", "answer", "Answer", "gold"])
        if not question:
            continue
        files = _maybe_files(
            row,
            split=split_norm,
        )
        out.append(
            {
                "question": question,
                "solution": gold,
                "gold": gold,
                "files": files,
                "meta": {"source_row": row},
            }
        )

    if max_samples > 0:
        out = out[:max_samples]

    for row in out:
        yield row
