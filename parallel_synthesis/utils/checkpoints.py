from __future__ import annotations

import os
from pathlib import Path
from typing import Union


PathLike = Union[str, os.PathLike[str]]

_HUB_ALLOW_PATTERNS = [
    "cache_mapper.pt",
    "cache_mapper_config.json",
    "checkpoint_meta.json",
    "run_args.json",
    "judger_lora/*",
    "latest/cache_mapper.pt",
    "latest/cache_mapper_config.json",
    "latest/checkpoint_meta.json",
    "latest/run_args.json",
    "latest/judger_lora/*",
    "step_checkpoints/latest/cache_mapper.pt",
    "step_checkpoints/latest/cache_mapper_config.json",
    "step_checkpoints/latest/checkpoint_meta.json",
    "step_checkpoints/latest/run_args.json",
    "step_checkpoints/latest/judger_lora/*",
]


def has_parallel_kv_artifacts(path: Path) -> bool:
    return (path / "cache_mapper.pt").is_file() or (path / "judger_lora").is_dir()


def _resolve_local_checkpoint(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    if has_parallel_kv_artifacts(candidate):
        return candidate
    for child in (candidate / "latest", candidate / "step_checkpoints" / "latest"):
        if child.is_dir() and has_parallel_kv_artifacts(child):
            return child
    raise FileNotFoundError(
        f"Could not find Parallel Synthesis artifacts under {candidate}; expected "
        "cache_mapper.pt or judger_lora/."
    )


def _as_hub_repo_id(source: str) -> str | None:
    if source.startswith("hf://"):
        source = source[len("hf://") :]
    elif source.startswith(("/", "./", "../", "~")):
        return None

    parts = source.split("/")
    if len(parts) != 2 or not all(parts):
        return None
    return source


def resolve_parallel_kv_checkpoint(source: PathLike) -> Path:
    """Resolve a local checkpoint directory or download a Hugging Face Hub bundle."""

    raw_source = os.fspath(source).strip()
    if not raw_source:
        raise ValueError("Checkpoint source must not be empty.")

    local_candidate = Path(raw_source).expanduser()
    if local_candidate.exists():
        return _resolve_local_checkpoint(local_candidate)

    repo_id = _as_hub_repo_id(raw_source)
    if repo_id is None:
        return _resolve_local_checkpoint(local_candidate)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "Loading a Parallel Synthesis checkpoint from the Hugging Face Hub "
            "requires huggingface_hub. Install the project dependencies with "
            "`pip install -e .`."
        ) from exc

    try:
        snapshot_path = snapshot_download(
            repo_id=repo_id,
            allow_patterns=_HUB_ALLOW_PATTERNS,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not download Parallel Synthesis checkpoint {repo_id!r} from "
            "the Hugging Face Hub. If this was meant to be a local path, check "
            "that the directory exists."
        ) from exc

    return _resolve_local_checkpoint(Path(snapshot_path))


__all__ = ["has_parallel_kv_artifacts", "resolve_parallel_kv_checkpoint"]
