import argparse
import math
from typing import Any, Callable, Dict, List, Optional, Sequence


SUPPORTED_METHODS = {"parallel_kv", "fixed_parallel_kv"}
DEFAULT_MIXED_PRETRAIN_TASKS = [
    "wildchat",
    "ultrachat",
    "lmsys_chat",
    "toucan_single_parallel",
    "toucan_multi_parallel",
    "dta_tool",
    "flan",
    "2wiki_multihopqa",
]


def canonicalize_task_name(task: str) -> str:
    key = str(task or "").strip().lower().replace("-", "_")
    aliases = {
        "gsmk8": "gsm8k",
        "lmsys-chat": "lmsys_chat",
        "lmsyschat": "lmsys_chat",
        "lmsys": "lmsys_chat",
        "ultra_chat": "ultrachat",
        "2wiki": "2wiki_multihopqa",
        "2wikimultihopqa": "2wiki_multihopqa",
        "wild_chat": "wildchat",
        "toucan_single": "toucan_single_parallel",
        "toucan_multi": "toucan_multi_parallel",
        "dta": "dta_tool",
        "dta-tool": "dta_tool",
        "browsecomp_textmas": "browsecomp_textmas_toolcall",
        "browsecomp_textmas_toolcall": "browsecomp_textmas_toolcall",
        "browsecomp-textmas": "browsecomp_textmas_toolcall",
        "browsecomp-textmas-toolcall": "browsecomp_textmas_toolcall",
    }
    return aliases.get(key, key)


def safe_exp(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        return math.exp(float(value))
    except OverflowError:
        return None


def parse_csv(text: str) -> List[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def parse_task_caps(
    tasks: Sequence[str],
    raw_caps: str,
    *,
    default_cap_fn: Callable[[str], int],
) -> Dict[str, int]:
    if raw_caps.strip():
        parts = parse_csv(raw_caps)
        if len(parts) != len(tasks):
            raise ValueError(
                "Per-task cap list length must match --tasks "
                f"(got {len(parts)} caps for {len(tasks)} tasks)."
            )
        caps: Dict[str, int] = {}
        for task, raw in zip(tasks, parts):
            value = int(raw)
            if value == 0 or value < -1:
                raise ValueError("Per-task caps must be -1 (full) or positive integers.")
            caps[task] = value
        return caps

    caps: Dict[str, int] = {}
    for task in tasks:
        value = int(default_cap_fn(task))
        if value == 0 or value < -1:
            raise ValueError("Default task cap must be -1 (full) or positive integers.")
        caps[task] = value
    return caps


def set_method_task(method: Any, task: str) -> None:
    task = canonicalize_task_name(task)
    method.task = task
    if hasattr(method, "args") and method.args is not None:
        method.args.task = task


def clone_args_with_task(args: argparse.Namespace, task: str, split: str) -> argparse.Namespace:
    cloned = argparse.Namespace(**vars(args))
    cloned.task = canonicalize_task_name(task)
    cloned.split = split
    return cloned
