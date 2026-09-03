from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_module(module_name: str, relative_path: str) -> ModuleType:
    if module_name in sys.modules:
        return sys.modules[module_name]

    path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_context_dataset_utils() -> ModuleType:
    return _load_module("_processor_context_dataset_utils", "data/context_dataset_utils.py")


def load_dialogue_dataset_utils() -> ModuleType:
    return _load_module("_processor_dialogue_dataset_utils", "data/dialogue_dataset_utils.py")
