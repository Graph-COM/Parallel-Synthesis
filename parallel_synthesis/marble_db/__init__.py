"""MARBLE database benchmark support."""

from .backend import build_db_session
from .data import load_marble_db_rows
from .scoring import evaluate_marble_db_preds, score_marble_db_prediction

try:
    from .methods import ParallelKVMarbleDBMethod, TextMASMarbleDBMethod
except Exception:
    ParallelKVMarbleDBMethod = None
    TextMASMarbleDBMethod = None

__all__ = [
    "build_db_session",
    "load_marble_db_rows",
    "TextMASMarbleDBMethod",
    "ParallelKVMarbleDBMethod",
    "score_marble_db_prediction",
    "evaluate_marble_db_preds",
]
