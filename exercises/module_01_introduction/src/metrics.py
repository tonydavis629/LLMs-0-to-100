"""Evaluation metrics for the n-gram models.

You do NOT need to edit this file. `perplexity()` is the companion to the
extra-credit `cross_entropy()` you write in `exercise.py`.
"""

from __future__ import annotations

from collections import Counter


def perplexity(text: str, model: dict[str, Counter]) -> float:
    """Compute perplexity: 2^(cross_entropy).

    Companion to the extra-credit `cross_entropy()` in `exercise.py`.
    The import is inside the function so this module never depends on
    student code at import time.
    """
    from exercise import cross_entropy

    return 2 ** cross_entropy(text, model)
