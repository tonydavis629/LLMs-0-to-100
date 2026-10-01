"""Comparing generated answers with the expected ones, provided for you.

You do NOT need to edit this file.
"""

from __future__ import annotations


def answer_matches(pred: str, truth: str) -> bool:
    """Exact match, ignoring leading and trailing spaces."""
    return pred.strip() == truth.strip()


def verdict(pred: str, truth: str) -> str:
    """'(correct)', or '(wrong, want ...)' with the expected answer."""
    return "(correct)" if answer_matches(pred, truth) else f"(wrong, want {truth!r})"
