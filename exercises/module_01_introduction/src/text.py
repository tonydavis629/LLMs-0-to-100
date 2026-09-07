"""Text-splitting helpers.

You do NOT need to edit this file.
"""

from __future__ import annotations


def tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer. Lowercase and split on whitespace."""
    return text.lower().split()
