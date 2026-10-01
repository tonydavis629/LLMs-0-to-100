"""Finding the bundled text and turning token IDs back into text, provided for you.

You do NOT need to edit this file.
"""

from __future__ import annotations

from pathlib import Path

import torch


def find_data_file() -> Path:
    """Walk up from this file to find data/tinyshakespeare.txt."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "data" / "tinyshakespeare.txt"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate data/tinyshakespeare.txt")


def decode(itos: dict[int, str], ids: torch.Tensor) -> str:
    """Turn a 1-D tensor of token IDs back into text."""
    return "".join(itos[int(i)] for i in ids)


def print_indented(text: str) -> None:
    """Print text indented under its label, one line at a time."""
    for line in text.split("\n"):
        print(f"    {line}")
