"""Printing small matrices and vectors, provided for you.

You do NOT need to edit this file. The runner uses these to show the
5 x 5 attention matrices with one token label per row and column.
"""

from __future__ import annotations

import torch


def print_matrix(M: torch.Tensor, rows: list[str], columns: list[str] | None = None,
                 fmt: str = "{:8.4f}") -> None:
    """Print a small matrix with one label per row (and per column, if given)."""
    if columns:
        print("       " + "".join(f"{c:>8}" for c in columns))
    for label, row in zip(rows, M.tolist()):
        print(f"  {label:>4} " + "".join(fmt.format(v) for v in row))


def format_row(v: torch.Tensor) -> str:
    """One vector as a bracketed row of 3-decimal numbers."""
    return "[" + " ".join(f"{x:6.3f}" for x in v.tolist()) + "]"
