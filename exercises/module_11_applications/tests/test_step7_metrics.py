"""Step 7: recall_at_k() and reciprocal_rank()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import math

from tests.check import Check, bad, ok


def _score_check(name: str, got, expected: float, nudge: str = "") -> Check:
    """Pass if got is within 1e-9 of expected, otherwise show both values."""
    if isinstance(got, (int, float)) and math.isclose(got, expected, abs_tol=1e-9):
        return ok(name)
    shown = f"{got:.4f}" if isinstance(got, float) else repr(got)
    return bad(name, f"expected {expected:.4f}, got {shown}{nudge}")


def check_recall_at_k(recall_at_k) -> list[Check]:
    """Hand-made rankings where the cutoff k decides the answer."""
    ranking = ["a", "b", "c", "d"]
    return [
        # 'b' is 2nd, inside the top 2
        _score_check("recall@k: 'b' ranked 2nd is inside the top 2, so k=2 gives 1.0",
                     recall_at_k(ranking, ["b"], 2), 1.0,
                     " (look at ranked_ids[:k], the first k entries)"),
        # 'c' is 3rd, just past the cutoff
        _score_check("recall@k: 'c' ranked 3rd is just past the cutoff, so k=2 gives 0.0",
                     recall_at_k(ranking, ["c"], 2), 0.0,
                     " (ranked_ids[:k] already stops after k entries)"),
        # Two relevant documents, only 'b' is found in the top 3
        _score_check("recall@k is a fraction: 1 of the 2 relevant ['b', 'z'] in the top 3 gives 0.5",
                     recall_at_k(ranking, ["b", "z"], 3), 0.5,
                     " (divide by len(relevant_ids), not by k)"),
    ]


def check_reciprocal_rank(reciprocal_rank) -> list[Check]:
    """The right answer at 1st, 2nd, and 4th place."""
    ranking = ["a", "b", "c", "d"]
    first = reciprocal_rank(ranking, ["a"])
    second = reciprocal_rank(ranking, ["b"])
    fourth = reciprocal_rank(ranking, ["d"])
    return [
        _score_check("reciprocal rank: first place scores 1/1 = 1.0", first, 1.0,
                     " (position already starts at 1)"),
        _score_check("reciprocal rank: second place scores 1/2 = 0.5", second, 0.5,
                     " (position already starts at 1)"),
        _score_check("reciprocal rank: fourth place scores 1/4 = 0.25", fourth, 0.25),
    ]
