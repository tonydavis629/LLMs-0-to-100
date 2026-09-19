"""Step 8: pass_at_k()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import math
from itertools import combinations

from tests.check import Check, bad, ok


def _fmt(value) -> str:
    """Show a number to 4 decimal places, anything else as it is."""
    return f"{value:.4f}" if isinstance(value, (int, float)) else repr(value)


def _close(name: str, got, expected: float, n: int, c: int, k: int) -> Check:
    """Pass if the estimate is within 1e-9 of the expected probability."""
    if isinstance(got, (int, float)) and math.isclose(got, expected, abs_tol=1e-9):
        return ok(name)
    detail = f"expected {expected:.4f}, got {_fmt(got)}"
    if isinstance(got, (int, float)):
        if math.isclose(got, 1 - expected, abs_tol=1e-9):
            detail += " (that is the chance of missing; subtract it from 1)"
        elif math.isclose(got, 1 - (1 - c / n) ** k, abs_tol=1e-9):
            detail += " (1 - (1 - c/n)**k draws with replacement; use the math.comb formula)"
    return bad(name, detail)


def _brute_force(n: int, c: int, k: int) -> float:
    """Count every k-subset of n samples (the first c correct) that holds a correct one."""
    subsets = list(combinations(range(n), k))
    hits = sum(1 for subset in subsets if any(i < c for i in subset))
    return hits / len(subsets)


def check_pass_at_k(pass_at_k) -> list[Check]:
    """1 - C(n - c, k) / C(n, k): the chance a random k-subset holds a correct sample."""
    checks = []

    # k = 1: one random draw is correct with probability c / n
    checks.append(_close("pass@1 is plain accuracy: n=5, c=2, k=1 gives 2/5 = 0.4",
                         pass_at_k(5, 2, 1), 0.4, 5, 2, 1))

    # Of the C(5, 3) = 10 ways to pick 3 samples, only 1 picks all three wrong ones
    checks.append(_close("n=5, c=2, k=3 gives 1 - C(3,3)/C(5,3) = 1 - 1/10 = 0.9",
                         pass_at_k(5, 2, 3), 0.9, 5, 2, 3))

    # No correct sample anywhere: no budget can find one
    checks.append(_close("no correct samples scores 0.0 at any k: n=5, c=0, k=5",
                         pass_at_k(5, 0, 5), 0.0, 5, 0, 5))

    # An independent check: enumerate every subset instead of using the formula
    wrong = [(c, k, pass_at_k(6, c, k), _brute_force(6, c, k))
             for c in range(7) for k in range(1, 7)]
    wrong = [w for w in wrong if not (isinstance(w[2], (int, float)) and math.isclose(w[2], w[3], abs_tol=1e-9))]
    checks.append(
        ok("agrees with counting every k-subset of n=6 samples, for every c and k")
        if not wrong
        else bad("agrees with counting every k-subset of n=6 samples, for every c and k",
                 f"{len(wrong)} of 42 cases differ, e.g. c={wrong[0][0]}, k={wrong[0][1]}: "
                 f"expected {wrong[0][3]:.4f}, got {_fmt(wrong[0][2])}")
    )
    return checks
