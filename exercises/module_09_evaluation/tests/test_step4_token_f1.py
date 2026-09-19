"""Step 4: token_f1()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import math

from tests.check import Check, bad, ok


def _close(name: str, got, expected: float, precision: float, recall: float) -> Check:
    """Pass if the F1 score is within 1e-6 of the expected value."""
    if isinstance(got, (int, float)) and math.isclose(got, expected, abs_tol=1e-6):
        return ok(name)
    shown = f"{got:.4f}" if isinstance(got, (int, float)) else repr(got)
    detail = f"expected {expected:.4f}, got {shown}"
    if isinstance(got, (int, float)):
        # Name the usual mix-ups when the number gives them away
        if math.isclose(got, (precision + recall) / 2, abs_tol=1e-6):
            detail += " (that is the arithmetic mean; F1 is the harmonic mean)"
        elif math.isclose(got, precision * recall / (precision + recall), abs_tol=1e-6):
            detail += " (did you forget the factor of 2?)"
    return bad(name, detail)


def check_token_f1(token_f1) -> list[Check]:
    """F1 = 2PR / (P + R): partial credit for overlapping tokens."""
    checks = []

    # 'it is bluu' vs 'it is blue': 2 of 3 tokens shared, so P = R = 2/3
    checks.append(_close("partial credit: 'it is bluu' against 'it is blue' has P = R = 2/3, so F1 = 0.667",
                         token_f1("it is bluu", "it is blue"), 2 / 3, 2 / 3, 2 / 3))

    # 'blue' vs 'it is blue': every predicted token is right (P = 1), but only
    # one of three reference tokens was produced (R = 1/3). 2 * 1 * 1/3 / (4/3) = 0.5
    checks.append(_close("P and R differ: 'blue' against 'it is blue' has P = 1, R = 1/3, so F1 = 0.5",
                         token_f1("blue", "it is blue"), 0.5, 1.0, 1 / 3))

    # Identical after normalization means perfect precision and recall
    checks.append(_close("a correct answer scores 1.0: 'It is blue.' against 'it is blue'",
                         token_f1("It is blue.", "it is blue"), 1.0, 1.0, 1.0))
    return checks
