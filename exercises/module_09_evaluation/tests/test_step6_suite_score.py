"""Step 6: suite_score()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import math

from tests.check import Check, bad, ok


def _close(name: str, got, expected: float, note: str = "") -> Check:
    """Pass if the score is within 1e-9 of the expected value."""
    if isinstance(got, (int, float)) and math.isclose(got, expected, abs_tol=1e-9):
        return ok(name)
    return bad(name, f"expected {expected}, got {got!r}" + (f" ({note})" if note else ""))


def check_suite_score(suite_score) -> list[Check]:
    """The headline number: every task counts the same, whatever its size."""
    checks = []

    # The instruct model's real exact-match scores: (0.8 + 1 + 0.75 + 1) / 4
    per_task = {"uppercase": 0.8, "repeat": 1.0, "reverse": 0.75, "qa": 1.0}
    got = suite_score(per_task)
    checks.append(_close("the instruct model's four task scores (80%, 100%, 75%, 100%) average to 0.8875",
                         got, 0.8875,
                         "did you divide by the number of tasks?" if isinstance(got, (int, float)) and got > 1 else ""))

    # Two tasks, one perfect and one zero: the midpoint, whatever their case counts
    checks.append(_close("one task at 1.0 and one at 0.0 give 0.5",
                         suite_score({"reverse": 1.0, "qa": 0.0}), 0.5))

    # With a single task, the suite is that task
    checks.append(_close("a one-task suite scores the same as its task: {'qa': 0.25} gives 0.25",
                         suite_score({"qa": 0.25}), 0.25))
    return checks
