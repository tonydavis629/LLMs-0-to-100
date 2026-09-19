"""Step 5: task_accuracy()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import math

from tests.check import Check, bad, ok


def _matches(got, expected: dict[str, float]) -> bool:
    """True if got is a dict with the same tasks and (nearly) the same scores."""
    return (isinstance(got, dict) and set(got) == set(expected)
            and all(isinstance(got[t], (int, float)) and math.isclose(got[t], v, abs_tol=1e-9)
                    for t, v in expected.items()))


def check_task_accuracy(task_accuracy) -> list[Check]:
    """One mean per task: the breakdown the headline number hides."""
    checks = []

    # qa: 3 of 4 cases right; repeat: 1 of 2
    got = task_accuracy({"qa": [1.0, 0.0, 1.0, 1.0], "repeat": [0.0, 1.0]})
    expected = {"qa": 0.75, "repeat": 0.5}
    checks.append(
        ok("averages within each task: qa [1, 0, 1, 1] gives 0.75, repeat [0, 1] gives 0.5")
        if _matches(got, expected)
        else bad("averages within each task: qa [1, 0, 1, 1] gives 0.75, repeat [0, 1] gives 0.5",
                 f"expected {expected}\n        got      {got!r}"
                 + ("\n(return a dict with one entry per task, not one number)" if not isinstance(got, dict) else ""))
    )

    # F1 scores are fractions, not just 0 or 1
    got = task_accuracy({"qa": [0.5, 1.0, 0.0]})
    checks.append(
        ok("works on partial-credit scores: qa [0.5, 1.0, 0.0] gives 0.5")
        if _matches(got, {"qa": 0.5})
        else bad("works on partial-credit scores: qa [0.5, 1.0, 0.0] gives 0.5", f"got {got!r}")
    )

    # Each task divides by its OWN case count, not the total across tasks
    got = task_accuracy({"reverse": [1.0] * 24, "qa": [0.0] * 8})
    checks.append(
        ok("each task is divided by its own case count: 24 right of 24 is 1.0, 0 of 8 is 0.0")
        if _matches(got, {"reverse": 1.0, "qa": 0.0})
        else bad("each task is divided by its own case count: 24 right of 24 is 1.0, 0 of 8 is 0.0",
                 f"got {got!r} (divide by len(scores) for that task)")
    )
    return checks
