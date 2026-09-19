"""Shared pieces for the per-step test files in this folder.

You do NOT need to edit anything in tests/. Each `test_step*.py` file holds
the checks for one step of the exercise. `src/main.py` runs them after it
prints that step's output. Every check calls YOUR function with small,
hand-made inputs whose correct answer is known, and reports CORRECT or
INCORRECT for each thing it looks at.

The test files take the student's function as an argument (instead of
importing it) so they never depend on `exercise.py` directly.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Check:
    """One CORRECT/INCORRECT result.

    name:   what was checked, in plain words (always shown)
    passed: True for CORRECT, False for INCORRECT
    detail: what went wrong, shown only when the check fails
    """

    name: str
    passed: bool
    detail: str = ""


def ok(name: str) -> Check:
    """Shorthand for a passing check."""
    return Check(name, True)


def bad(name: str, detail: str) -> Check:
    """Shorthand for a failing check, with a note on what went wrong."""
    return Check(name, False, detail)


def expect_equal(name: str, got, expected) -> Check:
    """Pass if got == expected, otherwise show both values."""
    if got == expected:
        return ok(name)
    return bad(name, f"expected {expected!r}\n        got      {got!r}")
