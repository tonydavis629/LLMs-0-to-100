"""Step 2: verifiable_reward()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

from tests.check import Check, bad, ok


def check_verifiable_reward(verifiable_reward) -> list[Check]:
    """A deterministic check: 1.0 for an exact match, 0.0 for anything else."""
    checks = []

    got = verifiable_reward("tac", "tac")
    checks.append(
        ok("an exact match scores 1.0: response 'tac', target 'tac'")
        if got == 1.0
        else bad("an exact match scores 1.0: response 'tac', target 'tac'", f"got {got!r}")
    )

    got = verifiable_reward("tca", "tac")
    checks.append(
        ok("a wrong answer scores 0.0: response 'tca', target 'tac'")
        if got == 0.0
        else bad("a wrong answer scores 0.0: response 'tca', target 'tac'", f"got {got!r}")
    )

    # Exact means the whole string: no credit for a prefix or a stray space
    got = [verifiable_reward("ta", "tac"), verifiable_reward("tac ", "tac")]
    checks.append(
        ok("no partial credit: 'ta' and 'tac ' both score 0.0 against 'tac'")
        if got == [0.0, 0.0]
        else bad("no partial credit: 'ta' and 'tac ' both score 0.0 against 'tac'",
                 f"got {got!r} (compare the whole strings with ==)")
    )

    # A bool would make score_group() build a bool tensor, which cannot be averaged
    got = verifiable_reward("tac", "tac")
    checks.append(
        ok("returns a Python float (1.0 or 0.0), not a bool")
        if type(got) is float
        else bad("returns a Python float (1.0 or 0.0), not a bool",
                 f"got {got!r} of type {type(got).__name__} (return 1.0 or 0.0)")
    )
    return checks
