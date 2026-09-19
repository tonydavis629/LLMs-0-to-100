"""Step 3: exact_match()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

from tests.check import Check, bad, ok

# exact_match() compares normalized strings, so a wrong Step 2 breaks it too
STEP2_NOTE = "exact_match relies on normalize_answer: is Step 2 CORRECT?"


def _score(name: str, got, expected: float, note: str) -> Check:
    """Pass if the score equals the expected 1.0 or 0.0."""
    if got == expected:
        return ok(name)
    return bad(name, f"expected {expected}, got {got!r} ({note})")


def check_exact_match(exact_match) -> list[Check]:
    """All-or-nothing: the normalized prediction must equal one acceptable answer."""
    checks = []

    # Case, punctuation, and spaces are formatting, not content
    checks.append(_score("formatting does not count: ' Blue. ' against ['blue'] scores 1.0",
                         exact_match(" Blue. ", ["blue"]), 1.0,
                         f"did you normalize the prediction? {STEP2_NOTE}"))

    # The second acceptable answer, written with its own formatting, still counts
    checks.append(_score("any acceptable answer counts: 'it is blue' against ['blue', 'It is blue.'] scores 1.0",
                         exact_match("it is blue", ["blue", "It is blue."]), 1.0,
                         "check every answer in the list, and normalize each one too"))

    # Exact means exact: a correct word inside a longer answer is still a miss
    checks.append(_score("an extra word scores zero: 'it is blue' against ['blue'] scores 0.0",
                         exact_match("it is blue", ["blue"]), 0.0,
                         "compare with ==, not with `in`"))

    # The runner adds these scores up, so they should be numbers, not booleans
    hit, miss = exact_match("blue", ["blue"]), exact_match("red", ["blue"])
    checks.append(
        ok("returns the float 1.0 or 0.0, not True or False")
        if type(hit) is float and type(miss) is float and (hit, miss) == (1.0, 0.0)
        else bad("returns the float 1.0 or 0.0, not True or False",
                 f"got {hit!r} for a hit and {miss!r} for a miss (wrap the any(...) in float())")
    )
    return checks
