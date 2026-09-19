"""Step 7: score_multiple_choice()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

from tests.check import Check, bad, ok


def _pick(name: str, got, expected: int, note: str = "") -> Check:
    """Pass if the chosen index is exactly the expected int."""
    if type(got) is int and got == expected:
        return ok(name)
    detail = f"expected index {expected}, got {got!r}"
    if not isinstance(got, int):
        detail += " (return the option's index, not its score)"
    elif note:
        detail += f" ({note})"
    return bad(name, detail)


def check_score_multiple_choice(score_multiple_choice) -> list[Check]:
    """Highest log-probability per token wins; nothing is generated."""
    checks = []

    # Same length everywhere: the highest (least negative) total wins
    checks.append(_pick("equal lengths: totals [-4, -2, -6] over 2 tokens each pick option 1",
                        score_multiple_choice([-4.0, -2.0, -6.0], [2, 2, 2]), 1,
                        "log-probabilities are negative: the best is the one closest to 0"))

    # The raw sum picks option 0 (-3 > -8), but per token option 1 wins (-2 > -3)
    checks.append(_pick("divides by length: [-3, -8] over [1, 4] tokens picks option 1 (-2.0 beats -3.0)",
                        score_multiple_choice([-3.0, -8.0], [1, 4]), 1,
                        "that is the raw-sum choice: divide each total by its number of tokens"))

    # Four options, like every question in the suite; the answer is the last one
    checks.append(_pick("four options: [-9, -6, -12, -7] over [3, 2, 3, 4] tokens pick option 3",
                        score_multiple_choice([-9.0, -6.0, -12.0, -7.0], [3, 2, 3, 4]), 3,
                        "per token the four options score -3.0, -3.0, -4.0, -1.75"))
    return checks
