"""Step 2: normalize_answer()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

from tests.check import Check, bad, ok


def _same(name: str, got, expected: str, note: str = "") -> Check:
    """Pass if got == expected, otherwise show both strings."""
    if got == expected:
        return ok(name)
    return bad(name, f"expected {expected!r}\n        got      {got!r}" + (f"\n({note})" if note else ""))


def check_normalize_answer(normalize_answer) -> list[Check]:
    """Lowercase and punctuation are provided; the whitespace rule is yours."""
    checks = []

    # The docstring's own example: three strings, one answer
    variants = [normalize_answer(text) for text in ("4", " 4 ", "4.")]
    checks.append(
        ok("'4', ' 4 ' and '4.' all normalize to '4'")
        if variants == ["4", "4", "4"]
        else bad("'4', ' 4 ' and '4.' all normalize to '4'",
                 f"got {variants} (are the leading and trailing spaces removed?)")
    )

    # Runs of spaces and tabs in the middle shrink to one space
    got = normalize_answer("  It   is\tBlue. ")
    checks.append(_same("collapses internal whitespace: '  It   is\\tBlue. ' gives 'it is blue'",
                        got, "it is blue",
                        "strip() only trims the ends; split() then join() also fixes the middle"
                        if got == "it   is\tblue" else ""))

    # Collapsing is not deleting: single spaces between words stay
    got = normalize_answer("it is down")
    checks.append(_same("keeps one space between words: 'it is down' stays 'it is down'",
                        got, "it is down",
                        "join the pieces with a space, not with an empty string"
                        if got == "itisdown" else ""))

    # Nothing left after punctuation and whitespace are gone
    checks.append(_same("an answer of only spaces and punctuation normalizes to ''",
                        normalize_answer("  ...  "), ""))
    return checks
