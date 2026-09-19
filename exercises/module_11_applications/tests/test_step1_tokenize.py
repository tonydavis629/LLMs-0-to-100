"""Step 1: tokenize()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

from tests.check import Check, expect_equal


def check_tokenize(tokenize) -> list[Check]:
    """Short strings whose terms are easy to list by hand."""
    checks = []

    # Lowercase, punctuation becomes a break, and repeats stay in order
    checks.append(expect_equal(
        "lowercase, split at punctuation: 'Fuser E-341: fuser' gives ['fuser', 'e', '341', 'fuser']",
        tokenize("Fuser E-341: fuser"),
        ["fuser", "e", "341", "fuser"],
    ))

    # Several spaces or punctuation marks in a row must not leave empty terms
    got = tokenize("  paper,,  jam ")
    check = expect_equal(
        "leaves no empty terms when spaces repeat: '  paper,,  jam ' gives ['paper', 'jam']",
        got,
        ["paper", "jam"],
    )
    if not check.passed and "" in got:
        check.detail += "\n(split() with no argument drops empty pieces; split(' ') keeps them)"
    checks.append(check)

    # Nothing but punctuation: there are no terms at all
    checks.append(expect_equal(
        "a string with no letters or digits gives no terms: '?! ...' gives []",
        tokenize("?! ..."),
        [],
    ))
    return checks
