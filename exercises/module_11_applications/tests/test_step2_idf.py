"""Step 2: inverse_document_frequency()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import math

from tests.check import Check, bad, expect_equal, ok


def _weight_check(name: str, idf: dict, term: str, n_docs: int, df: int) -> Check:
    """Compare one term's weight with log(n_docs / df), and guess at common slips."""
    expected = math.log(n_docs / df)
    got = idf.get(term)
    if isinstance(got, (int, float)) and math.isclose(got, expected, abs_tol=1e-9):
        return ok(name)
    shown = f"{got:.4f}" if isinstance(got, float) else repr(got)
    detail = f"expected {expected:.4f} for {term!r}, got {shown}"
    if isinstance(got, (int, float)) and df < n_docs:
        if math.isclose(got, math.log2(n_docs / df), abs_tol=1e-9):
            detail += " (use math.log, the natural log, not log2)"
        elif math.isclose(got, math.log(df / n_docs), abs_tol=1e-9):
            detail += " (the fraction is upside down: total_docs / df)"
        elif math.isclose(got, n_docs / df, abs_tol=1e-9):
            detail += " (did you forget the log?)"
    return bad(name, detail)


def check_inverse_document_frequency(inverse_document_frequency) -> list[Check]:
    """A 3-document corpus where every IDF weight can be worked out by hand."""
    checks = []

    # 'the' is in all 3 documents, 'fuser' in 2, 'fault' and 'jam' in 1 each
    docs = [["the", "fuser", "fault"], ["the", "fuser"], ["the", "jam", "jam"]]
    idf = inverse_document_frequency(docs)

    checks.append(expect_equal(
        "returns one weight per distinct term: 'the', 'fuser', 'fault', 'jam'",
        sorted(idf),
        sorted(["the", "fuser", "fault", "jam"]),
    ))
    checks.append(_weight_check(
        "a term in 1 of 3 documents weighs log(3/1) = 1.099 ('jam', even though it repeats)",
        idf, "jam", 3, 1,
    ))
    checks.append(_weight_check(
        "a term in 2 of 3 documents weighs log(3/2) = 0.405 ('fuser')",
        idf, "fuser", 3, 2,
    ))
    checks.append(_weight_check(
        "a term in every document weighs log(3/3) = 0: it cannot tell documents apart ('the')",
        idf, "the", 3, 3,
    ))
    return checks
