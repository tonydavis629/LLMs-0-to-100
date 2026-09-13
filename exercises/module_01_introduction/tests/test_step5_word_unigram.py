"""Step 5: word_unigram()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import random

from tests.check import Check, bad, expect_equal, ok


def check_word_unigram(word_unigram) -> list[Check]:
    """Same idea as char_unigram, but the units are whole words."""
    checks = []

    # Output is words joined by single spaces, so split() recovers them
    out = word_unigram("the cat sat on the mat", 7)
    checks.append(expect_equal("returns exactly as many words as requested, separated by spaces",
                                len(out.split()), 7))

    # Only words from the training text may appear
    sample = word_unigram("alpha beta gamma", 200).split()
    extra = sorted(set(sample) - {"alpha", "beta", "gamma"})
    checks.append(
        ok("only produces words that appear in the training text")
        if not extra
        else bad("only produces words that appear in the training text",
                  f"generated words not in the text: {extra}")
    )

    # 9 "the" and 1 "cat": about 90% of the output should be "the"
    random.seed(0)
    sample = word_unigram(" ".join(["the"] * 9 + ["cat"]), 2000).split()
    frac_the = sample.count("the") / len(sample) if sample else 0.0
    checks.append(
        ok("samples in proportion to frequency (9 the : 1 cat gives ~90% the)")
        if 0.85 <= frac_the <= 0.95
        else bad("samples in proportion to frequency (9 the : 1 cat gives ~90% the)",
                  f"expected about 90% 'the', got {frac_the:.0%}")
    )
    return checks
