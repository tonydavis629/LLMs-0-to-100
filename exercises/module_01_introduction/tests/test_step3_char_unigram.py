"""Step 3: char_unigram()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import random

from tests.check import Check, bad, expect_equal, ok


def check_char_unigram(char_unigram) -> list[Check]:
    """Frequency-weighted sampling: common characters come out more often."""
    checks = []

    checks.append(expect_equal("returns exactly as many characters as requested",
                                len(char_unigram("hello world", 50)), 50))

    # Characters that never appear in the text must never be generated
    sample = char_unigram("abc", 300)
    extra = sorted(set(sample) - set("abc"))
    checks.append(
        ok("only produces characters that appear in the training text")
        if not extra
        else bad("only produces characters that appear in the training text",
                  f"trained on 'abc' but generated: {extra}")
    )

    # 9 a's and 1 b: about 90% of the output should be 'a'
    random.seed(0)
    sample = char_unigram("aaaaaaaaab", 2000)
    frac_a = sample.count("a") / len(sample) if sample else 0.0
    checks.append(
        ok("samples in proportion to frequency (9 a's : 1 b gives ~90% a)")
        if 0.85 <= frac_a <= 0.95
        else bad("samples in proportion to frequency (9 a's : 1 b gives ~90% a)",
                  f"expected about 90% 'a', got {frac_a:.0%} "
                  f"(did you pass weights=weights to random.choices?)")
    )
    return checks
