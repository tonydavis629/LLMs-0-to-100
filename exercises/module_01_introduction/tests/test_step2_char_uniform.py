"""Step 2: char_uniform()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import random
from collections import Counter

from tests.check import Check, bad, expect_equal, ok


def check_char_uniform(char_uniform) -> list[Check]:
    """Uniform sampling: right length, right alphabet, every letter equally likely."""
    checks = []
    alphabet = set("abcdefghijklmnopqrstuvwxyz ")

    # The length argument must be respected, whatever it is
    lengths = [len(char_uniform(500)), len(char_uniform(37))]
    checks.append(expect_equal("returns exactly as many characters as requested",
                                lengths, [500, 37]))

    # Only the 27 allowed characters may appear
    sample = char_uniform(2000)
    extra = sorted(set(sample) - alphabet)
    checks.append(
        ok("uses only the letters a-z and space")
        if not extra
        else bad("uses only the letters a-z and space",
                  f"found unexpected characters: {extra}")
    )

    # With 27 characters and 27,000 draws, each should appear ~1000 times.
    # Seeding makes the check give the same answer every run.
    random.seed(0)
    counts = Counter(char_uniform(27000))
    lo, hi = min(counts.values()), max(counts.values())
    checks.append(
        ok("every character is about equally common (z as often as e)")
        if len(counts) == 27 and 800 <= lo and hi <= 1200
        else bad("every character is about equally common (z as often as e)",
                  f"over 27,000 draws each character should appear ~1000 times; "
                  f"saw counts from {lo} to {hi} across {len(counts)} characters")
    )
    return checks
