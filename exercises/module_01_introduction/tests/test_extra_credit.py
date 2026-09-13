"""Extra credit: cross_entropy()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import math
from collections import Counter

from tests.check import Check, bad, ok


def _count_ngrams(text: str, n: int) -> dict[str, Counter]:
    """A private copy of the n-gram counter, so the extra-credit checks
    still work even if Step 4 is not finished yet."""
    model: dict[str, Counter] = {}
    for i in range(len(text) - n + 1):
        context, next_char = text[i : i + n - 1], text[i + n - 1]
        model.setdefault(context, Counter())[next_char] += 1
    return model


def check_cross_entropy(cross_entropy, text: str) -> list[Check]:
    """Hand-computable cases first, then the real corpus."""
    checks = []

    # Model where a->b and b->a always. Scoring "abab": every prediction is
    # certain, so each costs log2(1) = 0 bits.
    ce = cross_entropy("abab", {"a": Counter({"b": 1}), "b": Counter({"a": 1})})
    checks.append(
        ok("a model that predicts every character perfectly scores 0 bits")
        if math.isclose(ce, 0.0, abs_tol=1e-9)
        else bad("a model that predicts every character perfectly scores 0 bits",
                  f"got {ce}")
    )

    # After 'a', the model says a and b are 50/50. Scoring "aaab" the three
    # predictions each cost log2(2) = 1 bit, so the average is 1.0.
    ce = cross_entropy("aaab", {"a": Counter({"a": 1, "b": 1})})
    checks.append(
        ok("a 50/50 guess costs exactly 1 bit per character")
        if math.isclose(ce, 1.0, abs_tol=1e-9)
        else bad("a 50/50 guess costs exactly 1 bit per character", f"got {ce}")
    )

    # The model has only seen a->b. Scoring "aa" asks for P(a | a), which was
    # never seen, so it should fall back to the 1e-6 smoothing probability.
    ce = cross_entropy("aa", {"a": Counter({"b": 1})})
    expected = -math.log2(1e-6)
    checks.append(
        ok("an unseen character is smoothed to 1e-6 (about 19.93 bits)")
        if math.isclose(ce, expected, rel_tol=1e-6)
        else bad("an unseen character is smoothed to 1e-6 (about 19.93 bits)",
                  f"expected {expected:.4f}, got {ce}")
    )

    # Train on 90% of Alice, score the last 10%: more context should help
    # (at least up to trigrams; beyond that, sparsity starts to hurt).
    text = text.lower()
    split = int(len(text) * 0.9)
    train, held = text[:split], text[split:]
    bits = [cross_entropy(held, _count_ngrams(train, n)) for n in (1, 2, 3)]
    checks.append(
        ok("on held-out Alice, bits/char drops from unigram to bigram to trigram")
        if bits[0] > bits[1] > bits[2]
        else bad("on held-out Alice, bits/char drops from unigram to bigram to trigram",
                  "got " + ", ".join(f"n={n}: {b:.3f}" for n, b in zip((1, 2, 3), bits)))
    )
    return checks
