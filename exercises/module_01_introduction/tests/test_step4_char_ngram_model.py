"""Step 4: build_char_ngram_model()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

from collections import Counter

from tests.check import Check, bad, expect_equal, ok


def check_build_char_ngram_model(build_char_ngram_model, text: str) -> list[Check]:
    """Count tables built from tiny strings with known answers, then one from Alice."""
    checks = []

    # "abab" as bigrams: a->b, b->a, a->b
    checks.append(expect_equal(
        "bigram counts for 'abab' are a->b twice and b->a once",
        build_char_ngram_model("abab", 2),
        {"a": Counter({"b": 2}), "b": Counter({"a": 1})},
    ))

    # "abcabd" as trigrams: ab->c, bc->a, ca->b, ab->d
    checks.append(expect_equal(
        "trigram contexts are 2 characters long ('abcabd' gives ab->c, ab->d, ...)",
        build_char_ngram_model("abcabd", 3),
        {"ab": Counter({"c": 1, "d": 1}), "bc": Counter({"a": 1}), "ca": Counter({"b": 1})},
    ))

    # A real-corpus sanity check: in English, q is almost always followed by u
    model = build_char_ngram_model(text, 2)
    top = model.get("q", Counter()).most_common(1)
    after_q = top[0][0] if top else None
    checks.append(
        ok("in Alice, the most common character after 'q' is 'u'")
        if after_q == "u"
        else bad("in Alice, the most common character after 'q' is 'u'",
                  f"got {after_q!r}")
    )
    return checks
