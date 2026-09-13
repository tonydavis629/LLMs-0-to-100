"""Step 6: build_word_ngram_model()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

from collections import Counter

from tests.check import Check, bad, expect_equal, ok


def check_build_word_ngram_model(build_word_ngram_model, text: str) -> list[Check]:
    """Word count tables: contexts must be TUPLES of words."""
    checks = []
    sentence = "the cat the dog the cat"

    # Bigrams: the->cat (x2), cat->the, the->dog, dog->the
    checks.append(expect_equal(
        "bigram contexts are 1-word tuples ('the cat the dog the cat')",
        build_word_ngram_model(sentence, 2),
        {("the",): Counter({"cat": 2, "dog": 1}),
         ("cat",): Counter({"the": 1}),
         ("dog",): Counter({"the": 1})},
    ))

    # Trigrams: (the,cat)->the, (cat,the)->dog, (the,dog)->the, (dog,the)->cat
    checks.append(expect_equal(
        "trigram contexts are 2-word tuples",
        build_word_ngram_model(sentence, 3),
        {("the", "cat"): Counter({"the": 1}),
         ("cat", "the"): Counter({"dog": 1}),
         ("the", "dog"): Counter({"the": 1}),
         ("dog", "the"): Counter({"cat": 1})},
    ))

    # Real-corpus sanity check
    model = build_word_ngram_model(text, 3)
    top = model.get(("the", "white"), Counter()).most_common(1)
    after = top[0][0] if top else None
    checks.append(
        ok("in Alice, the most common word after 'the white' is 'rabbit'")
        if after == "rabbit"
        else bad("in Alice, the most common word after 'the white' is 'rabbit'",
                  f"got {after!r}")
    )
    return checks
