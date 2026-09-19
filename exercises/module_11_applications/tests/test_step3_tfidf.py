"""Step 3: tfidf_vector()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import random

import numpy as np

from tests.check import Check, bad, ok


def check_tfidf_vector(tfidf_vector) -> list[Check]:
    """Hand-made vocabularies and IDF tables, then a random one."""
    checks = []

    # Three slots: fuser (idf 2.0), jam (idf 0.5), tray (idf 1.0).
    # 'jam' appears twice, 'fuser' once, 'tray' not at all.
    vocab_index = {"fuser": 0, "jam": 1, "tray": 2}
    idf = {"fuser": 2.0, "jam": 0.5, "tray": 1.0}
    got = tfidf_vector(["jam", "fuser", "jam"], idf, vocab_index)
    expected = np.array([2.0, 1.0, 0.0])
    name = "count x IDF: ['jam', 'fuser', 'jam'] with idf fuser=2, jam=0.5 gives [2.0, 1.0, 0.0]"
    if got.shape == expected.shape and np.allclose(got, expected):
        checks.append(ok(name))
    else:
        detail = f"expected {expected.tolist()}, got {np.asarray(got).tolist()}"
        if np.allclose(got, [2.0, 0.5, 0.0]):
            detail += " (did you leave out the count?)"
        elif np.allclose(got, [1.0, 2.0, 0.0]):
            detail += " (did you leave out the IDF weight?)"
        checks.append(bad(name, detail))

    # A term every document uses has IDF 0, so it adds nothing however often it repeats
    got = tfidf_vector(["the"] * 5 + ["jam"], {"the": 0.0, "jam": 1.5}, {"the": 0, "jam": 1})
    checks.append(
        ok("a term with IDF 0 contributes 0, even repeated 5 times: gives [0.0, 1.5]")
        if np.allclose(got, [0.0, 1.5])
        else bad("a term with IDF 0 contributes 0, even repeated 5 times: gives [0.0, 1.5]",
                 f"got {np.asarray(got).tolist()}")
    )

    # Cross-check: a random 200-term text over a 20-term vocabulary
    rng = random.Random(0)
    terms = [f"t{i}" for i in range(20)]
    vocab_index = {term: slot for slot, term in enumerate(terms)}
    idf = {term: rng.uniform(0.0, 4.0) for term in terms}
    tokens = [rng.choice(terms) for _ in range(200)]
    got = tfidf_vector(tokens, idf, vocab_index)
    expected = np.array([tokens.count(term) * idf[term] for term in terms])
    checks.append(
        ok("matches count x IDF in every slot for a random 200-term text")
        if got.shape == expected.shape and np.allclose(got, expected)
        else bad("matches count x IDF in every slot for a random 200-term text",
                 f"first 4 slots: expected {np.round(expected[:4], 3).tolist()}, "
                 f"got {np.round(np.asarray(got)[:4], 3).tolist()}")
    )
    return checks
