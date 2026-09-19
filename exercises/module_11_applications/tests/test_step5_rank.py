"""Step 5: rank_documents()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import math
from contextlib import contextmanager

import numpy as np

from tests.check import Check, bad, expect_equal, ok


def _reference_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """A known-good cosine similarity (the Step 4 answer)."""
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denominator == 0.0 else float(np.dot(a, b) / denominator)


@contextmanager
def known_good_cosine(rank_documents):
    """rank_documents() calls your cosine_similarity() from Step 4. While these
    tests run, a known-good copy stands in for it, so they judge only Step 5."""
    namespace = rank_documents.__globals__  # the module rank_documents lives in
    original = namespace.get("cosine_similarity")
    namespace["cosine_similarity"] = _reference_cosine
    try:
        yield
    finally:
        namespace["cosine_similarity"] = original


def _docs_with_scores(scores: list[float]) -> np.ndarray:
    """2D unit vectors whose cosine with the query [1, 0] is exactly each score."""
    return np.array([[s, math.sqrt(1.0 - s * s)] for s in scores])


def check_rank_documents(rank_documents) -> list[Check]:
    """Documents whose scores are set by hand, then a cross-check against numpy."""
    checks = []
    query = np.array([1.0, 0.0])

    with known_good_cosine(rank_documents):
        # Scores 0.1, 0.9, 0.5, 0.7: the best two are document 1, then document 3
        # np.asarray(...).tolist() turns a list, tuple, or numpy array into a plain list
        got = np.asarray(rank_documents(query, _docs_with_scores([0.1, 0.9, 0.5, 0.7]), 2)).tolist()
        check = expect_equal(
            "returns the k best indices, best first: scores [0.1, 0.9, 0.5, 0.7], k=2 give [1, 3]",
            got, [1, 3],
        )
        if not check.passed and got == [0, 2]:
            check.detail += "\n(that is the worst two: did you forget reverse=True?)"
        elif not check.passed and any(isinstance(i, float) and not i.is_integer() for i in got):
            check.detail += "\n(return document indices, not the scores themselves)"
        checks.append(check)

        # The runner scores with k = the corpus size, to find every document's rank
        got = np.asarray(rank_documents(query, _docs_with_scores([0.2, 0.8, 0.5]), 3)).tolist()
        checks.append(expect_equal(
            "k = corpus size ranks every document: scores [0.2, 0.8, 0.5], k=3 give [1, 2, 0]",
            got, [1, 2, 0],
        ))

        # Cross-check: 30 random documents in 8 dimensions, top 5
        rng = np.random.default_rng(0)
        docs, q = rng.normal(size=(30, 8)), rng.normal(size=8)
        scores = np.array([_reference_cosine(q, d) for d in docs])
        expected = [int(i) for i in np.argsort(-scores)[:5]]
        got = np.asarray(rank_documents(q, docs, 5)).tolist()
        checks.append(expect_equal(
            "agrees with numpy's argsort on 30 random documents, k=5",
            got, expected,
        ))
    return checks
