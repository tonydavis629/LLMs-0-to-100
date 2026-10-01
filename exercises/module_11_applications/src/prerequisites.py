"""Checking that earlier steps are finished, provided for you.

You do NOT need to edit this file. Most steps call code from earlier steps:
ranking needs cosine similarity, the TF-IDF index needs tokenize, and so on.
If Step 1 were unfinished, running Step 4 would report Step 1's TODO under
Step 4's header. Two helpers avoid that:

- `probe_own_blank(step)` calls the current step's own function on a tiny
  input first, so an unfinished blank reports its own TODO message.
- `needs(steps, purpose)` tries each earlier step on a tiny input and names
  the first one that is still missing.
"""

from __future__ import annotations

import numpy as np

from exercise import (
    cosine_similarity,
    inverse_document_frequency,
    mean_pool,
    rank_documents,
    recall_at_k,
    reciprocal_rank,
    tfidf_vector,
    tokenize,
)
from tests.test_step5_rank import known_good_cosine


def _probe_rank_documents():
    """rank_documents() calls cosine_similarity() from Step 4, so a known-good
    copy stands in for it during this one call."""
    with known_good_cosine(rank_documents):
        rank_documents(np.ones(2), np.ones((2, 2)), 1)


def _probe_metrics():
    """Step 7 has two blanks; try both."""
    recall_at_k(["probe"], ["probe"], 1)
    reciprocal_rank(["probe"], ["probe"])


# Each step's own function on a tiny input, used by probe_own_blank()
_OWN_PROBES = {
    2: lambda: inverse_document_frequency([["probe"]]),
    3: lambda: tfidf_vector(["probe"], {"probe": 1.0}, {"probe": 0}),
    4: lambda: cosine_similarity(np.ones(2), np.ones(2)),
    5: _probe_rank_documents,
    6: lambda: mean_pool(np.ones((2, 2)), np.ones(2)),
    7: _probe_metrics,
}

# One tiny call per earlier step, used by needs(). If it raises
# NotImplementedError, that step is unfinished.
_PROBES = {
    1: ("tokenize", lambda: tokenize("a b")),
    2: ("inverse_document_frequency", lambda: inverse_document_frequency([["a"]])),
    3: ("tfidf_vector", lambda: tfidf_vector(["a"], {"a": 1.0}, {"a": 0})),
    4: ("cosine_similarity", lambda: cosine_similarity(np.ones(2), np.ones(2))),
    5: ("rank_documents", lambda: rank_documents(np.ones(2), np.ones((2, 2)), 1)),
    6: ("mean_pool", lambda: mean_pool(np.ones((2, 2)), np.ones(2))),
}


def probe_own_blank(step: int) -> None:
    """Call this step's own function first, so an unfinished blank reports its own TODO."""
    _OWN_PROBES[step]()


def needs(steps: list[int], purpose: str) -> None:
    """Stop a step with a pointed message if an earlier step is unfinished."""
    for number in steps:
        name, probe = _PROBES[number]
        try:
            probe()
        except NotImplementedError:
            raise NotImplementedError(f"needs Step {number} ({name}) to {purpose}") from None
        except Exception:  # noqa: BLE001
            pass  # it runs; whether it is right is for that step's own tests to say
