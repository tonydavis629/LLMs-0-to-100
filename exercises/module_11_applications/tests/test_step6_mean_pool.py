"""Step 6: mean_pool()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import numpy as np

from tests.check import Check, bad, ok


def _close(got, expected: list[float]) -> bool:
    """True if got has the expected shape and values (a wrong shape is just False)."""
    return np.shape(got) == (len(expected),) and bool(np.allclose(got, expected))


def check_mean_pool(mean_pool, encoder) -> list[Check]:
    """Tiny hand-made token vectors, then the real encoder with and without padding."""
    checks = []

    # Two real tokens: the mean of [1, 2] and [3, 4] is [2, 3]
    got = mean_pool(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([1.0, 1.0]))
    checks.append(
        ok("averages the real tokens: [1, 2] and [3, 4] with mask [1, 1] give [2.0, 3.0]")
        if _close(got, [2.0, 3.0])
        else bad("averages the real tokens: [1, 2] and [3, 4] with mask [1, 1] give [2.0, 3.0]",
                 f"got {np.asarray(got).tolist()}")
    )

    # Same two tokens plus one padding position: the answer must not change
    got = mean_pool(np.array([[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]), np.array([1.0, 1.0, 0.0]))
    if _close(got, [2.0, 3.0]):
        checks.append(ok("ignores padding: adding a row [100, 100] with mask 0 still gives [2.0, 3.0]"))
    else:
        detail = f"expected [2.0, 3.0], got {np.round(np.asarray(got), 4).tolist()}"
        if _close(got, [4 / 3, 2.0]):
            detail += "\n(you divided by all 3 positions; count only the real tokens with mask.sum())"
        elif _close(got, [104 / 3, 106 / 3]):
            detail += "\n(the padding row was averaged in; multiply by the mask first)"
        checks.append(bad("ignores padding: adding a row [100, 100] with mask 0 still gives [2.0, 3.0]",
                          detail))

    # One vector per text, averaged over tokens (axis 0), not over the 384 dimensions
    got = mean_pool(np.ones((10, 384)), np.ones(10))
    checks.append(
        ok("returns one vector of shape (384,) for 10 tokens of shape (10, 384)")
        if np.shape(got) == (384,)
        else bad("returns one vector of shape (384,) for 10 tokens of shape (10, 384)",
                 f"got shape {np.shape(got)} (average over axis 0, the token axis)")
    )

    # The real encoder: a short query encoded alone has no padding. Encoded next to
    # a long article, it gets padded to the article's length. Pooling must give the
    # same vector both times.
    query = "my pages come out crumpled"
    long_text = "Error E-341 means the fuser did not reach operating temperature. " * 3
    (alone_vectors, alone_mask), = encoder.encode([query])
    (padded_vectors, padded_mask), _ = encoder.encode([query, long_text])
    alone = mean_pool(alone_vectors, alone_mask)
    padded = mean_pool(padded_vectors, padded_mask)
    if np.shape(alone) == np.shape(padded):
        gap = float(np.abs(np.asarray(alone) - np.asarray(padded)).max())
    else:
        gap = float("inf")  # different shapes: one of them was pooled over the wrong axis
    n_real, n_total = int(padded_mask.sum()), len(padded_mask)
    checks.append(
        ok(f"real encoder: a query pooled alone and padded from {n_real} to {n_total} tokens "
           "gives the same vector")
        if gap < 1e-4
        else bad(f"real encoder: a query pooled alone and padded from {n_real} to {n_total} tokens "
                 "gives the same vector",
                 f"the two vectors differ by up to {gap:.3f} (padding is leaking into the average)")
    )
    return checks
