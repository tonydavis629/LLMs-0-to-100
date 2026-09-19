"""Step 3: TinyAttentionLayer.raw_attention_scores()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import torch

from tests.check import Check, bad, ok


def check_raw_attention_scores(TinyAttentionLayer) -> list[Check]:
    """One score per (query, key) pair: the dot product of the two vectors."""
    checks = []
    layer = TinyAttentionLayer(d_model=2, d_k=2)

    # 3 queries and 2 keys, so the answer is a 3 x 2 matrix (and a 2 x 3 one is wrong)
    Q = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    K = torch.tensor([[1.0, 1.0], [2.0, 0.0]])
    scores = layer.raw_attention_scores(Q, K)

    name = "one row per query, one column per key: 3 queries and 2 keys give shape (3, 2)"
    checks.append(
        ok(name)
        if tuple(scores.shape) == (3, 2)
        else bad(name, f"expected (3, 2), got {tuple(scores.shape)} (queries go on the left: Q @ K.T)")
    )

    # Row 0: q=[1,0] with k=[1,1] -> 1 and with k=[2,0] -> 2, and so on
    expected = torch.tensor([[1.0, 2.0], [2.0, 0.0], [2.0, 2.0]])
    name = "entry (i, j) is q_i . k_j: Q=[[1,0],[0,2],[1,1]], K=[[1,1],[2,0]] gives [[1,2],[2,0],[2,2]]"
    if scores.shape == expected.shape and torch.allclose(scores, expected, atol=1e-6):
        checks.append(ok(name))
    else:
        nudge = ""
        if scores.shape == expected.shape and torch.allclose(scores, expected / 2 ** 0.5, atol=1e-4):
            nudge = "\n(these are already scaled by 1/sqrt(d_k); leave that to Step 4)"
        got = [[round(v, 4) for v in row] for row in scores.tolist()]
        checks.append(bad(name, f"expected {expected.tolist()}\ngot      {got}{nudge}"))

    # Random 5-token Q and K, checked against torch.einsum spelling out the dot products
    torch.manual_seed(0)
    Q, K = torch.randn(5, 4), torch.randn(5, 4)
    scores = layer.raw_attention_scores(Q, K)
    reference = torch.einsum("qd,kd->qk", Q, K)
    name = "matches torch.einsum('qd,kd->qk', Q, K) on random 5-token Q and K"
    if scores.shape == reference.shape and torch.allclose(scores, reference, atol=1e-5):
        checks.append(ok(name))
    elif scores.shape != reference.shape:
        checks.append(bad(name, f"expected shape (5, 5), got {tuple(scores.shape)}"))
    else:
        nudge = " (that is K @ Q.T, the transpose)" if torch.allclose(scores, reference.T, atol=1e-5) else ""
        checks.append(bad(name, f"expected entry (0, 1) = {reference[0, 1]:.4f}, got {scores[0, 1]:.4f}{nudge}"))
    return checks
