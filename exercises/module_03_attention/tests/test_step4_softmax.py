"""Step 4: TinyAttentionLayer.scaled_softmax()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tests.check import Check, bad, ok


def _row_softmax(rows: list[list[float]], divide_by: float) -> torch.Tensor:
    """Softmax of each row after dividing by a constant, for the nudges below."""
    return torch.softmax(torch.tensor(rows) / divide_by, dim=-1)


def check_scaled_softmax(TinyAttentionLayer) -> list[Check]:
    """Divide by sqrt(d_k), then softmax across each row so the weights sum to 1."""
    checks = []

    # d_k = 4, so the scores are divided by sqrt(4) = 2 before the softmax
    layer = TinyAttentionLayer(d_model=4, d_k=4)
    scores = [[2.0, 0.0], [0.0, 0.0]]
    weights = layer.scaled_softmax(torch.tensor(scores))

    # Row 0 becomes softmax([1, 0]) = [e/(e+1), 1/(e+1)] = [0.731, 0.269]
    expected = torch.tensor([[0.7311, 0.2689], [0.5, 0.5]])
    name = "d_k=4 divides by 2: scores [[2,0],[0,0]] give weights [[0.731,0.269],[0.5,0.5]]"
    if weights.shape == expected.shape and torch.allclose(weights, expected, atol=1e-3):
        checks.append(ok(name))
    else:
        # Name the most likely slip by comparing against each wrong version
        nudge = ""
        if weights.shape == expected.shape:
            if torch.allclose(weights, _row_softmax(scores, 1.0), atol=1e-3):
                nudge = " (did you divide by sqrt(d_k) before the softmax?)"
            elif torch.allclose(weights, _row_softmax(scores, 4.0), atol=1e-3):
                nudge = " (divide by the square root of d_k, not by d_k itself)"
            elif torch.allclose(weights, _row_softmax(scores, 2 ** 0.5), atol=1e-3):
                nudge = " (use self.d_k, the vector size, not the number of keys)"
            elif torch.allclose(weights, torch.softmax(torch.tensor(scores) / 2, dim=0), atol=1e-3):
                nudge = " (softmax runs along dim=-1, across each row)"
        got = [[round(v, 4) for v in row] for row in weights.tolist()]
        checks.append(bad(name, f"expected [[0.7311, 0.2689], [0.5, 0.5]]\ngot      {got}{nudge}"))

    # 3 queries x 5 keys: every query's weights over the 5 keys must sum to 1
    torch.manual_seed(0)
    weights = layer.scaled_softmax(torch.randn(3, 5) * 3)
    row_sums = weights.sum(dim=-1)
    name = "each query's weights over the keys sum to 1 (3 queries x 5 keys)"
    checks.append(
        ok(name)
        if tuple(weights.shape) == (3, 5) and torch.allclose(row_sums, torch.ones(3), atol=1e-5)
        else bad(name, f"row sums are {[round(v, 4) for v in row_sums.tolist()]} "
                       "(softmax along dim=-1, the key dimension)")
    )

    # torch's own attention with V = identity returns exactly its attention weights
    Q, K = torch.randn(4, 4), torch.randn(4, 4)
    weights = layer.scaled_softmax(Q @ K.T)
    reference = F.scaled_dot_product_attention(Q, K, torch.eye(4))
    name = "matches the weights inside torch's F.scaled_dot_product_attention (4 tokens, d_k=4)"
    checks.append(
        ok(name)
        if weights.shape == reference.shape and torch.allclose(weights, reference, atol=1e-5)
        else bad(name, f"expected row 0 = {[round(v, 4) for v in reference[0].tolist()]}\n"
                       f"got      row 0 = {[round(v, 4) for v in weights.flatten()[:4].tolist()]}")
    )
    return checks
