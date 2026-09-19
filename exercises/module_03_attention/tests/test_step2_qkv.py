"""Step 2: TinyAttentionLayer.compute_qkv()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tests.check import Check, bad, ok


def _hand_layer(TinyAttentionLayer):
    """A layer with d_model=3, d_k=2 and weights simple enough to check by hand."""
    layer = TinyAttentionLayer(d_model=3, d_k=2)
    # W_Q keeps features 1 and 2, W_K keeps features 2 and 3
    layer.W_Q = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    layer.W_K = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    # W_V: first output is feature 1, second is feature 1 + 2 * feature 3
    layer.W_V = torch.tensor([[1.0, 1.0], [0.0, 0.0], [0.0, 2.0]])
    return layer


def check_compute_qkv(TinyAttentionLayer) -> list[Check]:
    """Three projections of the same tokens, one weight matrix each."""
    checks = []

    # Two tokens of width d_model=3
    layer = _hand_layer(TinyAttentionLayer)
    X = torch.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]])
    Q, K, V = layer.compute_qkv(X)

    # d_model=3 and d_k=2 differ, so a transposed product shows up as a wrong shape
    shapes = [tuple(t.shape) for t in (Q, K, V)]
    name = "returns Q, K, V, each of shape (seq_len, d_k): 2 tokens, d_k=2 gives (2, 2)"
    checks.append(
        ok(name)
        if shapes == [(2, 2)] * 3
        else bad(name, f"expected [(2, 2), (2, 2), (2, 2)], got {shapes} (the tokens X go on the left of @)")
    )

    # Token [1, 2, 3]: Q = [1, 2], K = [2, 3], V = [1, 1 + 2*3] = [1, 7]
    expected = {
        "Q": torch.tensor([[1.0, 2.0], [0.0, 1.0]]),
        "K": torch.tensor([[2.0, 3.0], [1.0, 0.0]]),
        "V": torch.tensor([[1.0, 7.0], [0.0, 0.0]]),
    }
    got = {"Q": Q, "K": K, "V": V}
    wrong = [
        key for key in "QKV"
        if got[key].shape != expected[key].shape or not torch.allclose(got[key].float(), expected[key])
    ]
    name = "token [1, 2, 3] projects to Q=[1, 2], K=[2, 3], V=[1, 7] with hand-picked weights"
    if not wrong:
        checks.append(ok(name))
    else:
        detail = "; ".join(f"{key} expected {expected[key][0].tolist()}, got {got[key].flatten()[:2].tolist()}"
                           for key in wrong)
        checks.append(bad(name, detail + "\n(Q uses W_Q, K uses W_K, V uses W_V, returned in that order)"))

    # The layer's own random weights, checked against torch's linear layer function
    torch.manual_seed(0)
    layer = TinyAttentionLayer(d_model=8, d_k=4)
    X = torch.randn(5, 8)
    Q, K, V = layer.compute_qkv(X)
    reference = [F.linear(X, W.T) for W in (layer.W_Q, layer.W_K, layer.W_V)]
    name = "on the layer's random weights (5 tokens, d_model=8, d_k=4) matches torch's F.linear"
    shapes = [tuple(t.shape) for t in (Q, K, V)]
    if shapes != [(5, 4)] * 3:
        checks.append(bad(name, f"expected three tensors of shape (5, 4), got {shapes}"))
    else:
        mismatched = [key for key, a, b in zip("QKV", (Q, K, V), reference) if not torch.allclose(a, b, atol=1e-6)]
        checks.append(
            ok(name)
            if not mismatched
            else bad(name, f"{', '.join(mismatched)} differ from X @ W for the layer's own weight matrices")
        )
    return checks
