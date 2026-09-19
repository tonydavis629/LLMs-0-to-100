"""Step 10: merge_lora_weight()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tests.check import Check, bad, ok


def check_merge_lora_weight(merge_lora_weight) -> list[Check]:
    """Fold the adapter into the weight: W + scale * (B @ A), with the same output as base + adapter."""
    checks = []

    # Hand example: B @ A = [[1, 2], [0, 0]], scaled by 0.5, added to the identity
    W = torch.eye(2)
    A = torch.tensor([[1.0, 2.0]])     # shape (r=1, in=2)
    B = torch.tensor([[1.0], [0.0]])   # shape (out=2, r=1)
    W_before = W.clone()
    merged = merge_lora_weight(W, A, B, 0.5)
    expected = torch.tensor([[1.5, 1.0], [0.0, 1.0]])
    checks.append(
        ok("W = I, A = [[1, 2]], B = [[1], [0]], scale = 0.5 gives [[1.5, 1], [0, 1]]")
        if isinstance(merged, torch.Tensor) and merged.shape == expected.shape and torch.allclose(merged, expected)
        else bad("W = I, A = [[1, 2]], B = [[1], [0]], scale = 0.5 gives [[1.5, 1], [0, 1]]",
                 f"got {merged.tolist() if isinstance(merged, torch.Tensor) else merged!r}"
                 + (" (that is (B @ A).t(): multiply B @ A, not A.t() @ B.t())"
                    if isinstance(merged, torch.Tensor) and torch.allclose(merged, torch.tensor([[1.5, 0.0], [1.0, 1.0]]))
                    else ""))
    )
    checks.append(
        ok("returns a new tensor and leaves base_W itself unchanged")
        if torch.equal(W, W_before)
        else bad("returns a new tensor and leaves base_W itself unchanged",
                 f"base_W changed to {W.tolist()} (use + to build a new tensor, not +=)")
    )

    # The point of merging: one plain layer gives the same output as base layer + adapter
    torch.manual_seed(0)
    W, bias = torch.randn(6, 8), torch.randn(6)
    A, B = torch.randn(4, 8), torch.randn(6, 4)
    x = torch.randn(3, 8)
    with_adapter = F.linear(x, W, bias) + 8.0 * (x @ A.t() @ B.t())
    merged = merge_lora_weight(W, A, B, 8.0)
    same = merged.shape == W.shape and torch.allclose(F.linear(x, merged, bias), with_adapter, atol=1e-3)
    checks.append(
        ok("one merged layer gives the same output as base layer + adapter (random 8 -> 6 layer, r = 4)")
        if same
        else bad("one merged layer gives the same output as base layer + adapter (random 8 -> 6 layer, r = 4)",
                 f"merged weight shape {tuple(merged.shape)}, expected {tuple(W.shape)}"
                 if merged.shape != W.shape
                 else f"largest output difference "
                      f"{float((F.linear(x, merged, bias) - with_adapter).abs().max()):.4f}")
    )
    return checks


def check_merge_on_model(results: dict) -> list[Check]:
    """The runner merged every adapter in the real model; its logits must not move."""
    diff = results["merge_diff"]
    return [
        ok("real model: merged and adapter logits agree (max difference under 1e-4)")
        if diff < 1e-4
        else bad("real model: merged and adapter logits agree (max difference under 1e-4)",
                 f"max difference {diff:.2e}")
    ]
