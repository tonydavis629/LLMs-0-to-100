"""Step 5: lora_forward_delta()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import torch
from torch import nn

from tests.check import Check, bad, ok


def check_lora_forward_delta(lora_forward_delta) -> list[Check]:
    """The low-rank update: dropout(x), through A (down to rank r), through B (back up), times the scale."""
    checks = []

    # Hand example with rank r = 1: x @ A.t() = [[3]], then @ B.t() = [[6, 0, 3]], then * 0.5
    x = torch.tensor([[1.0, 2.0]])
    A = torch.tensor([[1.0, 1.0]])          # shape (r=1, in=2)
    B = torch.tensor([[2.0], [0.0], [1.0]])  # shape (out=3, r=1)
    delta = lora_forward_delta(x, A, B, 0.5, nn.Identity())
    expected = torch.tensor([[3.0, 0.0, 1.5]])
    checks.append(
        ok("x=[1,2], A=[[1,1]], B=[[2],[0],[1]], scale=0.5 gives [3, 0, 1.5]")
        if isinstance(delta, torch.Tensor) and delta.shape == expected.shape and torch.allclose(delta, expected)
        else bad("x=[1,2], A=[[1,1]], B=[[2],[0],[1]], scale=0.5 gives [3, 0, 1.5]",
                 f"got {delta.tolist() if isinstance(delta, torch.Tensor) else delta!r}")
    )

    # Cross-check: the same update written as one (out x in) matrix, scale * (B @ A)
    torch.manual_seed(0)
    x = torch.randn(2, 5, 8)   # (batch, time, in_features)
    A = torch.randn(4, 8)      # r = 4
    B = torch.randn(6, 4)      # out_features = 6
    delta = lora_forward_delta(x, A, B, 8.0, nn.Identity())
    expected = x @ (8.0 * (B @ A)).t()
    checks.append(
        ok("equals x @ (scale * B @ A).t() on a random (2, 5, 8) batch, output shape (2, 5, 6)")
        if delta.shape == expected.shape and torch.allclose(delta, expected, atol=1e-4)
        else bad("equals x @ (scale * B @ A).t() on a random (2, 5, 8) batch, output shape (2, 5, 6)",
                 f"expected shape {tuple(expected.shape)}, got {tuple(delta.shape)}"
                 if delta.shape != expected.shape
                 else f"largest difference {float((delta - expected).abs().max()):.4f} "
                      "(did you multiply by scale?)")
    )

    # LoRA starts with B = 0, so a fresh adapter changes nothing
    delta = lora_forward_delta(x, A, torch.zeros(6, 4), 8.0, nn.Identity())
    checks.append(
        ok("B = 0 (how LoRA starts) gives an all-zero update, so the adapter begins as a no-op")
        if torch.count_nonzero(delta) == 0
        else bad("B = 0 (how LoRA starts) gives an all-zero update, so the adapter begins as a no-op",
                 f"largest entry {float(delta.abs().max()):.4f}")
    )

    # Dropout must be applied to x: with p = 1.0 every input is dropped, so the update is zero
    drop_all = nn.Dropout(p=1.0).train()
    delta = lora_forward_delta(x, A, B, 8.0, drop_all)
    checks.append(
        ok("passes x through dropout first (nn.Dropout(p=1.0) drops every input, so the update is 0)")
        if torch.count_nonzero(delta) == 0
        else bad("passes x through dropout first (nn.Dropout(p=1.0) drops every input, so the update is 0)",
                 "got a nonzero update (call dropout(x) before multiplying by A.t())")
    )
    return checks
