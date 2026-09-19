"""Step 2: FeedForward.forward()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tests.check import Check, bad, ok


def check_feed_forward(FeedForward) -> list[Check]:
    """fc1, tanh GELU, fc2 and dropout, checked on hand-set and random layers."""
    checks = []

    # Shape: widen to d_ff inside, but come back out at d_model
    torch.manual_seed(0)
    ffn = FeedForward(d_model=4, d_ff=8).eval()
    x = torch.randn(1, 5, 4) * 3
    out = ffn(x)
    checks.append(
        ok("keeps the shape: (1, 5, 4) in gives (1, 5, 4) out, even with d_ff=8 inside")
        if tuple(out.shape) == (1, 5, 4)
        else bad("keeps the shape: (1, 5, 4) in gives (1, 5, 4) out, even with d_ff=8 inside",
                 f"expected (1, 5, 4), got {tuple(out.shape)} (end with fc2, which maps d_ff back to d_model)")
    )
    if tuple(out.shape) != (1, 5, 4):
        return checks

    # Hand example: fc1 copies x=[2, -2] and adds their sum, GELU([2, -2, 0]) is
    # about [1.9546, -0.0454, 0], and fc2 adds the first two and copies the third, plus 0.5
    small = FeedForward(d_model=2, d_ff=3).eval()
    with torch.no_grad():
        small.fc1.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
        small.fc1.bias.zero_()
        small.fc2.weight.copy_(torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
        small.fc2.bias.fill_(0.5)
    got = small(torch.tensor([[[2.0, -2.0]]])).detach().flatten()
    expected = torch.tensor([2.4092, 0.5])
    checks.append(
        ok("hand-set layers: x=[2, -2] gives fc2(GELU(fc1(x))) = [2.4092, 0.5000]")
        if got.shape == expected.shape and torch.allclose(got, expected, atol=1e-3)
        else bad("hand-set layers: x=[2, -2] gives fc2(GELU(fc1(x))) = [2.4092, 0.5000]",
                 f"expected {expected.tolist()}\ngot      {[round(v, 4) for v in got.tolist()]}\n"
                 "(apply fc1, then GELU, then fc2)")
    )

    # GPT-2 used the tanh approximation of GELU; the exact curve differs by up to 0.0005
    with torch.no_grad():
        want = ffn.fc2(F.gelu(ffn.fc1(x), approximate="tanh"))
        exact = ffn.fc2(F.gelu(ffn.fc1(x)))
    name = 'uses the tanh GELU that GPT-2 was trained with (approximate="tanh")'
    if torch.allclose(out.detach(), want, atol=1e-6):
        checks.append(ok(name))
    elif torch.allclose(out.detach(), exact, atol=1e-6):
        checks.append(bad(name, 'got the exact GELU instead (pass approximate="tanh" to F.gelu)'))
    else:
        diff = float((out.detach() - want).abs().max())
        checks.append(bad(name, f"largest difference from the reference is {diff:.4f}"))

    # Dropout is active only in training mode, so train and eval outputs must differ
    torch.manual_seed(0)
    drop = FeedForward(d_model=4, d_ff=8, dropout=0.5)
    with torch.no_grad():
        train_out = drop.train()(x)
        eval_out = drop.eval()(x)
    checks.append(
        ok("applies dropout: with p=0.5, training and eval outputs differ")
        if not torch.allclose(train_out, eval_out)
        else bad("applies dropout: with p=0.5, training and eval outputs differ",
                 "the two outputs are identical (did you apply self.dropout?)")
    )
    return checks
