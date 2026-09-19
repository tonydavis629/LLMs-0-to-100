"""Step 4: compute_loss()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import math

import torch

from tests.check import Check, bad, ok


def _reference_loss(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Cross-entropy written out by hand: -log(probability of the true token), averaged."""
    log_probs = torch.log_softmax(logits, dim=-1)
    picked = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return float(-picked.mean())


def _value(loss) -> float:
    """The loss as a plain number, whether it came back as a tensor or a float."""
    return float(loss.detach()) if isinstance(loss, torch.Tensor) else float(loss)


def check_compute_loss(compute_loss) -> list[Check]:
    """Average surprise at the true next token, over every position in the batch."""
    checks = []

    # Equal logits over 3 tokens: every token has probability 1/3, so the loss is ln 3
    uniform = torch.zeros(1, 1, 3)
    loss = compute_loss(uniform, torch.tensor([[1]]))
    checks.append(
        ok("equal logits over 3 tokens cost ln 3 = 1.0986 nats")
        if math.isclose(_value(loss), math.log(3), abs_tol=1e-4)
        else bad("equal logits over 3 tokens cost ln 3 = 1.0986 nats", f"got {_value(loss):.4f}")
    )

    # Two positions: probabilities 1/3 and 1/2 on the true token, so the
    # surprises are ln 3 and ln 2, and the average is (1.0986 + 0.6931) / 2
    logits = torch.tensor([[[0.0, 0.0, 0.0], [math.log(2), 0.0, 0.0]]])
    loss = compute_loss(logits, torch.tensor([[2, 0]]))
    expected = (math.log(3) + math.log(2)) / 2
    checks.append(
        ok("averages over positions: surprises ln 3 and ln 2 give 0.8959")
        if math.isclose(_value(loss), expected, abs_tol=1e-4)
        else bad("averages over positions: surprises ln 3 and ln 2 give 0.8959",
                 f"got {_value(loss):.4f} (average over every position, do not sum)")
    )

    # Random logits where seq_len == vocab_size (4), so a missing reshape
    # still runs but scores the wrong axis
    gen = torch.Generator().manual_seed(0)
    logits = torch.randn(2, 4, 4, generator=gen).requires_grad_()
    targets = torch.randint(0, 4, (2, 4), generator=gen)
    loss = compute_loss(logits, targets)
    expected = _reference_loss(logits.detach(), targets)
    summed = math.isclose(_value(loss), expected * 8, abs_tol=1e-3)  # 2 x 4 = 8 positions
    checks.append(
        ok("matches -log_softmax at the true token, averaged, on a random (2, 4, 4) batch")
        if math.isclose(_value(loss), expected, abs_tol=1e-4)
        else bad("matches -log_softmax at the true token, averaged, on a random (2, 4, 4) batch",
                 f"expected {expected:.4f}, got {_value(loss):.4f} "
                 + ("(that is the sum over the 8 positions: average instead)" if summed
                    else "(did you flatten logits to (batch * time, vocab) before F.cross_entropy?)"))
    )

    # train_step() calls loss.backward(), so this has to stay a tensor
    is_scalar = isinstance(loss, torch.Tensor) and loss.dim() == 0 and loss.requires_grad
    checks.append(
        ok("returns a 0-dim tensor that autograd can backpropagate through")
        if is_scalar
        else bad("returns a 0-dim tensor that autograd can backpropagate through",
                 f"got {type(loss).__name__}"
                 + (f" with shape {tuple(loss.shape)}" if isinstance(loss, torch.Tensor) else "")
                 + " (return the loss tensor itself, not .item())")
    )
    return checks
