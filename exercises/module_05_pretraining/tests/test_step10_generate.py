"""Step 10: generate()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from tests.check import Check, bad, ok

VOCAB = 5  # a 5-token vocabulary keeps every tensor tiny


def _bigram(weights: torch.Tensor) -> nn.Embedding:
    """A bigram model: row i of the table is the logits for the token after token i."""
    return nn.Embedding.from_pretrained(weights.clone(), freeze=False)


def _seeded(seed: int) -> torch.Generator:
    """A fresh random number generator, so every check is repeatable."""
    return torch.Generator().manual_seed(seed)


def check_generate(generate) -> list[Check]:
    """Softmax, then sample: checked on tiny models whose next token is known."""
    checks = []

    # A table that strongly prefers token i + 1 after token i (4 wraps to 0)
    counting = _bigram(3.0 * torch.roll(torch.eye(VOCAB), shifts=1, dims=1))
    seed_ids = torch.tensor([[0, 1]])

    out = generate(counting, seed_ids, 5, 4, temperature=1.0, generator=_seeded(0))
    checks.append(
        ok("appends max_new_tokens tokens: a 2-token seed plus 5 new gives shape (1, 7)")
        if tuple(out.shape) == (1, 7) and out[0, :2].tolist() == [0, 1]
        else bad("appends max_new_tokens tokens: a 2-token seed plus 5 new gives shape (1, 7)",
                 f"got shape {tuple(out.shape)}, first tokens {out[0, :2].tolist()} "
                 "(next_id should have shape (batch, 1))")
    )

    # Near temperature 0 the distribution is one-hot, so sampling picks the argmax
    out = generate(counting, seed_ids, 5, 4, temperature=0.01, generator=_seeded(0))
    checks.append(
        ok("at temperature 0.01 it matches argmax: the counting table gives 0 1 2 3 4 0 1")
        if out.tolist() == [[0, 1, 2, 3, 4, 0, 1]]
        else bad("at temperature 0.01 it matches argmax: the counting table gives 0 1 2 3 4 0 1",
                 f"got {out.tolist()} (softmax over the last dimension, dim=-1)")
    )

    # Every row says 0.6 for token 0 and 0.1 for each other token.
    # Sample 2,000 next tokens at once (one per row of the batch).
    probs = torch.tensor([0.6, 0.1, 0.1, 0.1, 0.1])
    fixed = _bigram(torch.log(probs).repeat(VOCAB, 1))
    starts = torch.zeros(2000, 1, dtype=torch.long)
    out = generate(fixed, starts, 1, 4, temperature=1.0, generator=_seeded(1))
    share = float((out[:, -1] == 0).float().mean())
    checks.append(
        ok("samples in proportion: a token with probability 0.6 comes up about 60% of the time")
        if math.isclose(share, 0.6, abs_tol=0.05)
        else bad("samples in proportion: a token with probability 0.6 comes up about 60% of the time",
                 f"drawn {100 * share:.1f}% of the time "
                 "(sample with torch.multinomial instead of always taking the most likely token)")
    )

    # The runner's before/after samples rely on the generator for repeatable text
    first = generate(fixed, starts[:4], 10, 4, generator=_seeded(7))
    second = generate(fixed, starts[:4], 10, 4, generator=_seeded(7))
    checks.append(
        ok("the same generator seed gives the same sample twice")
        if torch.equal(first, second)
        else bad("the same generator seed gives the same sample twice",
                 "two runs with generator seed 7 differed (pass generator=generator to torch.multinomial)")
    )
    return checks
