"""Seeding PyTorch so every run matches the lecture slides, provided for you.

You do NOT need to edit this file.
"""

from __future__ import annotations

import torch

from src.model import GPTConfig, TinyGPT


def seed_like_captured_run(seed: int) -> None:
    """Seed PyTorch's random numbers so every run builds the same model.

    After seeding, this draws the same random numbers that an earlier version
    of the runner drew for a quick self-test on a scratch model. The draws
    are thrown away. They only keep the starting weights, and so every number
    and sample shown in the lecture slides, identical to the captured run.
    """
    torch.manual_seed(seed)
    scratch = TinyGPT(GPTConfig(vocab_size=7, block_size=8, n_layer=1, n_head=2, n_embd=16))
    for shape in [(64,), (2, 8), (2, 8)]:
        torch.randint(0, 7, shape)
    torch.randn(2, 8, 7)
    torch.randint(56, (2,))                              # one batch of start indices
    scratch(torch.zeros(2, 8, dtype=torch.long))         # one forward pass with dropout on
    torch.randint(56, (2,))                              # another batch of start indices
    for _ in range(2):
        torch.multinomial(torch.full((1, 7), 1 / 7), 1)  # two sampled tokens
