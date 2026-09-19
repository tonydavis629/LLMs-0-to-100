"""Tiny stand-ins for GPT-2 pieces, used by the tests and the runner.

You do NOT need to edit this file. A test for one step should only depend
on that step's code, so the tests swap the other parts of the model for
these small, predictable replacements:

- `tiny_gpt2()` builds a 3-block GPT2Model whose embedding and blocks are
  replaced by simple reference layers (for Step 4).
- `CountingModel`, `FixedModel` and `NumberTokenizer` replace GPT-2 and its
  tokenizer in the decoding tests (Steps 5 and 6). The "tokens" are just
  integers written as text, so "1 2 3" encodes to the IDs [1, 2, 3].
"""

from __future__ import annotations

import torch
from torch import nn

# ---------------------------------------------------------------------------
# Step 4: a tiny model with known parts
# ---------------------------------------------------------------------------


class ReferenceEmbedding(nn.Module):
    """A correct token + position embedding, so Step 4 does not need Step 1."""

    def __init__(self, vocab_size: int, d_model: int, max_pos: int) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_pos, d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Positions 0, 1, ..., t-1 for every sequence in the batch
        position = torch.arange(token_ids.shape[1]).unsqueeze(0)
        return self.token_embed(token_ids) + self.pos_embed(position)


def first_feature(value: float, d_model: int) -> torch.Tensor:
    """A vector [value, 0, 0, ...] of length d_model."""
    vector = torch.zeros(d_model)
    vector[0] = value
    return vector


class AddBlock(nn.Module):
    """A stand-in transformer block: bumps the first feature and logs that it ran.

    It changes only one feature (not all of them equally) because a layer
    norm would erase a shift applied to every feature.
    """

    def __init__(self, index: int, log: list) -> None:
        super().__init__()
        self.index = index  # which block this is (0, 1, 2, ...)
        self.log = log      # shared list: each block appends its index when called

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.log.append(self.index)
        # Block 0 adds 1 to the first feature, block 1 adds 2, block 2 adds 3
        return x + first_feature(self.index + 1, x.shape[-1])


def tiny_gpt2(GPT2Model, n_layers: int = 3, vocab_size: int = 7, d_model: int = 4):
    """Build a small GPT2Model with reference embedding and logging blocks.

    Returns (model, log). The model's own final layer norm and LM head are
    kept, with random weights; `log` fills with block indices as they run.
    """
    torch.manual_seed(0)
    model = GPT2Model(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
                      n_heads=2, d_ff=8, max_pos=8, dropout=0.0)
    log: list = []
    model.embed = ReferenceEmbedding(vocab_size, d_model, max_pos=8)
    model.blocks = nn.ModuleList([AddBlock(i, log) for i in range(n_layers)])
    # Random (not the default 1 and 0) layer-norm parameters, so skipping
    # ln_f changes the answer
    with torch.no_grad():
        model.ln_f.weight.uniform_(0.5, 1.5)
        model.ln_f.bias.uniform_(-0.5, 0.5)
    return model.eval(), log


# ---------------------------------------------------------------------------
# Steps 5 and 6: fake language models and a fake tokenizer
# ---------------------------------------------------------------------------


class NumberTokenizer:
    """Encodes "3 4 5" as token IDs [[3, 4, 5]] and decodes them back."""

    def encode(self, text: str, return_tensors: str = "pt") -> torch.Tensor:
        return torch.tensor([[int(word) for word in text.split()]])

    def decode(self, token_ids) -> str:
        return " ".join(str(int(t)) for t in token_ids)


class CountingModel(nn.Module):
    """Always gives the highest logit to (last token + 1), wrapping at 10."""

    vocab_size = 10

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # One row of logits per position: 5.0 for the next integer, 0 elsewhere
        next_ids = (token_ids + 1) % self.vocab_size
        return 5.0 * nn.functional.one_hot(next_ids, self.vocab_size).float()


class FixedModel(nn.Module):
    """Returns the same logits at every position, whatever the input."""

    def __init__(self, logits: list[float]) -> None:
        super().__init__()
        self.logits = torch.tensor(logits)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        b, t = token_ids.shape
        return self.logits.expand(b, t, len(self.logits)).clone()
