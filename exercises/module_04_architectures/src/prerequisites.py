"""Checking that earlier steps are finished, provided for you.

You do NOT need to edit this file. Running the real GPT-2 calls the code
from several steps at once. If Step 2 were unfinished, running Step 4 would
report Step 2's TODO under Step 4's header. `require()` avoids that: it
first tries the current step's own blank on a tiny input, then each earlier
step it depends on, and names the first one that is still missing.
"""

from __future__ import annotations

import torch

from exercise import (
    EmbeddingLayer,
    FeedForward,
    GPT2Model,
    TransformerBlock,
    greedy_decode,
    sample_with_temperature_topk,
)
from tests.fakes import CountingModel, FixedModel, NumberTokenizer, tiny_gpt2

# Short names for the steps, used in "needs Step N (...)" messages
STEP_NAMES = {
    "1": "EmbeddingLayer.forward",
    "2": "FeedForward.forward",
    "3": "TransformerBlock.forward",
    "4": "GPT2Model.forward",
    "5": "greedy_decode",
    "6": "sample_with_temperature_topk",
}


def _probe_block():
    """Run a tiny block with its FFN swapped out, so only Step 3's own line runs."""
    block = TransformerBlock(d_model=4, n_heads=2, d_ff=8)
    block.ffn = torch.nn.Identity()
    return block(torch.zeros(1, 2, 4))


# One tiny call per step. Each raises NotImplementedError while that step's
# blank is unfinished (and only then), and runs in a fraction of a second.
_PROBES = {
    "1": lambda: EmbeddingLayer(vocab_size=4, d_model=2, max_pos=4)(torch.zeros(1, 2, dtype=torch.long)),
    "2": lambda: FeedForward(d_model=2, d_ff=4)(torch.zeros(1, 2, 2)),
    "3": _probe_block,
    "4": lambda: tiny_gpt2(GPT2Model)[0](torch.zeros(1, 2, dtype=torch.long)),
    "5": lambda: greedy_decode(CountingModel(), NumberTokenizer(), "1", max_new=1),
    "6": lambda: sample_with_temperature_topk(FixedModel([0.0, 1.0]), NumberTokenizer(), "0",
                                              max_new=1, top_k=2),
}


def require(step: str, needs: str, purpose: str) -> None:
    """Raise NotImplementedError if `step` or any step in `needs` is unfinished.

    The current step is tried first so that its own TODO message is the one
    reported. A missing earlier step gets a "needs Step N" message instead.
    """
    with torch.no_grad():
        _PROBES[step]()  # lets this step's own TODO message through unchanged
        for earlier in needs:
            try:
                _PROBES[earlier]()
            except NotImplementedError:
                raise NotImplementedError(
                    f"needs Step {earlier} ({STEP_NAMES[earlier]}) to {purpose}"
                ) from None
