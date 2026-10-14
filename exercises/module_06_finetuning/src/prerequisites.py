"""Checking that earlier steps are finished, provided for you.

You do NOT need to edit this file. Several steps run code from earlier
steps (Step 6's training loop uses Steps 1-5). If Step 2 were unfinished,
running Step 3 would report Step 2's TODO under Step 3's header. `require()`
avoids that: it first tries the current step's own blank on a tiny input,
then each earlier step it depends on, and names the first one still missing.
"""

from __future__ import annotations

import torch
from torch import nn

from exercise import (
    build_generation_prompt,
    build_optimizer,
    build_targets,
    count_trainable_params,
    format_example,
    lora_forward_delta,
    masked_cross_entropy,
    merge_lora_weight,
)

# Short names for the steps, used in "needs Step N (...)" messages
STEP_NAMES = {
    "1": "format_example",
    "2": "build_targets",
    "3": "masked_cross_entropy",
    "4": "build_optimizer",
    "5": "lora_forward_delta",
    "6": "sft_train_step",
    "7": "count_trainable_params",
    "8": "build_generation_prompt",
    "9": "merge_lora_weight",
}

# A tiny chat vocabulary for the probes below
_SPECIAL = {"<|user|>": 0, "<|assistant|>": 1, "<|end|>": 2}

# One tiny call per step, used only to see whether that step is finished yet.
# (Step 6 has no probe: its own tests run first instead.)
_PROBES = {
    "1": lambda: format_example("a", "b", _SPECIAL, lambda s: [3]),
    "2": lambda: build_targets([0, 1, 2, 3], 2),
    "3": lambda: masked_cross_entropy(torch.zeros(1, 2, 4), torch.tensor([[-100, 1]])),
    "4": lambda: build_optimizer(nn.Linear(2, 2), 1e-3),
    "5": lambda: lora_forward_delta(torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(2, 1), 1.0, nn.Identity()),
    "7": lambda: count_trainable_params(nn.Linear(1, 1)),
    "8": lambda: build_generation_prompt("a", _SPECIAL, lambda s: [3]),
    "9": lambda: merge_lora_weight(torch.zeros(2, 2), torch.zeros(1, 2), torch.zeros(2, 1), 1.0),
}


def require(step: str | None = None, needs: dict[str, str] | None = None) -> None:
    """Raise NotImplementedError if `step` or any earlier step in `needs` is unfinished.

    `needs` maps each earlier step to what it is needed for, e.g.
    {"1": "build training batches"}. The current step is tried first so that
    its own TODO message is the one reported; a missing earlier step gets a
    "needs Step N (...) to ..." message instead.
    """
    if step is not None:
        _PROBES[step]()  # lets this step's own TODO message through unchanged
    for earlier, purpose in (needs or {}).items():
        try:
            _PROBES[earlier]()
        except NotImplementedError:
            raise NotImplementedError(f"needs Step {earlier} ({STEP_NAMES[earlier]}) to {purpose}") from None
