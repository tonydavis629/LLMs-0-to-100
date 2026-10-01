"""Calling earlier steps for a later step's demo, provided for you.

You do NOT need to edit this file. Most steps build on the ones before:
Step 5, for example, needs the weights from Steps 2, 3, and 4. If one of
those were unfinished, Step 5 would report that step's TODO under its own
header. These helpers call the earlier steps and, if one is still missing,
name it instead ("needs Step 3 (raw_attention_scores) to ...").
"""

from __future__ import annotations

import torch


def need(step: str, purpose: str, fn, *args):
    """Call an earlier step's function, naming that step if it is unfinished."""
    try:
        return fn(*args)
    except NotImplementedError:
        raise NotImplementedError(f"needs Step {step} to {purpose}") from None


def earlier_qkv(layer, X: torch.Tensor, purpose: str):
    """Q, K, V from your Step 2."""
    return need("2 (compute_qkv)", purpose, layer.compute_qkv, X)


def earlier_scores(layer, X: torch.Tensor, purpose: str) -> torch.Tensor:
    """Raw attention scores from your Steps 2 and 3."""
    Q, K, _ = earlier_qkv(layer, X, purpose)
    return need("3 (raw_attention_scores)", purpose, layer.raw_attention_scores, Q, K)


def earlier_weights(layer, X: torch.Tensor, purpose: str) -> torch.Tensor:
    """Unmasked attention weights from your Steps 2, 3, and 4."""
    scores = earlier_scores(layer, X, purpose)
    return need("4 (scaled_softmax)", purpose, layer.scaled_softmax, scores)
