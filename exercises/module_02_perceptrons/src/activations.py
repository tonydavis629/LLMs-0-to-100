"""Activation functions provided for you.

You do NOT need to edit this file. `relu()` is yours to write in
exercise.py; `sigmoid()` lives here because you have already seen it.
"""

from __future__ import annotations

import torch


def sigmoid(z: torch.Tensor) -> torch.Tensor:
    """The sigmoid activation: 1 / (1 + exp(-z)).

    Maps any real number into (0, 1), so we read the output as a probability.
    We use it on the OUTPUT neuron for binary classification.

    Args:
        z: A tensor of pre-activation values (any shape).

    Returns:
        A tensor the same shape as z, with every entry in (0, 1).
    """
    return 1.0 / (1.0 + torch.exp(-z))
