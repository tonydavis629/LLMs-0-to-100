"""Loading the datasets and scoring trained models, provided for you.

You do NOT need to edit this file. `load_dataset()` turns a bundled CSV
into tensors. The `*_accuracy()` functions count how many samples a trained
model classifies correctly, and the `*_predict()` functions wrap a model so
the plotting code can color every point of a grid by its predicted class.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import torch

from src.activations import sigmoid
from src.visualization import load_csv


@cache  # read each CSV once, then reuse the same tensors in every step
def load_dataset(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a CSV and return (X, y) as float32 tensors.

    Args:
        path: Path to a bundled CSV dataset.

    Returns:
        (X, y): feature tensor of shape (n_samples, 2) and label tensor of shape (n_samples,).
    """
    X_np, y_np = load_csv(str(path))
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)
    return X, y


# ---------------------------------------------------------------------------
# Single neuron (Step 5)
# ---------------------------------------------------------------------------


def neuron_accuracy(X: torch.Tensor, y: torch.Tensor, weights, bias) -> tuple[int, int]:
    """Count correctly classified samples for the single neuron.

    Args:
        X: Feature tensor, shape (n_samples, 2).
        y: Label tensor, shape (n_samples,).
        weights: Learned weight vector, shape (2,).
        bias: Learned scalar bias.

    Returns:
        (correct, total): number correct and total number of samples.
    """
    with torch.no_grad():
        preds = (sigmoid(X @ weights + bias) >= 0.5).float()
    correct = int((preds == y).sum())
    return correct, X.shape[0]


def neuron_predict(weights, bias):
    """Return a vectorized predict_fn(grid) -> probabilities for plotting.

    Args:
        weights: Learned weight vector, shape (2,).
        bias: Learned scalar bias.

    Returns:
        A function that maps a NumPy grid of points to predicted probabilities.
    """
    def predict(grid):
        g = torch.tensor(grid, dtype=torch.float32)
        with torch.no_grad():
            return sigmoid(g @ weights + bias).numpy()
    return predict


# ---------------------------------------------------------------------------
# MLP (Step 7 and extra credit)
# ---------------------------------------------------------------------------


def mlp_accuracy(X: torch.Tensor, y: torch.Tensor, model) -> tuple[int, int]:
    """Count correctly classified samples for the MLP.

    Args:
        X: Feature tensor, shape (n_samples, 2).
        y: Label tensor, shape (n_samples,).
        model: Trained MLP model.

    Returns:
        (correct, total): number correct and total number of samples.
    """
    with torch.no_grad():
        preds = (model(X).squeeze(1) >= 0.5).float()
    correct = int((preds == y).sum())
    return correct, X.shape[0]


def mlp_predict(model):
    """Return a vectorized predict_fn(grid) -> probabilities for plotting.

    Args:
        model: Trained MLP model.

    Returns:
        A function that maps a NumPy grid of points to predicted probabilities.
    """
    def predict(grid):
        g = torch.tensor(grid, dtype=torch.float32)
        with torch.no_grad():
            return model(g).squeeze(1).numpy()
    return predict
