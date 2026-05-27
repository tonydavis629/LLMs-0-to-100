"""
Module 2 Solution: Perceptrons and Neural Networks (PyTorch)

Complete reference implementation of the single-neuron classifier (with
hand-written gradient math), the MLP as an nn.Module, and a from-scratch
SGD optimizer for the extra credit.
"""

from __future__ import annotations

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Activation functions (these operate on PyTorch tensors)
# ---------------------------------------------------------------------------


def sigmoid(z: torch.Tensor) -> torch.Tensor:
    """The sigmoid activation: 1 / (1 + exp(-z)).

    Args:
        z: A tensor of pre-activation values (any shape).

    Returns:
        A tensor the same shape as z, with every entry in (0, 1).
    """
    return 1.0 / (1.0 + torch.exp(-z))


# ---------------------------------------------------------------------------
# Step 1: Forward pass (single neuron)
# ---------------------------------------------------------------------------


def forward(X: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Compute the output of a single neuron for a whole batch.

    Args:
        X: Input tensor, shape (n_samples, 2).
        weights: Weight vector, shape (2,).
        bias: Scalar bias tensor.

    Returns:
        A probability tensor of shape (n_samples,) with values in (0, 1).
    """
    return sigmoid(X @ weights + bias)


# ---------------------------------------------------------------------------
# Step 2: Loss function (binary cross-entropy)
# ---------------------------------------------------------------------------


def binary_cross_entropy(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Compute the binary cross-entropy loss: -[y log(p) + (1 - y) log(1 - p)].

    Args:
        y_true: True label(s), 0 or 1 (scalar or tensor).
        y_pred: Predicted probability/probabilities in (0, 1).

    Returns:
        The per-sample loss (same shape as the inputs); lower is better.
    """
    eps = 1e-7
    y_pred = torch.clamp(y_pred, eps, 1 - eps)
    return -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))


# ---------------------------------------------------------------------------
# Step 3: Gradient computation (single neuron, by hand)
# ---------------------------------------------------------------------------


def compute_gradients(
    X: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute gradients of the BCE loss w.r.t. the neuron's weights and bias.

    For sigmoid + BCE the chain rule simplifies to:
        error = y_pred - y_true
        dw    = X.T @ error / n
        db    = error.mean()

    Args:
        X: Input batch, shape (n_samples, 2).
        y_true: True labels, shape (n_samples,).
        y_pred: Predicted probabilities, shape (n_samples,).

    Returns:
        (dw, db): gradient w.r.t. weights (shape (2,)) and bias (scalar tensor).
    """
    error = y_pred - y_true
    n = X.shape[0]
    dw = X.T @ error / n
    db = error.mean()
    return (dw, db)


# ---------------------------------------------------------------------------
# Step 4: Parameter update (one gradient-descent step)
# ---------------------------------------------------------------------------


def update_parameters(
    weights: torch.Tensor,
    bias: torch.Tensor,
    dw: torch.Tensor,
    db: torch.Tensor,
    learning_rate: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update the weights and bias with one gradient-descent step.

    Args:
        weights: Current weight vector, shape (2,).
        bias: Current bias (0-dim tensor).
        dw: Gradient w.r.t. weights, shape (2,).
        db: Gradient w.r.t. bias (0-dim tensor).
        learning_rate: Step size, e.g. 0.5.

    Returns:
        (new_weights, new_bias): the updated parameters.
    """
    return (weights - learning_rate * dw, bias - learning_rate * db)


# ---------------------------------------------------------------------------
# Step 6: ReLU activation (the hidden-layer nonlinearity)
# ---------------------------------------------------------------------------


def relu(z: torch.Tensor) -> torch.Tensor:
    """The ReLU (Rectified Linear Unit) activation: max(0, z).

    Args:
        z: A tensor of pre-activation values (any shape).

    Returns:
        A tensor the same shape as z, with negatives replaced by 0.
    """
    return torch.clamp(z, min=0.0)


# ---------------------------------------------------------------------------
# Step 7: Multi-layer perceptron (ReLU hidden layer, sigmoid output)
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """A two-layer perceptron: a ReLU hidden layer, then a sigmoid output.

    Args:
        input_size: Number of input features (2 for our 2D data).
        hidden_size: Number of neurons in the hidden layer.
    """

    def __init__(self, input_size: int = 2, hidden_size: int = 8) -> None:
        super().__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass through both layers.

        Args:
            x: Input batch, shape (n_samples, 2).

        Returns:
            Output probabilities, shape (n_samples, 1), each in (0, 1).
        """
        h = relu(self.hidden(x))
        return sigmoid(self.output(h))


# ---------------------------------------------------------------------------
# Extra credit: write your own optimizer
# ---------------------------------------------------------------------------


class SGD:
    """A minimal stochastic-gradient-descent optimizer (mirrors torch.optim.SGD).

    Args:
        params: Iterable of parameter tensors to update (requires_grad=True).
        lr: Learning rate (step size).
    """

    def __init__(self, params, lr: float = 0.1) -> None:
        self.params = list(params)
        self.lr = lr

    def zero_grad(self) -> None:
        """Reset every parameter's accumulated gradient to zero.

        Args:
            None.

        Returns:
            None.
        """
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self) -> None:
        """Apply one gradient-descent update to every parameter, in place.

        Args:
            None.

        Returns:
            None. Each parameter tensor is modified in place.
        """
        with torch.no_grad():
            for p in self.params:
                p -= self.lr * p.grad
