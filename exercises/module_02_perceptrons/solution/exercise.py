"""
Module 2 Solution: Perceptrons and Neural Networks (PyTorch)

Complete reference implementation of the single-neuron classifier (with
hand-written gradient math), the MLP as an nn.Module, and a from-scratch
SGD optimizer for the extra credit.
"""

from __future__ import annotations

import torch
from torch import nn

# Provided for you - see src/activations.py
from src.activations import sigmoid


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

    For ONE sample, sigmoid + BCE collapse to dL/dz = y_pred - y_true, so
        dL/dw_j = (y_pred - y_true) * x_j     (error times that weight's input)
        dL/db   = (y_pred - y_true)
    The batch loss is a mean, so average over the n samples: for weight j,
    sum error_i * x_ij over every sample i, then divide by n. No loop needed.

    Args:
        X: Input batch, shape (n_samples, 2).
        y_true: True labels, shape (n_samples,).
        y_pred: Predicted probabilities, shape (n_samples,).

    Returns:
        (dw, db): gradient w.r.t. weights (shape (2,)) and bias (scalar tensor).
    """
    error = y_pred - y_true  # How far each prediction is from its label
    n = X.shape[0]  # Number of samples in the batch
    db = error.mean()  # Bias gradient: the average error (provided for you)
    dw = X.T @ error / n  # Weight gradient: each input column weighted by the error, averaged
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
    """A minimal stochastic-gradient-descent optimizer.

    This is Steps 3 and 4 again, but for every parameter of the MLP at once,
    with PyTorch doing Step 3 for you.

    In Step 3 you computed the gradients by hand and returned them:

        dw, db = compute_gradients(X, y, y_pred)

    For the MLP there are four parameter tensors (two weight matrices, two
    bias vectors) and the derivation is longer, so the runner (src/main.py)
    lets autograd do it:

        loss = binary_cross_entropy(y_col, model(X)).mean()
        loss.backward()

    `backward()` computes dL/dp for every tensor `p` created with
    `requires_grad=True` (every nn.Linear weight and bias is). Instead of
    returning the gradients, it stores each one ON the tensor it belongs to,
    in an attribute called `.grad`. For a parameter `p`, `p.grad` is a tensor
    of the same shape as `p` holding dL/dp, exactly what Step 3 called `dw`.
    A tiny example, with L = w1^2 + w2^2:

        w = torch.tensor([1.0, 2.0], requires_grad=True)
        loss = (w * w).sum()
        loss.backward()
        w.grad                       # tensor([2., 4.]) == dL/dw == 2w

    So a training step looks like this:

        optimizer.zero_grad()      # clear last step's .grad on every parameter
        loss = ...                 # forward pass
        loss.backward()            # autograd: fill every p.grad (Step 3)
        optimizer.step()           # YOUR code: p -= lr * p.grad (Step 4)

    `step()` is Step 4 (`update_parameters`) applied to every tensor in
    `self.params`, in place. Two PyTorch details matter:

    - `.grad` accumulates. `backward()` ADDS to whatever is already there, so
      `zero_grad()` (provided) wipes it before each new backward pass.
    - The update must not be recorded on autograd's tape (it is not part of
      the loss), so it goes inside `with torch.no_grad():`.

    This mirrors `torch.optim.SGD`.

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
