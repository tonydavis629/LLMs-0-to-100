"""
Module 2 Exercise: Perceptrons and Neural Networks (PyTorch)

Build a single-neuron classifier and a multi-layer perceptron, then train
them on 2D data. We use PyTorch tensors throughout. We still write our own
activation functions, and for the single neuron we do the gradient math by
hand, so nothing important is hidden. For the MLP we let PyTorch's autograd
compute the gradients and (in the extra credit) write our own optimizer to
apply them.

Fill in the lines marked with `raise NotImplementedError(...)`.
Each one needs only ONE line of code (or two at most).
"""

from __future__ import annotations

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Activation functions (these operate on PyTorch tensors)
# ---------------------------------------------------------------------------


def sigmoid(z: torch.Tensor) -> torch.Tensor:
    """The sigmoid activation: 1 / (1 + exp(-z)).

    Maps any real number into (0, 1), so we read the output as a probability.
    We use it on the OUTPUT neuron for binary classification.

    Args:
        z: A tensor of pre-activation values (any shape).

    Returns:
        A tensor the same shape as z, with every entry in (0, 1).
    """
    # This one is provided for you.
    return 1.0 / (1.0 + torch.exp(-z))


# ---------------------------------------------------------------------------
# Step 1: Forward pass (single neuron)
# ---------------------------------------------------------------------------


def forward(x: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Compute the output of a single neuron.

    Args:
        x: Shape (2,) input tensor.
        weights: Shape (2,) weight tensor.
        bias: Scalar bias tensor.

    Returns:
        A probability tensor in (0, 1).
    """
    # TODO: Compute the neuron's output in one line
    # HINT: use the dot product x @ weights, add bias, then call sigmoid
    raise NotImplementedError("TODO: implement the forward pass")


# ---------------------------------------------------------------------------
# Step 2: Loss function (binary cross-entropy)
# ---------------------------------------------------------------------------


def binary_cross_entropy(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Compute the binary cross-entropy loss: -[y log(p) + (1 - y) log(1 - p)].

    This is Shannon's entropy used as a loss: it measures how surprised we are
    by the prediction given the true label. It works elementwise, so it accepts
    a single sample or a whole batch.

    Args:
        y_true: True label(s), 0 or 1 (scalar or tensor).
        y_pred: Predicted probability/probabilities in (0, 1).

    Returns:
        The per-sample loss (same shape as the inputs); lower is better.
    """
    # Clip predictions away from 0 and 1 to avoid log(0) = -infinity
    eps = 1e-7
    y_pred = torch.clamp(y_pred, eps, 1 - eps)

    # TODO: Compute and return the BCE loss using the formula in the docstring
    # HINT: use torch.log on y_pred and on (1 - y_pred), mirroring the formula above
    raise NotImplementedError("TODO: implement binary cross-entropy")


# ---------------------------------------------------------------------------
# Step 3: Gradient computation (single neuron, by hand)
# ---------------------------------------------------------------------------


def compute_gradients(
    x: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute gradients of the BCE loss w.r.t. the neuron's weights and bias.

    For sigmoid + BCE the chain rule simplifies beautifully to:
        dL/dw = (y_pred - y_true) * x
        dL/db = (y_pred - y_true)

    Args:
        x: Input vector, shape (2,).
        y_true: True label (0 or 1).
        y_pred: Predicted probability.

    Returns:
        (dw, db): gradient w.r.t. weights (shape (2,)) and bias (0-dim tensor).
    """
    # TODO: Compute the gradients using the formulas in the docstring
    # HINT: the error is (y_pred - y_true); the weight gradient is error * x and the bias gradient is the error
    raise NotImplementedError("TODO: implement gradient computation")


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

    Gradient descent moves each parameter a small step in the negative
    gradient direction (downhill on the loss surface).

    Args:
        weights: Current weight vector, shape (2,).
        bias: Current bias (0-dim tensor).
        dw: Gradient w.r.t. weights, shape (2,).
        db: Gradient w.r.t. bias (0-dim tensor).
        learning_rate: Step size, e.g. 0.5.

    Returns:
        (new_weights, new_bias): the updated parameters.
    """
    # TODO: Apply the gradient-descent update rule
    # HINT: subtract learning_rate * dw from weights, and learning_rate * db from bias
    raise NotImplementedError("TODO: implement parameter update")


# ---------------------------------------------------------------------------
# Step 6: ReLU activation (the hidden-layer nonlinearity)
# ---------------------------------------------------------------------------


def relu(z: torch.Tensor) -> torch.Tensor:
    """The ReLU (Rectified Linear Unit) activation: max(0, z).

    Keeps positive values unchanged and clamps negatives to zero. It is cheap
    and avoids the vanishing-gradient problem, so it dominates the HIDDEN
    layers of modern networks.

    Args:
        z: A tensor of pre-activation values (any shape).

    Returns:
        A tensor the same shape as z, with negatives replaced by 0.
    """
    # TODO: Return the elementwise maximum of 0 and z in one line
    # HINT: torch.clamp(z, min=0.0) clamps every negative value to zero
    raise NotImplementedError("TODO: implement the ReLU activation")


# ---------------------------------------------------------------------------
# Step 7: Multi-layer perceptron (ReLU hidden layer, sigmoid output)
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """A two-layer perceptron: a ReLU hidden layer, then a sigmoid output.

    This is the simplest network that can solve non-linearly-separable
    problems like XOR. `nn.Linear(in, out)` stores a weight matrix and bias
    and computes `x @ W.T + b`; PyTorch tracks them as learnable parameters.

    Args:
        input_size: Number of input features (2 for our 2D data).
        hidden_size: Number of neurons in the hidden layer.
    """

    def __init__(self, input_size: int = 2, hidden_size: int = 8) -> None:
        super().__init__()
        # Two linear layers; their weights/biases are learned during training.
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass through both layers.

        Args:
            x: Input batch, shape (n_samples, 2).

        Returns:
            Output probabilities, shape (n_samples, 1), each in (0, 1).
        """
        # TODO: Compute the two-layer forward pass and return the output probabilities
        # HINT: apply relu to self.hidden(x), then sigmoid to self.output(...) of that result
        raise NotImplementedError("TODO: implement the MLP forward pass")


# ---------------------------------------------------------------------------
# Extra credit: write your own optimizer
# ---------------------------------------------------------------------------


class SGD:
    """A minimal stochastic-gradient-descent optimizer.

    When you call `loss.backward()`, PyTorch's autograd fills in each
    parameter's `.grad`. An optimizer is the piece that then nudges every
    parameter downhill. This mirrors `torch.optim.SGD`.

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
        # This one is provided for you.
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
        # TODO: update each parameter in place using its gradient
        # HINT: inside `with torch.no_grad():`, loop over self.params and do `p -= self.lr * p.grad`
        raise NotImplementedError("Extra credit: implement the optimizer step")
