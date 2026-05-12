"""
Module 2 Exercise: Perceptrons and Neural Networks

Build a single-neuron classifier and a multi-layer perceptron from scratch
using only numpy. Train them on 2D data and visualize decision boundaries.

Fill in the lines marked with `raise NotImplementedError(...)`.
Each one needs only ONE line of code (or two at most).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Step 1: Forward pass (single neuron)
# ---------------------------------------------------------------------------


def sigmoid(z: np.ndarray) -> np.ndarray:
    """The sigmoid activation function: 1 / (1 + exp(-z)).

    Maps any real number to the range (0, 1), interpreting the output
    as a probability.
    """
    # This one is provided for you
    return 1.0 / (1.0 + np.exp(-z))


def forward(x: np.ndarray, weights: np.ndarray, bias: float) -> float:
    """Compute the output of a single neuron.

    A neuron computes: sigmoid(dot(x, weights) + bias)

    Args:
        x: Input vector, shape (2,)
        weights: Weight vector, shape (2,)
        bias: Scalar bias term

    Returns:
        The neuron's output (a probability between 0 and 1)
    """
    # TODO: Compute the neuron's output in one line
    raise NotImplementedError("TODO: implement the forward pass")


# ---------------------------------------------------------------------------
# Step 2: Loss function (binary cross-entropy)
# ---------------------------------------------------------------------------


def binary_cross_entropy(y_true: float, y_pred: float) -> float:
    """Compute binary cross-entropy loss for one sample.

    BCE = -[y * log(p) + (1 - y) * log(1 - p)]

    This is Shannon's entropy applied as a loss function:
    it measures how surprised we are by the prediction given the true label.

    Args:
        y_true: True label (0 or 1)
        y_pred: Predicted probability (between 0 and 1)

    Returns:
        The loss (a non-negative number; lower is better)
    """
    # Clip prediction to avoid log(0) which gives -infinity
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # TODO: Compute and return the BCE loss using the formula in the docstring
    raise NotImplementedError("TODO: implement binary cross-entropy")


# ---------------------------------------------------------------------------
# Step 3: Gradient computation
# ---------------------------------------------------------------------------


def compute_gradients(
    x: np.ndarray, y_true: float, y_pred: float
) -> tuple[np.ndarray, float]:
    """Compute gradients of BCE loss w.r.t. weights and bias.

    For sigmoid + BCE, the math simplifies to:
        dL/dw = (y_pred - y_true) * x
        dL/db = (y_pred - y_true)

    This elegant simplification is why sigmoid + BCE is such a common pairing.

    Args:
        x: Input vector, shape (2,)
        y_true: True label (0 or 1)
        y_pred: Predicted probability

    Returns:
        (dw, db): gradient w.r.t. weights (shape (2,)) and bias (scalar)
    """
    # TODO: Compute the gradients using the formulas in the docstring
    raise NotImplementedError("TODO: implement gradient computation")


# ---------------------------------------------------------------------------
# Step 4: Parameter update (SGD step)
# ---------------------------------------------------------------------------


def update_parameters(
    weights: np.ndarray,
    bias: float,
    dw: np.ndarray,
    db: float,
    learning_rate: float,
) -> tuple[np.ndarray, float]:
    """Update weights and bias using gradient descent.

    Gradient descent subtracts a fraction (learning_rate) of the gradient
    from each parameter, moving downhill on the loss surface.

    Args:
        weights: Current weight vector, shape (2,)
        bias: Current bias scalar
        dw: Gradient w.r.t. weights, shape (2,)
        db: Gradient w.r.t. bias, scalar
        learning_rate: Step size (e.g. 0.1)

    Returns:
        (new_weights, new_bias)
    """
    # TODO: Apply the gradient descent update rule
    raise NotImplementedError("TODO: implement parameter update")


# ---------------------------------------------------------------------------
# Step 7: MLP forward pass
# ---------------------------------------------------------------------------


def mlp_forward(
    x: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
) -> float:
    """Forward pass through a two-layer MLP.

    Layer 1: hidden = sigmoid(W1 @ x + b1)
    Layer 2: output = sigmoid(W2 @ hidden + b2)

    This is the simplest network that can solve non-linearly-separable problems
    like XOR.

    Args:
        x: Input vector, shape (2,)
        W1: First layer weights, shape (hidden_size, 2)
        b1: First layer biases, shape (hidden_size,)
        W2: Second layer weights, shape (1, hidden_size)
        b2: Second layer bias, shape (1,)

    Returns:
        Output probability (scalar between 0 and 1)
    """
    # TODO: Compute the two-layer forward pass
    raise NotImplementedError("TODO: implement MLP forward pass")


# ---------------------------------------------------------------------------
# Extra credit: MLP backpropagation
# ---------------------------------------------------------------------------


def mlp_gradients(
    x: np.ndarray,
    y_true: float,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute gradients for the two-layer MLP using backpropagation.

    This is the chain rule applied through two layers. The math:
        hidden = sigmoid(W1 @ x + b1)
        output = sigmoid(W2 @ hidden + b2)
        error_out = output - y_true
        dW2 = error_out * hidden^T
        db2 = error_out
        error_hidden = (W2^T @ error_out) * hidden * (1 - hidden)
        dW1 = error_hidden @ x^T
        db1 = error_hidden

    Returns:
        (dW1, db1, dW2, db2)
    """
    raise NotImplementedError("Extra credit: implement MLP backpropagation")
