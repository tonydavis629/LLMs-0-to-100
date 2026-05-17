"""
Module 2 Solution: Perceptrons and Neural Networks

Complete reference implementation of the single-neuron classifier
and multi-layer perceptron (ReLU hidden layer, sigmoid output).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------


def sigmoid(z: np.ndarray) -> np.ndarray:
    """The sigmoid activation function: 1 / (1 + exp(-z))."""
    return 1.0 / (1.0 + np.exp(-z))


def relu(z: np.ndarray) -> np.ndarray:
    """The ReLU activation function: max(0, z)."""
    return np.maximum(0.0, z)


# ---------------------------------------------------------------------------
# Step 1: Forward pass (single neuron)
# ---------------------------------------------------------------------------


def forward(x: np.ndarray, weights: np.ndarray, bias: float) -> float:
    """Compute the output of a single neuron: sigmoid(dot(x, weights) + bias)."""
    return sigmoid(np.dot(x, weights) + bias)


# ---------------------------------------------------------------------------
# Step 2: Loss function (binary cross-entropy)
# ---------------------------------------------------------------------------


def binary_cross_entropy(y_true: float, y_pred: float) -> float:
    """Compute binary cross-entropy loss for one sample."""
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# ---------------------------------------------------------------------------
# Step 3: Gradient computation
# ---------------------------------------------------------------------------


def compute_gradients(
    x: np.ndarray, y_true: float, y_pred: float
) -> tuple[np.ndarray, float]:
    """Compute gradients of BCE loss w.r.t. weights and bias."""
    error = y_pred - y_true
    return (error * x, error)


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
    """Update weights and bias using gradient descent."""
    return (weights - learning_rate * dw, bias - learning_rate * db)


# ---------------------------------------------------------------------------
# Step 5: MLP forward pass (ReLU hidden layer, sigmoid output)
# ---------------------------------------------------------------------------


def mlp_forward(
    x: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
) -> float:
    """Forward pass through a two-layer MLP: ReLU hidden, sigmoid output."""
    hidden = relu(W1 @ x + b1)
    output = sigmoid(W2 @ hidden + b2)
    return output[0]


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
    """Compute gradients for the two-layer MLP using backpropagation."""
    # Forward pass (save intermediates)
    z1 = W1 @ x + b1
    hidden = relu(z1)
    z2 = W2 @ hidden + b2
    output = sigmoid(z2)

    # Output layer gradients
    error_out = output - y_true                 # shape (1,)
    dW2 = error_out[:, None] * hidden[None, :]   # shape (1, hidden_size)
    db2 = error_out                             # shape (1,)

    # Hidden layer gradients (backpropagate through ReLU)
    relu_grad = (z1 > 0).astype(float)          # ReLU derivative: 1 if z1 > 0 else 0
    error_hidden = (W2.T @ error_out) * relu_grad  # shape (hidden_size,)
    dW1 = error_hidden[:, None] * x[None, :]    # shape (hidden_size, 2)
    db1 = error_hidden                          # shape (hidden_size,)

    return dW1, db1, dW2, db2
