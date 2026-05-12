"""
Module 2 Exercise: Text Classifier with Decision Boundary Visualization

Run with:
    uv run python module_02_perceptrons/src/main.py

Trains a single-neuron classifier and an MLP on 2D data, visualizing
how they learn to separate classes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make the module root (parent of src/) importable so we can `import exercise`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Also ensure src/ is on the path so we can import sibling helpers
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exercise import (
    forward,
    binary_cross_entropy,
    compute_gradients,
    update_parameters,
    mlp_forward,
    mlp_gradients,
    sigmoid,
)
from visualization import (
    load_csv,
    plot_data,
    plot_decision_boundary,
    plot_loss_curve,
    save_comparison,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Paths to the bundled datasets
# Walk up from this file until we find the data/ directory
# (works from both src/main.py and solution/src/main.py)
_THIS_DIR = Path(__file__).resolve().parent
_MODULE_DIR = _THIS_DIR.parent
if not (_MODULE_DIR / "data").exists():
    _MODULE_DIR = _MODULE_DIR.parent
DATA_DIR = _MODULE_DIR / "data"
LINEAR_DATA = DATA_DIR / "linear_separable.csv"
NONLINEAR_DATA = DATA_DIR / "non_linear_separable.csv"
OUTPUT_DIR = _MODULE_DIR / "output"


def train_perceptron(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.5,
    epochs: int = 100,
) -> tuple[np.ndarray, float, list[float]]:
    """Train a single neuron on the dataset.

    Returns:
        (weights, bias, losses): trained parameters and loss history
    """
    # Initialize weights to small random values, bias to zero
    rng = np.random.default_rng(42)
    weights = rng.normal(0, 0.1, size=2)
    bias = 0.0
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0

        # One pass through all samples (full-batch gradient descent)
        total_dw = np.zeros(2)
        total_db = 0.0

        for i in range(len(X)):
            # Forward pass
            y_pred = forward(X[i], weights, bias)

            # Loss
            loss = binary_cross_entropy(y[i], y_pred)
            epoch_loss += loss

            # Gradients
            dw, db = compute_gradients(X[i], y[i], y_pred)
            total_dw += dw
            total_db += db

        # Average gradients over all samples
        total_dw /= len(X)
        total_db /= len(X)

        # Update parameters
        weights, bias = update_parameters(weights, bias, total_dw, total_db, learning_rate)

        avg_loss = epoch_loss / len(X)
        losses.append(avg_loss)

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1:3d}/{epochs}  loss={avg_loss:.4f}")

    return weights, bias, losses


def train_mlp(
    X: np.ndarray,
    y: np.ndarray,
    hidden_size: int = 8,
    learning_rate: float = 1.0,
    epochs: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """Train a two-layer MLP on the dataset.

    Returns:
        (W1, b1, W2, b2, losses): trained parameters and loss history
    """
    rng = np.random.default_rng(42)
    W1 = rng.normal(0, 0.5, size=(hidden_size, 2))
    b1 = np.zeros(hidden_size)
    W2 = rng.normal(0, 0.5, size=(1, hidden_size))
    b2 = np.zeros(1)
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0

        # Accumulate gradients
        dW1_total = np.zeros_like(W1)
        db1_total = np.zeros_like(b1)
        dW2_total = np.zeros_like(W2)
        db2_total = np.zeros_like(b2)

        for i in range(len(X)):
            y_pred = mlp_forward(X[i], W1, b1, W2, b2)

            eps = 1e-15
            y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
            loss = -(y[i] * np.log(y_pred_clipped) + (1 - y[i]) * np.log(1 - y_pred_clipped))
            epoch_loss += loss

            dW1, db1_g, dW2, db2_g = mlp_gradients(X[i], y[i], W1, b1, W2, b2)
            dW1_total += dW1
            db1_total += db1_g
            dW2_total += dW2
            db2_total += db2_g

        # Average and update
        n = len(X)
        W1 -= learning_rate * dW1_total / n
        b1 -= learning_rate * db1_total / n
        W2 -= learning_rate * dW2_total / n
        b2 -= learning_rate * db2_total / n

        avg_loss = epoch_loss / n
        losses.append(avg_loss)

        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch + 1:3d}/{epochs}  loss={avg_loss:.4f}")

    return W1, b1, W2, b2, losses


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Steps 1-5: Train single neuron on linearly separable data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1-5: Single Neuron on Linearly Separable Data")
    print("=" * 60)

    try:
        X_lin, y_lin = load_csv(str(LINEAR_DATA))
        print(f"Loaded {len(y_lin)} samples from linear_separable.csv\n")
    except FileNotFoundError:
        print(f"Data file not found: {LINEAR_DATA}")
        sys.exit(1)

    try:
        weights, bias, losses = train_perceptron(X_lin, y_lin, learning_rate=0.5, epochs=100)
        print(f"\nFinal weights: [{weights[0]:.4f}, {weights[1]:.4f}], bias: {bias:.4f}")

        # Compute accuracy
        correct = sum(
            1 for i in range(len(X_lin))
            if (forward(X_lin[i], weights, bias) >= 0.5) == y_lin[i]
        )
        print(f"Accuracy: {correct}/{len(y_lin)} ({100 * correct / len(y_lin):.1f}%)")

        # Save plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        plot_decision_boundary(
            lambda x: forward(x, weights, bias),
            X_lin, y_lin, title="Perceptron: Linear Data", ax=ax1,
        )
        plot_loss_curve(losses, title="Training Loss", ax=ax2)
        fig.tight_layout()
        fig.savefig(str(OUTPUT_DIR / "step5_linear_perceptron.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot to output/step5_linear_perceptron.png\n")

    except NotImplementedError as e:
        print(f"  [skipped: {e}]\n")

    # ------------------------------------------------------------------
    # Step 6: Same single neuron on non-linearly-separable data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 6: Single Neuron on Non-Linearly Separable Data (XOR)")
    print("=" * 60)

    try:
        X_nl, y_nl = load_csv(str(NONLINEAR_DATA))
        print(f"Loaded {len(y_nl)} samples from non_linear_separable.csv\n")
    except FileNotFoundError:
        print(f"Data file not found: {NONLINEAR_DATA}")
        sys.exit(1)

    try:
        weights_nl, bias_nl, losses_nl = train_perceptron(
            X_nl, y_nl, learning_rate=0.5, epochs=100,
        )
        print(f"\nFinal weights: [{weights_nl[0]:.4f}, {weights_nl[1]:.4f}], bias: {bias_nl:.4f}")

        correct = sum(
            1 for i in range(len(X_nl))
            if (forward(X_nl[i], weights_nl, bias_nl) >= 0.5) == y_nl[i]
        )
        print(f"Accuracy: {correct}/{len(y_nl)} ({100 * correct / len(y_nl):.1f}%)")
        print("(A single neuron cannot solve this non-linearly-separable problem.)\n")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        plot_decision_boundary(
            lambda x: forward(x, weights_nl, bias_nl),
            X_nl, y_nl, title="Perceptron: XOR Data (Fails)", ax=ax1,
        )
        plot_loss_curve(losses_nl, title="Training Loss (Plateaus)", ax=ax2)
        fig.tight_layout()
        fig.savefig(str(OUTPUT_DIR / "step6_nonlinear_perceptron.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot to output/step6_nonlinear_perceptron.png\n")

    except NotImplementedError as e:
        print(f"  [skipped: {e}]\n")

    # ------------------------------------------------------------------
    # Step 7: MLP on non-linearly-separable data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 7: MLP on Non-Linearly Separable Data")
    print("=" * 60)

    try:
        W1, b1, W2, b2, losses_mlp = train_mlp(
            X_nl, y_nl, hidden_size=8, learning_rate=1.0, epochs=500,
        )

        correct = sum(
            1 for i in range(len(X_nl))
            if (mlp_forward(X_nl[i], W1, b1, W2, b2) >= 0.5) == y_nl[i]
        )
        print(f"\nAccuracy: {correct}/{len(y_nl)} ({100 * correct / len(y_nl):.1f}%)")

        # Save comparison plot
        save_comparison(
            lambda x: forward(x, weights_nl, bias_nl),
            lambda x: mlp_forward(x, W1, b1, W2, b2),
            X_nl, y_nl,
            filepath=str(OUTPUT_DIR / "step7_comparison.png"),
        )

        # Save MLP loss curve
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_loss_curve(losses_mlp, title="MLP Training Loss", ax=ax)
        fig.savefig(str(OUTPUT_DIR / "step7_mlp_loss.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved MLP loss plot to output/step7_mlp_loss.png\n")

    except NotImplementedError as e:
        print(f"  [skipped: {e}]\n")

    print("=" * 60)
    print("Done! Check the output/ directory for plots.")
    print("=" * 60)


if __name__ == "__main__":
    main()
