"""
Module 2 Exercise runner: Perceptrons and Neural Networks (PyTorch)

Run with:
    uv run python module_02_perceptrons/src/main.py

Trains a single-neuron classifier (with hand-written gradients) and an MLP
(with autograd) on 2D data, visualizing how they learn to separate classes.
Any step that still raises NotImplementedError is skipped, so you can run
after each fill-in.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Make the module root (parent of src/) importable so we can `import exercise`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Also ensure src/ is on the path so we can import sibling helpers
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exercise import (
    forward,
    binary_cross_entropy,
    compute_gradients,
    update_parameters,
    sigmoid,
    relu,
    MLP,
    SGD,
)
from visualization import (
    load_csv,
    plot_decision_boundary,
    plot_loss_curve,
    save_comparison,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Paths to the bundled datasets. Walk up from this file until we find data/
# (works from both src/main.py and solution/src/main.py).
_THIS_DIR = Path(__file__).resolve().parent
_MODULE_DIR = _THIS_DIR.parent
if not (_MODULE_DIR / "data").exists():
    _MODULE_DIR = _MODULE_DIR.parent
DATA_DIR = _MODULE_DIR / "data"
LINEAR_DATA = DATA_DIR / "linear_separable.csv"
NONLINEAR_DATA = DATA_DIR / "non_linear_separable.csv"
OUTPUT_DIR = _MODULE_DIR / "output"


def load_tensors(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
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
# Single neuron: hand-written forward / loss / gradients / update
# ---------------------------------------------------------------------------


def train_perceptron(
    X: torch.Tensor,
    y: torch.Tensor,
    learning_rate: float = 0.5,
    epochs: int = 100,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """Train a single neuron with full-batch gradient descent (no autograd).

    Args:
        X: Feature tensor, shape (n_samples, 2).
        y: Label tensor, shape (n_samples,).
        learning_rate: Step size for each gradient-descent update.
        epochs: Number of full passes through the dataset.

    Returns:
        (weights, bias, losses): trained parameters and the loss history.
    """
    torch.manual_seed(42)
    weights = torch.randn(2) * 0.1
    bias = torch.zeros(())  # 0-dim tensor
    losses: list[float] = []

    for epoch in range(epochs):
        y_pred = forward(X, weights, bias)
        loss = binary_cross_entropy(y, y_pred).mean()
        losses.append(float(loss))

        dw, db = compute_gradients(X, y, y_pred)
        weights, bias = update_parameters(weights, bias, dw, db, learning_rate)

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1:3d}/{epochs}  loss={loss:.4f}")

    return weights, bias, losses


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
# MLP: autograd + an optimizer (torch's, or the student's for extra credit)
# ---------------------------------------------------------------------------


def train_mlp(
    X: torch.Tensor,
    y: torch.Tensor,
    hidden_size: int = 8,
    learning_rate: float = 1.0,
    epochs: int = 500,
    use_custom_optimizer: bool = False,
    verbose: bool = True,
) -> tuple[MLP, list[float]]:
    """Train the MLP with autograd.

    Args:
        X: Feature tensor, shape (n_samples, 2).
        y: Label tensor, shape (n_samples,).
        hidden_size: Number of neurons in the hidden layer.
        learning_rate: Step size for the optimizer.
        epochs: Number of training passes.
        use_custom_optimizer: Whether to use the exercise's SGD class.
        verbose: Whether to print progress every 100 epochs.

    Returns:
        (model, losses): trained model and per-epoch loss history.
    """
    torch.manual_seed(42)
    model = MLP(input_size=2, hidden_size=hidden_size)
    if use_custom_optimizer:
        optimizer = SGD(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    y_col = y.unsqueeze(1)  # shape (N, 1) to match the model output
    losses: list[float] = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)                                  # forward (student code)
        loss = binary_cross_entropy(y_col, output).mean()  # average over the batch
        loss.backward()                                    # autograd: every gradient
        optimizer.step()                                   # nudge parameters downhill

        losses.append(loss.item())
        if verbose and (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch + 1:3d}/{epochs}  loss={loss.item():.4f}")

    return model, losses


def mlp_accuracy(X: torch.Tensor, y: torch.Tensor, model: MLP) -> tuple[int, int]:
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


def mlp_predict(model: MLP):
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


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    X_lin, y_lin = load_tensors(LINEAR_DATA)
    X_nl, y_nl = load_tensors(NONLINEAR_DATA)

    neuron_nl_params = None  # (weights, bias) from Part 2, reused in Part 3
    mlp_model = None         # trained MLP from Part 3

    # ------------------------------------------------------------------
    # Part 1: single neuron on linearly separable data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("PART 1: Single Neuron on Linearly Separable Data")
    print("=" * 60)
    print(f"Loaded {len(y_lin)} samples from linear_separable.csv\n")
    try:
        weights, bias, losses = train_perceptron(X_lin, y_lin, learning_rate=0.5, epochs=100)
        print(f"\nFinal weights: [{weights[0]:.4f}, {weights[1]:.4f}], bias: {float(bias):.4f}")
        correct, total = neuron_accuracy(X_lin, y_lin, weights, bias)
        print(f"Accuracy: {correct}/{total} ({100 * correct / total:.1f}%)")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        plot_decision_boundary(neuron_predict(weights, bias), X_lin.numpy(), y_lin.numpy(),
                               title="Perceptron: Linear Data", ax=ax1)
        plot_loss_curve(losses, title="Training Loss", ax=ax2)
        fig.tight_layout()
        fig.savefig(str(OUTPUT_DIR / "step5_linear_perceptron.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved plot to output/step5_linear_perceptron.png\n")
    except NotImplementedError as e:
        print(f"  [skipped: {e}]\n")

    # ------------------------------------------------------------------
    # Part 2: same single neuron on non-linearly-separable data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("PART 2: Single Neuron on Non-Linearly Separable Data (XOR)")
    print("=" * 60)
    print(f"Loaded {len(y_nl)} samples from non_linear_separable.csv\n")
    try:
        weights_nl, bias_nl, losses_nl = train_perceptron(X_nl, y_nl, learning_rate=0.5, epochs=100)
        neuron_nl_params = (weights_nl, bias_nl)
        print(f"\nFinal weights: [{weights_nl[0]:.4f}, {weights_nl[1]:.4f}], bias: {float(bias_nl):.4f}")
        correct, total = neuron_accuracy(X_nl, y_nl, weights_nl, bias_nl)
        print(f"Accuracy: {correct}/{total} ({100 * correct / total:.1f}%)")
        print("(A single neuron cannot solve this non-linearly-separable problem.)\n")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        plot_decision_boundary(neuron_predict(weights_nl, bias_nl), X_nl.numpy(), y_nl.numpy(),
                               title="Perceptron: XOR Data (Fails)", ax=ax1)
        plot_loss_curve(losses_nl, title="Training Loss (Plateaus)", ax=ax2)
        fig.tight_layout()
        fig.savefig(str(OUTPUT_DIR / "step6_nonlinear_perceptron.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved plot to output/step6_nonlinear_perceptron.png\n")
    except NotImplementedError as e:
        print(f"  [skipped: {e}]\n")

    # ------------------------------------------------------------------
    # Part 3: MLP on non-linearly-separable data (autograd + torch.optim.SGD)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("PART 3: MLP on Non-Linearly Separable Data")
    print("=" * 60)
    try:
        mlp_model, losses_mlp = train_mlp(X_nl, y_nl, hidden_size=8, learning_rate=1.0, epochs=500)
        correct, total = mlp_accuracy(X_nl, y_nl, mlp_model)
        print(f"\nAccuracy: {correct}/{total} ({100 * correct / total:.1f}%)")

        if neuron_nl_params is not None:
            save_comparison(
                neuron_predict(*neuron_nl_params),
                mlp_predict(mlp_model),
                X_nl.numpy(), y_nl.numpy(),
                filepath=str(OUTPUT_DIR / "step7_comparison.png"),
            )
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_loss_curve(losses_mlp, title="MLP Training Loss", ax=ax)
        fig.savefig(str(OUTPUT_DIR / "step7_mlp_loss.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved MLP loss plot to output/step7_mlp_loss.png\n")
    except NotImplementedError as e:
        print(f"  [skipped: {e}]\n")

    # ------------------------------------------------------------------
    # Extra credit: train the MLP with your own SGD optimizer
    # ------------------------------------------------------------------
    print("=" * 60)
    print("EXTRA CREDIT: MLP trained with your own SGD optimizer")
    print("=" * 60)
    try:
        ec_model, _ = train_mlp(
            X_nl, y_nl, hidden_size=8, learning_rate=1.0, epochs=500,
            use_custom_optimizer=True, verbose=False,
        )
        correct, total = mlp_accuracy(X_nl, y_nl, ec_model)
        print(f"Accuracy: {correct}/{total} ({100 * correct / total:.1f}%)")
        print("(Your optimizer matches torch.optim.SGD.)\n")
    except NotImplementedError as e:
        print(f"  [skipped: {e}]\n")

    print("=" * 60)
    print("Done! Check the output/ directory for plots.")
    print("=" * 60)


if __name__ == "__main__":
    main()
