"""
Module 2 Exercise: Visualization helpers (provided, not a fill-in)

Plots decision boundaries, loss curves, and the raw datasets for the
perceptron exercise. Prediction functions are vectorized: they take a
whole grid of points, shape (n_points, 2), and return (n_points,)
probabilities.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend so it works without a display
import matplotlib.pyplot as plt

# Deck theme colors (used for the on-slide dataset image)
_BG = "#0a0e1a"
_FG = "#e8eaf0"
_MUTED = "#8892a4"
_RED = "#e74c3c"   # class 0
_GREEN = "#3fb950"  # class 1


def load_csv(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a CSV with columns x1, x2, label.

    Args:
        filepath: Path to the CSV file.

    Returns:
        (X, y): features of shape (n_samples, 2) and integer labels (n_samples,).
    """
    data = np.loadtxt(filepath, delimiter=",", skiprows=1)
    X = data[:, :2]
    y = data[:, 2].astype(int)
    return X, y


def plot_data(X: np.ndarray, y: np.ndarray, ax: plt.Axes | None = None) -> plt.Axes:
    """Scatter plot of 2D data colored by class label.

    Args:
        X: Feature array, shape (n_samples, 2).
        y: Label array, shape (n_samples,).
        ax: Optional matplotlib axes to draw on.

    Returns:
        The axes the data was drawn on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    colors = ["#e74c3c", "#3498db"]  # red for 0, blue for 1
    for label in [0, 1]:
        mask = y == label
        ax.scatter(
            X[mask, 0], X[mask, 1],
            c=colors[label], s=30, alpha=0.7,
            label=f"Class {label}", edgecolors="white", linewidths=0.5,
        )
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(loc="upper left", fontsize=9)
    return ax


def plot_decision_boundary(
    predict_fn,
    X: np.ndarray,
    y: np.ndarray,
    title: str = "Decision Boundary",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot the decision boundary as a filled contour behind the data points.

    Args:
        predict_fn: A function taking a grid of points (n_points, 2) and
            returning (n_points,) probabilities in [0, 1].
        X: Feature array, shape (n_samples, 2).
        y: Label array, shape (n_samples,).
        title: Plot title.
        ax: Optional matplotlib axes to draw on.

    Returns:
        The axes the boundary was drawn on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    margin = 1.0
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200),
    )

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = np.asarray(predict_fn(grid_points)).reshape(xx.shape)

    ax.contourf(xx, yy, Z, levels=50, cmap="RdBu", alpha=0.6, vmin=0, vmax=1)
    ax.contour(xx, yy, Z, levels=[0.5], colors="white", linewidths=2)

    plot_data(X, y, ax=ax)
    ax.set_title(title)
    return ax


def plot_loss_curve(
    losses: list[float],
    title: str = "Training Loss",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot loss over training epochs.

    Args:
        losses: Per-epoch loss values.
        title: Plot title.
        ax: Optional matplotlib axes to draw on.

    Returns:
        The axes the curve was drawn on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    ax.plot(losses, color="#4a9eff", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax


def save_comparison(
    predict_perceptron,
    predict_mlp,
    X: np.ndarray,
    y: np.ndarray,
    filepath: str = "comparison.png",
) -> None:
    """Save a side-by-side comparison of perceptron vs MLP decision boundaries.

    Args:
        predict_perceptron: Vectorized predict_fn for the single neuron.
        predict_mlp: Vectorized predict_fn for the MLP.
        X: Feature array, shape (n_samples, 2).
        y: Label array, shape (n_samples,).
        filepath: Where to save the figure.

    Returns:
        None.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plot_decision_boundary(predict_perceptron, X, y, title="Single Neuron", ax=ax1)
    plot_decision_boundary(predict_mlp, X, y, title="MLP (1 Hidden Layer)", ax=ax2)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison plot to {filepath}")


def plot_datasets(linear_csv: str, nonlinear_csv: str, filepath: str) -> None:
    """Save a dark-themed side-by-side scatter of the two exercise datasets.

    Used as a static image on the slides (no animation).

    Args:
        linear_csv: Path to the linearly separable dataset.
        nonlinear_csv: Path to the non-linearly separable (XOR) dataset.
        filepath: Where to save the figure (PNG).

    Returns:
        None.
    """
    panels = [
        (load_csv(linear_csv), "Linearly Separable"),
        (load_csv(nonlinear_csv), "Non-Linearly Separable (XOR)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    fig.patch.set_facecolor(_BG)
    for ax, ((X, y), title) in zip(axes, panels):
        ax.set_facecolor(_BG)
        for cls, color in [(0, _RED), (1, _GREEN)]:
            mask = y == cls
            ax.scatter(X[mask, 0], X[mask, 1], c=color, s=22, alpha=0.85,
                       label=f"Class {cls}", edgecolors="none")
        ax.set_title(title, color="#4a9eff", fontsize=14, fontweight="bold")
        ax.set_xlabel("x1", color=_FG)
        ax.set_ylabel("x2", color=_FG)
        ax.tick_params(colors=_MUTED)
        for spine in ax.spines.values():
            spine.set_color(_MUTED)
        leg = ax.legend(loc="upper left", fontsize=9, framealpha=0.0)
        for text in leg.get_texts():
            text.set_color(_FG)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    print(f"Saved dataset image to {filepath}")
