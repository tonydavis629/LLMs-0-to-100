"""
Module 2 Solution: Visualization helpers (same as student version)
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_csv(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a CSV file with columns x1, x2, label."""
    data = np.loadtxt(filepath, delimiter=",", skiprows=1)
    X = data[:, :2]
    y = data[:, 2].astype(int)
    return X, y


def plot_data(X: np.ndarray, y: np.ndarray, ax: plt.Axes | None = None) -> plt.Axes:
    """Scatter plot of 2D data colored by class label."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    colors = ["#e74c3c", "#3498db"]
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
    """Plot the decision boundary as a filled contour behind the data points."""
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
    Z = np.array([predict_fn(p) for p in grid_points])
    Z = Z.reshape(xx.shape)

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
    """Plot loss over training epochs."""
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
    """Save a side-by-side comparison of perceptron vs MLP decision boundaries."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    plot_decision_boundary(predict_perceptron, X, y, title="Single Neuron", ax=ax1)
    plot_decision_boundary(predict_mlp, X, y, title="MLP (1 Hidden Layer)", ax=ax2)

    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison plot to {filepath}")
