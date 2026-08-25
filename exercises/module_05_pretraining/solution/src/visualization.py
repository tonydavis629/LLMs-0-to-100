"""
Module 5: visualization helpers (PROVIDED — not a fill-in).

Plots the training and validation loss curves recorded during pretraining, so
you can SEE optimization progress rather than only reading printed numbers.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to a file, no interactive window needed
import matplotlib.pyplot as plt

_BG = "#0a0e1a"
_FG = "#e8eaf0"
_MUTED = "#8892a4"
_PRIMARY = "#4a9eff"
_SECONDARY = "#f5a623"


def _display_path(filepath: str) -> str:
    """Return a short path for terminal output when possible."""
    path = Path(filepath)
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def plot_loss_curve(
    steps: list[int],
    train_losses: list[float],
    val_losses: list[float],
    filepath: str = "loss_curve.png",
) -> None:
    """Save a line plot of training and validation loss versus step.

    Args:
        steps: The training-step number at each checkpoint.
        train_losses: Estimated training loss at each checkpoint.
        val_losses: Estimated validation loss at each checkpoint.
        filepath: Where to save the figure.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    ax.plot(steps, train_losses, color=_PRIMARY, linewidth=2.2, marker="o",
            markersize=4, label="train loss")
    ax.plot(steps, val_losses, color=_SECONDARY, linewidth=2.2, marker="s",
            markersize=4, label="validation loss")

    ax.set_xlabel("training step", color=_FG, fontsize=12)
    ax.set_ylabel("cross-entropy loss (nats)", color=_FG, fontsize=12)
    ax.set_title("Pretraining loss curve", color=_PRIMARY, fontsize=14, fontweight="bold")
    ax.tick_params(colors=_MUTED)
    for spine in ax.spines.values():
        spine.set_color(_MUTED)
    ax.grid(True, color=_MUTED, alpha=0.15)

    legend = ax.legend(facecolor=_BG, edgecolor=_MUTED, fontsize=11)
    for text in legend.get_texts():
        text.set_color(_FG)

    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    print(f"Saved loss curve to {_display_path(filepath)}")
