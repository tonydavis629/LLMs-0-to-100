"""
Module 4 Exercise: Visualization helpers (provided, not a fill-in)

Plots top token probabilities from the model's next-token distribution.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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


def plot_token_probs(
    tokens: list[str],
    probs: np.ndarray,
    title: str = "Next-Token Probabilities",
    filepath: str = "token_probs.png",
) -> None:
    """Save a horizontal bar chart of top token probabilities.

    Args:
        tokens: Token strings (already decoded) for the top-k entries.
        probs: Probabilities matching the tokens, shape (k,).
        title: Chart title.
        filepath: Where to save the figure.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    # Clean tokens for display (replace newlines/spaces for compactness).
    display = [t.replace("\n", "\\n").replace(" ", "_") for t in tokens]
    y_pos = np.arange(len(display))

    colors = [_PRIMARY if i == 0 else _SECONDARY for i in range(len(display))]
    bars = ax.barh(y_pos[::-1], probs[::-1] * 100, color=colors[::-1], height=0.7)

    ax.set_yticks(y_pos[::-1])
    ax.set_yticklabels(display[::-1], color=_FG, fontsize=10)
    ax.set_xlabel("Probability (%)", color=_FG, fontsize=11)
    ax.set_title(title, color=_PRIMARY, fontsize=13, fontweight="bold")
    ax.tick_params(colors=_MUTED)
    for spine in ax.spines.values():
        spine.set_color(_MUTED)
    ax.set_xlim(0, max(probs) * 100 * 1.15)

    # Label each bar with its percentage.
    for bar, p in zip(bars, probs[::-1]):
        width = bar.get_width()
        ax.text(
            width + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{p * 100:.1f}%",
            va="center", ha="left", color=_FG, fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    print(f"Saved token probability plot to {_display_path(filepath)}")
