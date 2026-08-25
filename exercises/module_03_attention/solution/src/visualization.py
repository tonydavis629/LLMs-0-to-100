"""
Module 3 Exercise: Visualization helpers (provided, not a fill-in)

Plots attention matrices, heatmaps comparing masked vs unmasked attention,
and the effect of positional embeddings on attention patterns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
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


def plot_attention_matrix(
    weights: np.ndarray,
    token_labels: list[str] | None = None,
    title: str = "Attention Weights",
    ax: plt.Axes | None = None,
    cmap: str = "Blues",
) -> plt.Axes:
    """Plot an attention weight matrix as a heatmap.

    Args:
        weights: Attention weights, shape (seq_len, seq_len).
        token_labels: Optional labels for rows/columns.
        title: Plot title.
        ax: Optional matplotlib axes.
        cmap: Colormap name.

    Returns:
        The axes the heatmap was drawn on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    seq_len = weights.shape[0]
    im = ax.imshow(weights, cmap=cmap, vmin=0, vmax=1, aspect="equal")
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    ax.set_title(title, color=_PRIMARY, fontsize=13, fontweight="bold")

    if token_labels is None:
        token_labels = [str(i) for i in range(seq_len)]
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(token_labels, fontsize=9)
    ax.set_yticks(range(seq_len))
    ax.set_yticklabels(token_labels, fontsize=9)

    for i in range(seq_len):
        for j in range(seq_len):
            val = weights[i, j]
            text_color = "white" if val > 0.5 else _FG
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=text_color)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_attention_comparison(
    unmasked: np.ndarray,
    masked: np.ndarray,
    token_labels: list[str] | None = None,
    filepath: str = "attention_comparison.png",
) -> None:
    """Save a side-by-side comparison of unmasked vs causal-masked attention.

    Args:
        unmasked: Unmasked attention weights, shape (seq_len, seq_len).
        masked: Causal-masked attention weights, shape (seq_len, seq_len).
        token_labels: Optional labels for rows/columns.
        filepath: Where to save the figure.

    Returns:
        None.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor(_BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(_BG)
        ax.tick_params(colors=_MUTED)
        for spine in ax.spines.values():
            spine.set_color(_MUTED)

    plot_attention_matrix(unmasked, token_labels, title="Unmasked Attention", ax=ax1)
    plot_attention_matrix(masked, token_labels, title="Causal (Masked) Attention", ax=ax2)

    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    print(f"Saved attention comparison to {_display_path(filepath)}")


def plot_positional_effect(
    no_pos: np.ndarray,
    with_pos: np.ndarray,
    token_labels: list[str] | None = None,
    filepath: str = "positional_effect.png",
) -> None:
    """Save a side-by-side comparison of attention without and with positional embeddings.

    Args:
        no_pos: Attention weights without position, shape (seq_len, seq_len).
        with_pos: Attention weights with position, shape (seq_len, seq_len).
        token_labels: Optional labels for rows/columns.
        filepath: Where to save the figure.

    Returns:
        None.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor(_BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(_BG)
        ax.tick_params(colors=_MUTED)
        for spine in ax.spines.values():
            spine.set_color(_MUTED)

    plot_attention_matrix(no_pos, token_labels, title="Without Positional Embeddings", ax=ax1)
    plot_attention_matrix(with_pos, token_labels, title="With Positional Embeddings", ax=ax2)

    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    print(f"Saved positional effect to {_display_path(filepath)}")


def plot_kv_cache_growth(
    cache_sizes: list[int],
    gen_times_no_cache: list[float],
    gen_times_with_cache: list[float],
    filepath: str = "kv_cache_growth.png",
) -> None:
    """Plot generation time vs sequence length for with/without KV cache.

    Args:
        cache_sizes: Sequence lengths tested.
        gen_times_no_cache: Time per token without cache.
        gen_times_with_cache: Time per token with cache.
        filepath: Where to save the figure.

    Returns:
        None.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)
    ax.plot(cache_sizes, gen_times_no_cache, color="#e74c3c", linewidth=2,
            marker="o", markersize=5, label="Without KV cache")
    ax.plot(cache_sizes, gen_times_with_cache, color="#3fb950", linewidth=2,
            marker="s", markersize=5, label="With KV cache")
    ax.set_xlabel("Sequence length", color=_FG)
    ax.set_ylabel("Time per token (ms)", color=_FG)
    ax.set_title("KV Cache: Generation Cost", color=_PRIMARY, fontsize=13, fontweight="bold")
    ax.tick_params(colors=_MUTED)
    for spine in ax.spines.values():
        spine.set_color(_MUTED)
    leg = ax.legend(loc="upper left", fontsize=10, framealpha=0.0)
    for text in leg.get_texts():
        text.set_color(_FG)
    ax.grid(True, alpha=0.2, color=_MUTED)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    print(f"Saved KV cache plot to {_display_path(filepath)}")
