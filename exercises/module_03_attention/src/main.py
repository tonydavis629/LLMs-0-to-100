"""
Module 3 Exercise runner: Attention Mechanisms

Run with:
    uv run python module_03_attention/src/main.py

Computes scaled dot-product attention step by step on a tiny token sequence,
then adds a causal mask and positional embeddings to see their effects.
Any step that still raises NotImplementedError is skipped, so you can run
after each fill-in.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Make the module root (parent of src/) importable so we can `import exercise`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Also ensure src/ is on the path so we can import sibling helpers
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exercise import (
    make_token_vectors,
    compute_qkv,
    raw_attention_scores,
    scaled_softmax,
    attention_output,
    causal_mask,
    masked_attention,
    add_positional_embeddings,
    kv_cache_step,
)
from visualization import (
    plot_attention_comparison,
    plot_positional_effect,
    plot_kv_cache_growth,
)

_THIS_DIR = Path(__file__).resolve().parent
_MODULE_DIR = _THIS_DIR.parent
if not (_MODULE_DIR / "data").exists():
    _MODULE_DIR = _MODULE_DIR.parent
OUTPUT_DIR = _MODULE_DIR / "output"


def _try_run(label: str, fn, *args, **kwargs):
    """Run a function, printing its result or skipping on NotImplementedError.

    Args:
        label: Section header to print.
        fn: The function to call.
        *args: Positional arguments to pass to fn.
        **kwargs: Keyword arguments to pass to fn.

    Returns:
        The function's return value, or None if skipped.
    """
    print(f"=== {label} ===")
    try:
        result = fn(*args, **kwargs)
        if result is not None:
            print(result)
        return result
    except NotImplementedError as e:
        print(f"  [skipped: {e}]")
        return None
    finally:
        print()


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    d_model = 8
    d_k = 4
    seq_len = 5
    token_labels = ["the", "cat", "sat", "on", "mat"]

    # Store results for later steps
    X = None
    Q = K = V = None
    scores = None
    weights_unmasked = None
    output_unmasked = None
    weights_masked = None
    output_masked = None

    # ------------------------------------------------------------------
    # Step 1: Create token vectors
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: Create Token Vectors")
    print("=" * 60)
    X = _try_run("Token embeddings", make_token_vectors, vocab_size=10, d_model=d_model, seq_len=seq_len)
    if X is not None:
        print(f"  Shape: {X.shape}")
        print(f"  First token vector: {X[0].tolist()}")

    # ------------------------------------------------------------------
    # Step 2: Compute Q, K, V projections
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 2: Compute Q, K, V Projections")
    print("=" * 60)
    if X is not None:
        result = _try_run("Q, K, V projections", compute_qkv, X, d_k)
        if result is not None:
            Q, K, V = result
            print(f"  Q shape: {Q.shape}")
            print(f"  K shape: {K.shape}")
            print(f"  V shape: {V.shape}")

    # ------------------------------------------------------------------
    # Step 3: Compute raw attention scores
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 3: Raw Attention Scores (QK^T)")
    print("=" * 60)
    if Q is not None:
        scores = _try_run("Raw scores", raw_attention_scores, Q, K)
        if scores is not None:
            print(f"  Shape: {scores.shape}")
            print(f"  Scores matrix:\n{scores.detach().numpy().round(3)}")

    # ------------------------------------------------------------------
    # Step 4: Apply scaled softmax
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 4: Scaled Softmax Attention Weights")
    print("=" * 60)
    if scores is not None:
        weights_unmasked = _try_run("Scaled softmax weights", scaled_softmax, scores, d_k)
        if weights_unmasked is not None:
            print(f"  Shape: {weights_unmasked.shape}")
            print(f"  Row sums (should be ~1.0): {weights_unmasked.sum(dim=-1).tolist()}")
            print(f"  Weight matrix:\n{weights_unmasked.detach().numpy().round(3)}")

    # ------------------------------------------------------------------
    # Step 5: Compute weighted sum of values
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 5: Attention Output (Weighted Sum of Values)")
    print("=" * 60)
    if weights_unmasked is not None and V is not None:
        output_unmasked = _try_run("Attention output", attention_output, weights_unmasked, V)
        if output_unmasked is not None:
            print(f"  Shape: {output_unmasked.shape}")
            print(f"  First output vector: {output_unmasked[0].detach().numpy().round(3)}")

    # ------------------------------------------------------------------
    # Step 6: Causal mask
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 6: Causal Mask")
    print("=" * 60)
    mask = _try_run("Causal mask", causal_mask, seq_len)
    if mask is not None:
        print(f"  Mask matrix:\n{mask.numpy().round(1)}")

    # ------------------------------------------------------------------
    # Step 7: Masked attention
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 7: Masked Attention Output")
    print("=" * 60)
    if Q is not None:
        output_masked = _try_run("Masked attention", masked_attention, Q, K, V, d_k)
        if output_masked is not None:
            # Recompute weights for visualization
            scores_m = raw_attention_scores(Q, K)
            scaled_m = scores_m / (d_k ** 0.5)
            mask_m = causal_mask(seq_len)
            scaled_m = scaled_m + mask_m
            weights_masked = F.softmax(scaled_m, dim=-1)
            print(f"  Masked weight matrix:\n{weights_masked.detach().numpy().round(3)}")
            print(f"  First output vector: {output_masked[0].detach().numpy().round(3)}")

    # ------------------------------------------------------------------
    # Step 8: Positional embeddings
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 8: Positional Embeddings")
    print("=" * 60)
    if X is not None:
        X_pos = _try_run("Token vectors + position", add_positional_embeddings, X)
        if X_pos is not None:
            print(f"  Shape: {X_pos.shape}")
            print(f"  First token (before pos): {X[0].detach().numpy().round(3)}")
            print(f"  First token (after pos):  {X_pos[0].detach().numpy().round(3)}")

            # Recompute attention with positional embeddings
            Q_pos, K_pos, V_pos = compute_qkv(X_pos, d_k)
            scores_pos = raw_attention_scores(Q_pos, K_pos)
            weights_pos = scaled_softmax(scores_pos, d_k)
            print(f"\n  Attention weights WITH position:\n{weights_pos.detach().numpy().round(3)}")

    # ------------------------------------------------------------------
    # Save comparison plots
    # ------------------------------------------------------------------
    print("=" * 60)
    print("VISUALIZATIONS")
    print("=" * 60)

    if weights_unmasked is not None and weights_masked is not None:
        plot_attention_comparison(
            weights_unmasked.detach().numpy(),
            weights_masked.detach().numpy(),
            token_labels=token_labels,
            filepath=str(OUTPUT_DIR / "attention_comparison.png"),
        )

    if weights_unmasked is not None and X is not None:
        # Recompute with positional embeddings for comparison
        try:
            X_pos = add_positional_embeddings(X)
            Q_pos, K_pos, V_pos = compute_qkv(X_pos, d_k)
            scores_pos = raw_attention_scores(Q_pos, K_pos)
            weights_pos = scaled_softmax(scores_pos, d_k)
            plot_positional_effect(
                weights_unmasked.detach().numpy(),
                weights_pos.detach().numpy(),
                token_labels=token_labels,
                filepath=str(OUTPUT_DIR / "positional_effect.png"),
            )
        except NotImplementedError:
            pass

    # ------------------------------------------------------------------
    # Extra credit: KV cache simulation
    # ------------------------------------------------------------------
    print("=" * 60)
    print("EXTRA CREDIT: KV Cache Simulation")
    print("=" * 60)
    if X is not None and Q is not None:
        try:
            torch.manual_seed(42)
            d_model_val = X.shape[1]
            W_Q_cache = torch.randn(d_model_val, d_k) * 0.1
            W_K_cache = torch.randn(d_model_val, d_k) * 0.1
            W_V_cache = torch.randn(d_model_val, d_k) * 0.1

            # Start with the first token's key and value in the cache
            cached_keys = K[:1]
            cached_values = V[:1]

            print("  Processing tokens one at a time with KV cache:")
            for t in range(1, seq_len):
                new_tok = X[t:t+1]
                output_t, cached_keys, cached_values = kv_cache_step(
                    new_tok, cached_keys, cached_values, W_Q_cache, W_K_cache, W_V_cache, d_k
                )
                print(f"    Token {t} ({token_labels[t]}): cache size = {cached_keys.shape[0]}, output norm = {output_t.norm().item():.4f}")

            # Compare: generation with and without KV cache
            cache_sizes = list(range(1, seq_len + 1))
            times_no_cache = []
            times_with_cache = []

            for n_tokens in cache_sizes:
                X_sub = X[:n_tokens]
                Q_sub, K_sub, V_sub = compute_qkv(X_sub, d_k)

                # Without cache: recompute everything from scratch
                start = time.perf_counter()
                for _ in range(100):
                    scores_full = Q_sub @ K_sub.T / (d_k ** 0.5)
                    w = F.softmax(scores_full, dim=-1)
                    out = w @ V_sub
                times_no_cache.append((time.perf_counter() - start) / 100 * 1000)

                # With cache: only compute the last token's attention
                start = time.perf_counter()
                for _ in range(100):
                    last_q = Q_sub[-1:]
                    s = last_q @ K_sub.T / (d_k ** 0.5)
                    w = F.softmax(s, dim=-1)
                    out = w @ V_sub
                times_with_cache.append((time.perf_counter() - start) / 100 * 1000)

            plot_kv_cache_growth(
                cache_sizes, times_no_cache, times_with_cache,
                filepath=str(OUTPUT_DIR / "kv_cache_growth.png"),
            )
        except NotImplementedError as e:
            print(f"  [skipped: {e}]")

    print("=" * 60)
    print("Done! Check the output/ directory for plots.")
    print("=" * 60)


if __name__ == "__main__":
    main()
