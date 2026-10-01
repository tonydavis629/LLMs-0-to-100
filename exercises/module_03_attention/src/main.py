"""
Module 3 Exercise runner: Attention Mechanisms

Run with:
    uv run python module_03_attention/src/main.py

Each step below runs one piece of the attention layer you build in
exercise.py on the 5-token sentence "the cat sat on mat", then tests it.
Read top to bottom, the steps follow the tokens through attention:

    1. token embeddings X (provided)
    2. X -> queries Q, keys K, values V
    3. Q and K -> raw scores (how well each query matches each key)
    4. scores -> attention weights (scaled softmax)
    5. weights and V -> output (a blend of value vectors per token)
    6. the causal mask (no token may look ahead)
    7. masked attention: Steps 2, 3, 5, and 6 together
    8. adding position to the embeddings
    Extra credit: generating one token at a time with a KV cache

Add --step N to run one step (1-8, or "ec" for extra credit).
Add --solution to run the finished answers from solution/exercise.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Make the module root (parent of src/) importable so we can `from exercise import ...`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# With --solution, swap in solution/exercise.py before anything imports `exercise`
from src.solution import use_solution_if_requested

use_solution_if_requested()

from exercise import (
    TinyAttentionLayer,
    add_positional_embeddings,
    kv_cache_step,
)
from src.embeddings import make_token_vectors
from src.prerequisites import earlier_qkv, earlier_scores, earlier_weights, need
from src.printing import format_row, print_matrix
from src.reporting import run_step
from src.timing import time_attention
from src.visualization import (
    plot_attention_comparison,
    plot_kv_cache_growth,
    plot_positional_effect,
)

# One test file per step lives in tests/
from tests.test_extra_credit import check_kv_cache_step
from tests.test_step2_qkv import check_compute_qkv
from tests.test_step3_scores import check_raw_attention_scores
from tests.test_step4_softmax import check_scaled_softmax
from tests.test_step5_output import check_attention_output
from tests.test_step6_mask import check_causal_mask
from tests.test_step7_masked import check_masked_attention
from tests.test_step8_positional import check_add_positional_embeddings

# Plots go to output/ inside this module (the parent of src/)
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

# The toy sentence and the sizes of the attention layer
TOKENS = ["the", "cat", "sat", "on", "mat"]
D_MODEL = 8  # width of each token embedding
D_K = 4      # width of each query, key, and value vector

# ---------------------------------------------------------------------------
# Step 1: token embeddings (provided)
# ---------------------------------------------------------------------------


def step_1(layer: TinyAttentionLayer, X: torch.Tensor) -> str:
    def show():
        print(f"  Tokens: {' '.join(TOKENS)}")
        print(f"  X has shape {tuple(X.shape)}: one row of d_model={D_MODEL} numbers per token")
        print(f"  the: {format_row(X[0])}")

    # Nothing to test: this step is written for you
    return run_step("Step 1: make_token_vectors() (provided)", show, lambda: [])


# ---------------------------------------------------------------------------
# Step 2: X -> queries, keys, values
# ---------------------------------------------------------------------------


def step_2(layer: TinyAttentionLayer, X: torch.Tensor) -> str:
    def show():
        Q, K, V = layer.compute_qkv(X)
        print(f"  Q, K, V shapes: {tuple(Q.shape)}, {tuple(K.shape)}, {tuple(V.shape)}")
        print("  Q, one query vector per token:")
        print_matrix(Q, TOKENS)

    return run_step("Step 2: compute_qkv()", show, lambda: check_compute_qkv(TinyAttentionLayer))


# ---------------------------------------------------------------------------
# Step 3: Q and K -> raw scores
# ---------------------------------------------------------------------------


def step_3(layer: TinyAttentionLayer, X: torch.Tensor) -> str:
    def show():
        # Try your function on a tiny input first, so an unfinished one shows its own TODO
        layer.raw_attention_scores(torch.zeros(1, D_K), torch.zeros(1, D_K))
        Q, K, _ = earlier_qkv(layer, X, "make the queries and keys it scores")
        scores = layer.raw_attention_scores(Q, K)  # shape (5, 5): one score per query/key pair
        print(f"  Scores {tuple(scores.shape)}, row = query, column = key:")
        print_matrix(scores, TOKENS, TOKENS)

    return run_step("Step 3: raw_attention_scores()", show,
                    lambda: check_raw_attention_scores(TinyAttentionLayer))


# ---------------------------------------------------------------------------
# Step 4: scores -> attention weights
# ---------------------------------------------------------------------------


def step_4(layer: TinyAttentionLayer, X: torch.Tensor) -> str:
    def show():
        # Try your function on a tiny input first, so an unfinished one shows its own TODO
        layer.scaled_softmax(torch.zeros(1, 1))
        scores = earlier_scores(layer, X, "make the scores it normalizes")
        weights = layer.scaled_softmax(scores)  # each row now sums to 1
        print(f"  Weights {tuple(weights.shape)}, row = query, column = key:")
        print_matrix(weights, TOKENS, TOKENS)
        print(f"  Row sums: {[round(s, 4) for s in weights.sum(dim=-1).tolist()]}")

    return run_step("Step 4: scaled_softmax()", show, lambda: check_scaled_softmax(TinyAttentionLayer))


# ---------------------------------------------------------------------------
# Step 5: weights and V -> output
# ---------------------------------------------------------------------------


def step_5(layer: TinyAttentionLayer, X: torch.Tensor) -> str:
    def show():
        # Try your function on a tiny input first, so an unfinished one shows its own TODO
        layer.attention_output(torch.ones(1, 1), torch.zeros(1, D_K))
        purpose = "make the weights and values it combines"
        _, _, V = earlier_qkv(layer, X, purpose)
        weights = earlier_weights(layer, X, purpose)
        output = layer.attention_output(weights, V)  # each token's weighted blend of the values
        print(f"  Output {tuple(output.shape)}, one blended value vector per token:")
        print_matrix(output, TOKENS)

    return run_step("Step 5: attention_output()", show, lambda: check_attention_output(TinyAttentionLayer))


# ---------------------------------------------------------------------------
# Step 6: the causal mask
# ---------------------------------------------------------------------------


def step_6(layer: TinyAttentionLayer, X: torch.Tensor) -> str:
    def show():
        mask = layer.causal_mask(len(TOKENS))  # 1 where a query may look, 0 where it may not
        print("  Mask for 5 tokens, row = query, column = key:")
        print_matrix(mask, TOKENS, TOKENS, fmt="{:8.0f}")

    return run_step("Step 6: causal_mask()", show, lambda: check_causal_mask(TinyAttentionLayer))


# ---------------------------------------------------------------------------
# Step 7: masked attention (no new code: Steps 2, 3, 5, and 6 together)
# ---------------------------------------------------------------------------


def step_7(layer: TinyAttentionLayer, X: torch.Tensor) -> str:
    def show():
        # Try each earlier step first, so an unfinished one is named here
        purpose = "run masked_attention() and plot it beside unmasked attention"
        Q, K, V = earlier_qkv(layer, X, purpose)
        need("3 (raw_attention_scores)", purpose, layer.raw_attention_scores, Q, K)
        need("5 (attention_output)", purpose, layer.attention_output, torch.eye(len(TOKENS)), V)
        need("6 (causal_mask)", purpose, layer.causal_mask, len(TOKENS))
        unmasked = earlier_weights(layer, X, purpose)

        _, masked = layer.masked_attention(X)
        print("  Masked weights, row = query, column = key:")
        print_matrix(masked, TOKENS, TOKENS)
        plot_attention_comparison(
            unmasked.detach().numpy(),
            masked.detach().numpy(),
            token_labels=TOKENS,
            filepath=str(OUTPUT_DIR / "attention_comparison.png"),
        )

    return run_step("Step 7: masked_attention() (Steps 2, 3, 5, 6 together)", show,
                    lambda: check_masked_attention(TinyAttentionLayer, X))


# ---------------------------------------------------------------------------
# Step 8: adding position to the embeddings
# ---------------------------------------------------------------------------


def step_8(layer: TinyAttentionLayer, X: torch.Tensor) -> str:
    def show():
        X_pos = add_positional_embeddings(X)
        print(f"  the, before position: {format_row(X[0])}")
        print(f"  the, after position:  {format_row(X_pos[0])}")

        # Same attention layer, run on the tokens with and without position added
        purpose = "compare attention with and without position"
        unmasked = earlier_weights(layer, X, purpose)
        with_pos = earlier_weights(layer, X_pos, purpose)
        print("  Unmasked weights with position, row = query, column = key:")
        print_matrix(with_pos, TOKENS, TOKENS)
        plot_positional_effect(
            unmasked.detach().numpy(),
            with_pos.detach().numpy(),
            token_labels=TOKENS,
            filepath=str(OUTPUT_DIR / "positional_effect.png"),
        )

    return run_step("Step 8: add_positional_embeddings()", show,
                    lambda: check_add_positional_embeddings(add_positional_embeddings))


# ---------------------------------------------------------------------------
# Extra credit: one token at a time with a KV cache
# ---------------------------------------------------------------------------


def step_extra(layer: TinyAttentionLayer, X: torch.Tensor) -> str:
    def show():
        # Start with an empty cache: zero rows of width d_k
        keys = values = torch.zeros(0, D_K)
        print("  Feeding the tokens in one at a time:")
        for t, label in enumerate(TOKENS):
            # Each call adds this token's key and value to the cache and attends over all of them
            output, keys, values = kv_cache_step(
                X[t:t + 1], keys, values, layer.W_Q, layer.W_K, layer.W_V, D_K
            )
            print(f"    Token {t} ({label}): cache size = {keys.shape[0]}, "
                  f"output norm = {output.norm().item():.4f}")
        plot_kv_cache_growth(*time_attention(layer, X), filepath=str(OUTPUT_DIR / "kv_cache_growth.png"))

    return run_step("Extra Credit: kv_cache_step()", show, lambda: check_kv_cache_step(kv_cache_step))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

STEPS = {"1": step_1, "2": step_2, "3": step_3, "4": step_4, "5": step_5,
         "6": step_6, "7": step_7, "8": step_8, "ec": step_extra}


def main():
    parser = argparse.ArgumentParser(description="Attention mechanisms")
    parser.add_argument("--step", choices=[*STEPS, "all"], default="all",
                        help="Which step to run (default: all)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    # Step 1 (provided): the 5 token vectors every step attends over
    X = make_token_vectors(vocab_size=10, d_model=D_MODEL, seq_len=len(TOKENS))
    # One attention layer with fixed random weights, shared by every step
    layer = TinyAttentionLayer(d_model=D_MODEL, d_k=D_K)

    for number, step in STEPS.items():
        if args.step in ("all", number):
            step(layer, X)


if __name__ == "__main__":
    main()
