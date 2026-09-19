"""
Module 3 Exercise runner: Attention Mechanisms

Run with:
    uv run python module_03_attention/src/main.py

Every step is tagged on its header line, then its output follows: the
matrices your code produced for the 5-token sentence and the result of
each test in tests/. The tags are:

    CORRECT     every test for the step passed
    INCORRECT   your code ran but at least one test failed (details follow)
    INCOMPLETE  the function still raises NotImplementedError

Add --step N to run one step (1-8, or "ec" for extra credit).
Add --solution to run the finished answers from solution/exercise.py.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import torch
import torch.nn.functional as F

# Make the module root (parent of src/) importable so we can `import exercise`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Also ensure src/ is on the path so we can import sibling helpers
sys.path.insert(0, str(Path(__file__).resolve().parent))

# `--solution` swaps in the finished answers from solution/exercise.py.
# Registering it as "exercise" before the imports below means every
# `from exercise import ...` in this file picks it up with no other change.
if "--solution" in sys.argv:
    import importlib.util

    sys.argv.remove("--solution")
    _sol = Path(__file__).resolve().parent.parent / "solution" / "exercise.py"
    _spec = importlib.util.spec_from_file_location("exercise", _sol)
    _exercise = importlib.util.module_from_spec(_spec)
    sys.modules["exercise"] = _exercise
    _spec.loader.exec_module(_exercise)

from exercise import (
    TinyAttentionLayer,
    add_positional_embeddings,
    kv_cache_step,
)
from src.embeddings import make_token_vectors

# One test file per step lives in tests/
from tests.test_extra_credit import check_kv_cache_step
from tests.test_step2_qkv import check_compute_qkv
from tests.test_step3_scores import check_raw_attention_scores
from tests.test_step4_softmax import check_scaled_softmax
from tests.test_step5_output import check_attention_output
from tests.test_step6_mask import check_causal_mask
from tests.test_step7_masked import check_masked_attention
from tests.test_step8_positional import check_add_positional_embeddings
from visualization import (
    plot_attention_comparison,
    plot_kv_cache_growth,
    plot_positional_effect,
)

# Plots go to output/ inside this module (the parent of src/)
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

# The toy sentence and the sizes of the attention layer
TOKENS = ["the", "cat", "sat", "on", "mat"]
D_MODEL = 8  # width of each token embedding
D_K = 4      # width of each query, key, and value vector

# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

# The three possible outcomes for a step
CORRECT = "CORRECT"
INCORRECT = "INCORRECT"
INCOMPLETE = "INCOMPLETE"

# ANSI color codes, used only when printing to a real terminal
_COLORS = {CORRECT: "\033[32m", INCORRECT: "\033[31m", INCOMPLETE: "\033[90m"}
_RESET = "\033[0m"


def _tag(status: str) -> str:
    """Format a status label in a fixed-width column, colored on a terminal."""
    label = f"{status:<10}"
    if sys.stdout.isatty():
        return f"{_COLORS[status]}{label}{_RESET}"
    return label


def _print_checks(checks) -> None:
    """Print one line per test, with details under any that failed."""
    for check in checks:
        print(f"  {_tag(CORRECT if check.passed else INCORRECT)} {check.name}")
        if not check.passed and check.detail:
            for line in check.detail.split("\n"):
                print(f"             {line.strip()}")


def run_step(title: str, show, check) -> str:
    """Run one step and print its header, tag, output, and test results.

    `show()` prints whatever the student's code produces (training progress,
    saved plots). `check()` returns the list of Check results for the step.

    The tag goes on the header line, so the output is captured first and
    printed after the tag is known. Returns CORRECT, INCORRECT, or INCOMPLETE.
    """
    buffer = io.StringIO()
    checks = []
    note = ""
    try:
        with redirect_stdout(buffer):
            show()
        checks = check()
        status = CORRECT if all(c.passed for c in checks) else INCORRECT
    except NotImplementedError as e:
        # The student has not filled in this blank yet
        status, note = INCOMPLETE, str(e)
    except Exception as e:  # noqa: BLE001 - show students any crash, whatever its type
        status, note = INCORRECT, f"your code crashed: {type(e).__name__}: {e}"

    print(f"=== {title} === {_tag(status).rstrip()}")
    if note:
        print(f"  {note}")
    output = buffer.getvalue()
    if output and status != INCOMPLETE:
        print(output, end="" if output.endswith("\n") else "\n")
    _print_checks(checks)
    print()
    return status


# ---------------------------------------------------------------------------
# Demo helpers
# ---------------------------------------------------------------------------


def _need(step: str, why: str, fn, *args):
    """Call an earlier step's function for this step's demo.

    If that earlier step is still unfinished, say which step is missing
    instead of repeating that step's own TODO message.
    """
    try:
        return fn(*args)
    except NotImplementedError:
        raise NotImplementedError(f"needs Step {step} to {why}") from None


def _qkv(layer: TinyAttentionLayer, X: torch.Tensor, why: str):
    """Q, K, V for the demo, from your Step 2."""
    return _need("2 (compute_qkv)", why, layer.compute_qkv, X)


def _scores(layer: TinyAttentionLayer, X: torch.Tensor, why: str) -> torch.Tensor:
    """Raw scores for the demo, from your Steps 2 and 3."""
    Q, K, _ = _qkv(layer, X, why)
    return _need("3 (raw_attention_scores)", why, layer.raw_attention_scores, Q, K)


def _weights(layer: TinyAttentionLayer, X: torch.Tensor, why: str) -> torch.Tensor:
    """Unmasked attention weights for the demo, from your Steps 2, 3, and 4."""
    scores = _scores(layer, X, why)
    return _need("4 (scaled_softmax)", why, layer.scaled_softmax, scores)


def _print_matrix(M: torch.Tensor, columns: list[str] | None = None, fmt: str = "{:8.4f}") -> None:
    """Print a small matrix with one token label per row (and per column, if given)."""
    if columns:
        print("       " + "".join(f"{c:>8}" for c in columns))
    for label, row in zip(TOKENS, M.tolist()):
        print(f"  {label:>4} " + "".join(fmt.format(v) for v in row))


def _row(v: torch.Tensor) -> str:
    """One vector as a bracketed row of 3-decimal numbers."""
    return "[" + " ".join(f"{x:6.3f}" for x in v.tolist()) + "]"


# ---------------------------------------------------------------------------
# The steps
# ---------------------------------------------------------------------------


def step_1(X: torch.Tensor) -> str:
    """Provided: the 5 token embeddings that every later step attends over."""

    def show():
        print(f"  Tokens: {' '.join(TOKENS)}")
        print(f"  X has shape {tuple(X.shape)}: one row of d_model={D_MODEL} numbers per token")
        print(f"  the: {_row(X[0])}")

    # Nothing to test: this step is written for you
    return run_step("Step 1: make_token_vectors() (provided)", show, lambda: [])


def step_2(layer: TinyAttentionLayer, X: torch.Tensor) -> str:
    def show():
        Q, K, V = layer.compute_qkv(X)
        print(f"  Q, K, V shapes: {tuple(Q.shape)}, {tuple(K.shape)}, {tuple(V.shape)}")
        print("  Q, one query vector per token:")
        _print_matrix(Q)

    return run_step("Step 2: compute_qkv()", show, lambda: check_compute_qkv(TinyAttentionLayer))


def step_3(layer: TinyAttentionLayer, X: torch.Tensor) -> str:
    def show():
        # Try your function on a tiny input first, so an unfinished one shows its own TODO
        layer.raw_attention_scores(torch.zeros(1, D_K), torch.zeros(1, D_K))
        Q, K, _ = _qkv(layer, X, "make the queries and keys it scores")
        scores = layer.raw_attention_scores(Q, K)
        print(f"  Scores {tuple(scores.shape)}, row = query, column = key:")
        _print_matrix(scores, TOKENS)

    return run_step("Step 3: raw_attention_scores()", show,
                    lambda: check_raw_attention_scores(TinyAttentionLayer))


def step_4(layer: TinyAttentionLayer, X: torch.Tensor) -> str:
    def show():
        # Try your function on a tiny input first, so an unfinished one shows its own TODO
        layer.scaled_softmax(torch.zeros(1, 1))
        scores = _scores(layer, X, "make the scores it normalizes")
        weights = layer.scaled_softmax(scores)
        print(f"  Weights {tuple(weights.shape)}, row = query, column = key:")
        _print_matrix(weights, TOKENS)
        print(f"  Row sums: {[round(s, 4) for s in weights.sum(dim=-1).tolist()]}")

    return run_step("Step 4: scaled_softmax()", show, lambda: check_scaled_softmax(TinyAttentionLayer))


def step_5(layer: TinyAttentionLayer, X: torch.Tensor) -> str:
    def show():
        # Try your function on a tiny input first, so an unfinished one shows its own TODO
        layer.attention_output(torch.ones(1, 1), torch.zeros(1, D_K))
        why = "make the weights and values it combines"
        _, _, V = _qkv(layer, X, why)
        weights = _weights(layer, X, why)
        output = layer.attention_output(weights, V)
        print(f"  Output {tuple(output.shape)}, one blended value vector per token:")
        _print_matrix(output)

    return run_step("Step 5: attention_output()", show, lambda: check_attention_output(TinyAttentionLayer))


def step_6(layer: TinyAttentionLayer) -> str:
    def show():
        mask = layer.causal_mask(len(TOKENS))
        print("  Mask for 5 tokens, row = query, column = key:")
        _print_matrix(mask, TOKENS, fmt="{:8.0f}")

    return run_step("Step 6: causal_mask()", show, lambda: check_causal_mask(TinyAttentionLayer))


def step_7(layer: TinyAttentionLayer, X: torch.Tensor) -> str:
    """No new code: masked_attention() chains your Steps 2, 3, 5, and 6."""

    def show():
        # Try each earlier step first, so an unfinished one is named here
        why = "run masked_attention() and plot it beside unmasked attention"
        Q, K, V = _qkv(layer, X, why)
        _need("3 (raw_attention_scores)", why, layer.raw_attention_scores, Q, K)
        _need("5 (attention_output)", why, layer.attention_output, torch.eye(len(TOKENS)), V)
        _need("6 (causal_mask)", why, layer.causal_mask, len(TOKENS))
        unmasked = _weights(layer, X, why)

        _, masked = layer.masked_attention(X)
        print("  Masked weights, row = query, column = key:")
        _print_matrix(masked, TOKENS)
        plot_attention_comparison(
            unmasked.detach().numpy(),
            masked.detach().numpy(),
            token_labels=TOKENS,
            filepath=str(OUTPUT_DIR / "attention_comparison.png"),
        )

    return run_step("Step 7: masked_attention() (Steps 2, 3, 5, 6 together)", show,
                    lambda: check_masked_attention(TinyAttentionLayer, X))


def step_8(layer: TinyAttentionLayer, X: torch.Tensor) -> str:
    def show():
        X_pos = add_positional_embeddings(X)
        print(f"  the, before position: {_row(X[0])}")
        print(f"  the, after position:  {_row(X_pos[0])}")

        # Same attention layer, run on the tokens with and without position added
        why = "compare attention with and without position"
        unmasked = _weights(layer, X, why)
        with_pos = _weights(layer, X_pos, why)
        print("  Unmasked weights with position, row = query, column = key:")
        _print_matrix(with_pos, TOKENS)
        plot_positional_effect(
            unmasked.detach().numpy(),
            with_pos.detach().numpy(),
            token_labels=TOKENS,
            filepath=str(OUTPUT_DIR / "positional_effect.png"),
        )

    return run_step("Step 8: add_positional_embeddings()", show,
                    lambda: check_add_positional_embeddings(add_positional_embeddings))


def time_attention(layer: TinyAttentionLayer, X: torch.Tensor) -> tuple[list[int], list[float], list[float]]:
    """Time one new token's attention with and without a KV cache.

    Args:
        layer: The attention layer whose weights project the tokens.
        X: Token embeddings, shape (seq_len, d_model).

    Returns:
        (lengths, ms_without_cache, ms_with_cache): average milliseconds per
        token for each sequence length 1..seq_len.
    """
    lengths = list(range(1, X.shape[0] + 1))
    no_cache, with_cache = [], []
    for n in lengths:
        Q, K, V = _qkv(layer, X[:n], "time attention with and without the cache")

        # Without a cache: recompute attention for every token from scratch
        start = time.perf_counter()
        for _ in range(100):
            weights = F.softmax(Q @ K.T / (D_K ** 0.5), dim=-1)
            _ = weights @ V
        no_cache.append((time.perf_counter() - start) / 100 * 1000)

        # With a cache: only the newest token's query attends over the stored keys
        start = time.perf_counter()
        for _ in range(100):
            weights = F.softmax(Q[-1:] @ K.T / (D_K ** 0.5), dim=-1)
            _ = weights @ V
        with_cache.append((time.perf_counter() - start) / 100 * 1000)
    return lengths, no_cache, with_cache


def step_extra(layer: TinyAttentionLayer, X: torch.Tensor) -> str:
    """Generate one token at a time, reusing every earlier key and value."""

    def show():
        # Start with an empty cache: zero rows of width d_k
        keys = values = torch.zeros(0, D_K)
        print("  Feeding the tokens in one at a time:")
        for t, label in enumerate(TOKENS):
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

STEP_CHOICES = ["1", "2", "3", "4", "5", "6", "7", "8", "ec"]


def main():
    parser = argparse.ArgumentParser(description="Attention mechanisms")
    parser.add_argument("--step", choices=[*STEP_CHOICES, "all"], default="all",
                        help="Which step to run (default: all)")
    args = parser.parse_args()
    steps = STEP_CHOICES if args.step == "all" else [args.step]

    OUTPUT_DIR.mkdir(exist_ok=True)
    # Step 1 (provided): the 5 token vectors every step attends over
    X = make_token_vectors(vocab_size=10, d_model=D_MODEL, seq_len=len(TOKENS))
    # One attention layer with fixed random weights, shared by every step
    layer = TinyAttentionLayer(d_model=D_MODEL, d_k=D_K)

    for step in steps:
        if step == "1":
            step_1(X)
        elif step == "2":
            step_2(layer, X)
        elif step == "3":
            step_3(layer, X)
        elif step == "4":
            step_4(layer, X)
        elif step == "5":
            step_5(layer, X)
        elif step == "6":
            step_6(layer)
        elif step == "7":
            step_7(layer, X)
        elif step == "8":
            step_8(layer, X)
        elif step == "ec":
            step_extra(layer, X)


if __name__ == "__main__":
    main()
