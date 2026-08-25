"""
Module 4 Exercise runner: Assemble GPT-2 and Generate Text

Run with:
    uv run python module_04_architectures/src/main.py

Wires the attention from Module 3 into a complete decoder-only model,
loads real GPT-2 weights, and generates text. Any step that still raises
NotImplementedError is skipped, so you can run after each fill-in.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Make the module root (parent of src/) importable so we can `from exercise import ...`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Also ensure src/ is on the path so we can import sibling helpers
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exercise import (
    EmbeddingLayer,
    FeedForward,
    TransformerBlock,
    GPT2Model,
    load_gpt2_weights,
    greedy_decode,
    sample_with_temperature_topk,
)
from visualization import plot_token_probs

_THIS_DIR = Path(__file__).resolve().parent
_MODULE_DIR = _THIS_DIR.parent
OUTPUT_DIR = _MODULE_DIR / "output"

# GPT-2 small hyperparameters
VOCAB_SIZE = 50257
D_MODEL = 768
N_LAYERS = 12
N_HEADS = 12
D_FF = 3072
MAX_POS = 1024

PROMPT = "The capital of France is"


def _try_run(label: str, fn, *args, **kwargs):
    """Run a function, printing its result or skipping on NotImplementedError."""
    print(f"=== {label} ===")
    try:
        result = fn(*args, **kwargs)
        if result is not None and not isinstance(result, (torch.Tensor, GPT2Model)):
            print(result)
        return result
    except NotImplementedError as e:
        print(f"  [skipped: {e}]")
        return None
    finally:
        print()


def count_parameters(model: torch.nn.Module) -> int:
    """Return the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters())


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Import tokenizer (needed by steps 5-7)
    # ------------------------------------------------------------------
    tokenizer = None
    try:
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Note: transformers not available ({e}). Steps 5-7 will be skipped.")
        print()

    print("=" * 60)
    print("MODULE 4: Assemble GPT-2 and Generate Text")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # Step 1: Embedding layer sanity check
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: Embedding Layer")
    print("=" * 60)
    embed = EmbeddingLayer(vocab_size=VOCAB_SIZE, d_model=D_MODEL)
    dummy_ids = torch.tensor([[0, 1, 2]])
    embed_out = _try_run("Embedding forward", embed, dummy_ids)
    if embed_out is not None:
        print(f"  Output shape: {embed_out.shape}")
        print(f"  Expected: (1, 3, {D_MODEL})")

    # ------------------------------------------------------------------
    # Step 2: Feed-forward sanity check
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 2: Feed-Forward Network")
    print("=" * 60)
    ffn = FeedForward(d_model=D_MODEL, d_ff=D_FF)
    dummy_x = torch.randn(1, 3, D_MODEL)
    ffn_out = _try_run("FFN forward", ffn, dummy_x)
    if ffn_out is not None:
        print(f"  Output shape: {ffn_out.shape}")
        print(f"  Expected: (1, 3, {D_MODEL})")

    # ------------------------------------------------------------------
    # Step 3: Transformer block sanity check
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 3: Transformer Block")
    print("=" * 60)
    block = TransformerBlock(d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF)
    block_out = _try_run("Block forward", block, dummy_x)
    if block_out is not None:
        print(f"  Output shape: {block_out.shape}")
        print(f"  Expected: (1, 3, {D_MODEL})")

    # ------------------------------------------------------------------
    # Step 4: Full model sanity check
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 4: Full GPT2Model Forward Pass")
    print("=" * 60)
    model = GPT2Model(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_ff=D_FF,
        max_pos=MAX_POS,
    )
    logits = _try_run("Model forward", model, dummy_ids)
    if logits is not None:
        print(f"  Output shape: {logits.shape}")
        print(f"  Expected: (1, 3, {VOCAB_SIZE})")
        print(f"  Total parameters: {count_parameters(model):,}")

    # ------------------------------------------------------------------
    # Step 5: Load pretrained weights
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 5: Load Pretrained GPT-2 Weights")
    print("=" * 60)
    if model is not None and tokenizer is not None:
        loaded = _try_run("Load weights", load_gpt2_weights, model)
        if loaded is None:
            # The function returns None on success; check if model has real weights.
            first_param = next(model.parameters())
            if first_param.abs().max().item() < 1e-6:
                print("  [skipped: weights still look uninitialized]")
            else:
                print("  Weights loaded successfully.")
    elif model is None:
        print("=== Load Pretrained GPT-2 Weights ===")
        print("  [skipped: complete Step 4 first]")
        print()
    else:
        print("=== Load Pretrained GPT-2 Weights ===")
        print("  [skipped: transformers library not available]")
        print()

    # ------------------------------------------------------------------
    # Step 6: Greedy decoding
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 6: Greedy Decoding")
    print("=" * 60)
    if model is not None and tokenizer is not None:
        generated = _try_run(
            "Greedy generation",
            greedy_decode,
            model, tokenizer, PROMPT, max_new=10
        )
        if generated is not None:
            print(f"  Prompt:  \"{PROMPT}\"")
            print(f"  Output:  \"{generated}\"")
    elif model is None:
        print("=== Greedy Decoding ===")
        print("  [skipped: complete Step 4 first]")
        print()
    else:
        print("=== Greedy Decoding ===")
        print("  [skipped: transformers library not available]")
        print()

    # ------------------------------------------------------------------
    # Step 7: Sampling with temperature and top-k
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 7: Temperature and Top-k Sampling")
    print("=" * 60)
    if model is not None and tokenizer is not None:
        sampled = _try_run(
            "Sampled generation (temp=0.8, top_k=40)",
            sample_with_temperature_topk,
            model, tokenizer, PROMPT, 10, 0.8, 40
        )
        if sampled is not None:
            print(f"  Prompt:  \"{PROMPT}\"")
            print(f"  Output:  \"{sampled}\"")

        # Also run with higher temperature to show the difference
        if sampled is not None:
            sampled_hot = _try_run(
                "Sampled generation (temp=1.4, top_k=40)",
                sample_with_temperature_topk,
                model, tokenizer, PROMPT, 10, 1.4, 40
            )
            if sampled_hot is not None:
                print(f"  Prompt:  \"{PROMPT}\"")
                print(f"  Output:  \"{sampled_hot}\"")
    elif model is None:
        print("=== Temperature and Top-k Sampling ===")
        print("  [skipped: complete Step 4 first]")
        print()
    else:
        print("=== Temperature and Top-k Sampling ===")
        print("  [skipped: transformers library not available]")
        print()

    # ------------------------------------------------------------------
    # Visualize next-token probabilities for the prompt
    # ------------------------------------------------------------------
    print("=" * 60)
    print("VISUALIZATIONS")
    print("=" * 60)
    if model is not None and logits is not None and tokenizer is not None:
        try:
            token_ids = tokenizer.encode(PROMPT, return_tensors="pt")
            with torch.no_grad():
                logits_prompt = model(token_ids)
            next_logits = logits_prompt[0, -1, :]
            probs = torch.softmax(next_logits, dim=-1)
            topk = torch.topk(probs, k=15)
            tokens = [tokenizer.decode([idx.item()]) for idx in topk.indices]
            plot_token_probs(
                tokens, topk.values.cpu().numpy(),
                title=f"Next-token probabilities after: '{PROMPT}'",
                filepath=str(OUTPUT_DIR / "token_probs.png"),
            )
        except NotImplementedError:
            pass
        except Exception as e:
            print(f"  Visualization error: {e}")

    print("=" * 60)
    print("Done! Check the output/ directory for plots.")
    print("=" * 60)


if __name__ == "__main__":
    main()
