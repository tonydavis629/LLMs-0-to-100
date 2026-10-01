"""
Module 4 Exercise runner: Assemble GPT-2 and Generate Text

Run with:
    uv run python module_04_architectures/src/main.py

Each step below runs one piece of the GPT-2 you build in exercise.py, then
tests it. Read top to bottom, the steps follow a prompt through the model:

    1. text -> token IDs -> embedding vectors
    2. one feed-forward network
    3. one transformer block (attention + feed-forward)
    4. the whole model, with the real GPT-2 weights loaded -> next-token scores
    5. greedy decoding: always take the most likely next token
    6. sampling: temperature + top-k

Add --step N to run one step (1-6).
Add --solution to run the finished answers from solution/exercise.py.
"""

from __future__ import annotations

import argparse
import sys
from functools import cache
from pathlib import Path

import torch

# Make the module root (parent of src/) importable so we can `from exercise import ...`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# With --solution, swap in solution/exercise.py before anything imports `exercise`
from src.solution import use_solution_if_requested

use_solution_if_requested()

from exercise import (
    EmbeddingLayer,
    FeedForward,
    GPT2Model,
    TransformerBlock,
    greedy_decode,
    sample_with_temperature_topk,
)
from src.prerequisites import require
from src.pretrained import load_gpt2_weights, load_tokenizer
from src.reporting import run_step
from src.visualization import plot_token_probs

# One test file per step lives in tests/
from tests.test_step1_embedding import check_embedding
from tests.test_step2_ffn import check_feed_forward
from tests.test_step3_block import check_transformer_block
from tests.test_step4_model import check_gpt2_forward, check_matches_hugging_face
from tests.test_step5_greedy import check_greedy_decode, check_greedy_on_gpt2
from tests.test_step6_temperature import check_temperature

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

# GPT-2 small hyperparameters
VOCAB_SIZE = 50257  # number of distinct tokens
D_MODEL = 768       # length of each token's vector
N_LAYERS = 12       # transformer blocks stacked on top of each other
N_HEADS = 12        # attention heads per block
D_FF = 3072         # hidden width of each feed-forward network (4 x D_MODEL)
MAX_POS = 1024      # longest sequence the model can read

PROMPT = "The capital of France is"


def count_parameters(model: torch.nn.Module) -> int:
    """Return the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters())


@cache  # build and load once, then reuse the same model in Steps 4-6
def load_gpt2():
    """Build your GPT2Model and copy the real GPT-2 weights into it.

    Also returns Hugging Face's own copy of GPT-2, which the tests use as a reference.
    """
    model = GPT2Model(vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layers=N_LAYERS,
                      n_heads=N_HEADS, d_ff=D_FF, max_pos=MAX_POS)
    hf_model = load_gpt2_weights(model)
    return model.eval(), hf_model


# ---------------------------------------------------------------------------
# Step 1: text -> token IDs -> embedding vectors
# ---------------------------------------------------------------------------


def step_1() -> str:
    def show():
        tokenizer = load_tokenizer()
        ids = tokenizer.encode(PROMPT, return_tensors="pt")  # shape (1, 5): one ID per token
        embed = EmbeddingLayer(vocab_size=VOCAB_SIZE, d_model=D_MODEL, max_pos=MAX_POS)
        with torch.no_grad():
            vectors = embed(ids)  # shape (1, 5, 768): one vector per token
        print(f'Prompt: "{PROMPT}"')
        print(f"Tokens: {[tokenizer.decode([i]) for i in ids[0].tolist()]}")
        print(f"Token IDs: {ids[0].tolist()}")
        print(f"Embeddings: {tuple(vectors.shape)}, one {D_MODEL}-number vector per token")

    return run_step("Step 1: EmbeddingLayer.forward()", show,
                    lambda: check_embedding(EmbeddingLayer))


# ---------------------------------------------------------------------------
# Step 2: one feed-forward network
# ---------------------------------------------------------------------------


def step_2() -> str:
    def show():
        ffn = FeedForward(d_model=D_MODEL, d_ff=D_FF).eval()
        x = torch.randn(1, 5, D_MODEL)  # 5 random token vectors
        with torch.no_grad():
            out = ffn(x)
        print(f"Input {tuple(x.shape)} -> output {tuple(out.shape)}, widening to d_ff={D_FF} in between")
        print(f"Parameters: {count_parameters(ffn):,}")

    return run_step("Step 2: FeedForward.forward()", show,
                    lambda: check_feed_forward(FeedForward))


# ---------------------------------------------------------------------------
# Step 3: one transformer block (attention + feed-forward)
# ---------------------------------------------------------------------------


def step_3() -> str:
    def show():
        require("3", needs="2", purpose="run a full-size block")  # the block calls your FFN
        block = TransformerBlock(d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF).eval()
        x = torch.randn(1, 5, D_MODEL)  # 5 random token vectors
        with torch.no_grad():
            out = block(x)
        print(f"Input {tuple(x.shape)} -> output {tuple(out.shape)}")
        print(f"Parameters: {count_parameters(block):,} per block, "
              f"{count_parameters(block.ffn):,} of them in the FFN")

    return run_step("Step 3: TransformerBlock.forward()", show,
                    lambda: check_transformer_block(TransformerBlock))


# ---------------------------------------------------------------------------
# Step 4: the whole model with real GPT-2 weights -> next-token scores
# ---------------------------------------------------------------------------


def step_4() -> str:
    def show():
        require("4", needs="123", purpose="run the real GPT-2")
        model, _ = load_gpt2()
        tokenizer = load_tokenizer()
        ids = tokenizer.encode(PROMPT, return_tensors="pt")
        with torch.no_grad():
            logits = model(ids)  # shape (1, 5, 50257): a score for every token at every position

        # GPT-2 small is usually quoted at 124M because it reuses the token
        # embedding as the LM head; this model keeps a separate lm_head
        total = count_parameters(model)
        shared = total - model.lm_head.weight.numel()
        print(f"Parameters: {total:,} ({shared:,} if lm_head shared the token embedding)")
        print(f"Logits: {tuple(logits.shape)}, one score per vocabulary token at each position")

        # The last position's scores are the model's guess for the token after the prompt
        probs = torch.softmax(logits[0, -1], dim=-1)
        top = torch.topk(probs, k=15)
        tokens = [tokenizer.decode([i]) for i in top.indices.tolist()]
        best = ", ".join(f"{t!r} {p:.1%}" for t, p in zip(tokens[:5], top.values.tolist()))
        print(f"Most likely next tokens: {best}")
        plot_token_probs(tokens, top.values.numpy(), title=f"Next-token probabilities after: '{PROMPT}'",
                         filepath=str(OUTPUT_DIR / "token_probs.png"))

    def check():
        model, hf_model = load_gpt2()
        ids = load_tokenizer().encode(PROMPT, return_tensors="pt")
        return check_gpt2_forward(GPT2Model) + check_matches_hugging_face(model, hf_model, ids)

    return run_step("Step 4: GPT2Model.forward()", show, check)


# ---------------------------------------------------------------------------
# Step 5: greedy decoding
# ---------------------------------------------------------------------------


def step_5() -> str:
    def show():
        require("5", needs="1234", purpose="generate text with your GPT-2")
        model, _ = load_gpt2()
        text = greedy_decode(model, load_tokenizer(), PROMPT, max_new=10)
        print(f'Prompt:  "{PROMPT}"')
        print(f'Greedy:  "{text}"')

    def check():
        _, hf_model = load_gpt2()
        return (check_greedy_decode(greedy_decode)
                + check_greedy_on_gpt2(greedy_decode, hf_model, load_tokenizer(), PROMPT))

    return run_step("Step 5: greedy_decode()", show, check)


# ---------------------------------------------------------------------------
# Step 6: sampling with temperature and top-k
# ---------------------------------------------------------------------------


def step_6() -> str:
    def show():
        require("6", needs="1234", purpose="generate text with your GPT-2")
        model, _ = load_gpt2()
        print(f'Prompt:  "{PROMPT}"')
        for temperature in (0.8, 1.4):  # a cautious and an adventurous setting
            torch.manual_seed(0)  # the same random draws on every run, so the output repeats
            text = sample_with_temperature_topk(model, load_tokenizer(), PROMPT, max_new=10,
                                                temperature=temperature, top_k=40)
            print(f'T={temperature}, top_k=40:  "{text}"')

    return run_step("Step 6: sample_with_temperature_topk()", show,
                    lambda: check_temperature(sample_with_temperature_topk))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

STEPS = {"1": step_1, "2": step_2, "3": step_3, "4": step_4, "5": step_5, "6": step_6}


def main():
    parser = argparse.ArgumentParser(description="Assemble GPT-2 and generate text")
    parser.add_argument("--step", choices=[*STEPS, "all"], default="all",
                        help="Which step to run (default: all)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    for number, step in STEPS.items():
        if args.step in ("all", number):
            step()


if __name__ == "__main__":
    main()
