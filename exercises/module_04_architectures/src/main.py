"""
Module 4 Exercise runner: Assemble GPT-2 and Generate Text

Run with:
    uv run python module_04_architectures/src/main.py

Wires the attention from Module 3 into a complete decoder-only model,
loads real GPT-2 weights, and generates text. Every step is tagged on its
header line, then its output follows: what your code produced and the
result of each test in tests/. The tags are:

    CORRECT     every test for the step passed
    INCORRECT   your code ran but at least one test failed (details follow)
    INCOMPLETE  the function still raises NotImplementedError

Add --step N to run one step (1-6).
Add --solution to run the finished answers from solution/exercise.py.
"""

from __future__ import annotations

import argparse
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import torch

# Make the module root (parent of src/) importable so we can `from exercise import ...`
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
    EmbeddingLayer,
    FeedForward,
    GPT2Model,
    TransformerBlock,
    greedy_decode,
    sample_with_temperature_topk,
)
from src.pretrained import load_gpt2_weights, load_tokenizer

# One test file per step lives in tests/
from tests.fakes import CountingModel, FixedModel, NumberTokenizer, tiny_gpt2
from tests.test_step1_embedding import check_embedding
from tests.test_step2_ffn import check_feed_forward
from tests.test_step3_block import check_transformer_block
from tests.test_step4_model import check_gpt2_forward, check_matches_hugging_face
from tests.test_step5_greedy import check_greedy_decode, check_greedy_on_gpt2
from tests.test_step6_temperature import check_temperature
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


def count_parameters(model: torch.nn.Module) -> int:
    """Return the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters())


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
# Shared pieces: earlier-step probes and the pretrained GPT-2
# ---------------------------------------------------------------------------

# Short names for the steps, used in "needs Step N (...)" messages
STEP_NAMES = {
    "1": "EmbeddingLayer.forward",
    "2": "FeedForward.forward",
    "3": "TransformerBlock.forward",
    "4": "GPT2Model.forward",
    "5": "greedy_decode",
    "6": "sample_with_temperature_topk",
}


def _probe_block():
    """Run a tiny block with its FFN swapped out, so only Step 3's own line runs."""
    block = TransformerBlock(d_model=4, n_heads=2, d_ff=8)
    block.ffn = torch.nn.Identity()
    return block(torch.zeros(1, 2, 4))


# One tiny call per step. Each raises NotImplementedError while that step's
# blank is unfinished (and only then), and runs in a fraction of a second.
_PROBES = {
    "1": lambda: EmbeddingLayer(vocab_size=4, d_model=2, max_pos=4)(torch.zeros(1, 2, dtype=torch.long)),
    "2": lambda: FeedForward(d_model=2, d_ff=4)(torch.zeros(1, 2, 2)),
    "3": _probe_block,
    "4": lambda: tiny_gpt2(GPT2Model)[0](torch.zeros(1, 2, dtype=torch.long)),
    "5": lambda: greedy_decode(CountingModel(), NumberTokenizer(), "1", max_new=1),
    "6": lambda: sample_with_temperature_topk(FixedModel([0.0, 1.0]), NumberTokenizer(), "0",
                                              max_new=1, top_k=2),
}


def _require(steps: str, purpose: str) -> None:
    """Stop with a pointed message if one of the given earlier steps is unfinished.

    Without this, running GPT-2 before (say) Step 2 is done would report
    Step 2's TODO under the wrong step's header.
    """
    for step in steps:
        try:
            with torch.no_grad():
                _PROBES[step]()
        except NotImplementedError:
            raise NotImplementedError(f"needs Step {step} ({STEP_NAMES[step]}) to {purpose}") from None


def _tokenizer(results: dict):
    """Load the GPT-2 tokenizer once and share it between steps."""
    if "tokenizer" not in results:
        results["tokenizer"] = load_tokenizer()
    return results["tokenizer"]


def _gpt2(results: dict):
    """Build your GPT2Model, load the real weights into it once, and share it.

    Also keeps Hugging Face's own copy of GPT-2 as a reference to test against.
    """
    if "model" not in results:
        model = GPT2Model(vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layers=N_LAYERS,
                          n_heads=N_HEADS, d_ff=D_FF, max_pos=MAX_POS)
        results["hf_model"] = load_gpt2_weights(model)
        results["model"] = model.eval()
    return results["model"], _tokenizer(results)


# ---------------------------------------------------------------------------
# The steps
# ---------------------------------------------------------------------------


def step_1(results: dict) -> str:
    """Tokenize the prompt and embed it with a fresh (untrained) embedding layer."""

    def show():
        tokenizer = _tokenizer(results)
        ids = tokenizer.encode(PROMPT, return_tensors="pt")  # shape (1, 5)
        embed = EmbeddingLayer(vocab_size=VOCAB_SIZE, d_model=D_MODEL, max_pos=MAX_POS)
        with torch.no_grad():
            out = embed(ids)
        pieces = [tokenizer.decode([i]) for i in ids[0].tolist()]  # the text of each token
        print(f'Prompt: "{PROMPT}"')
        print(f"Tokens: {pieces}")
        print(f"Token IDs: {ids[0].tolist()}")
        print(f"Embeddings: {tuple(out.shape)}, one {D_MODEL}-number vector per token")

    return run_step("Step 1: EmbeddingLayer.forward()", show, lambda: check_embedding(EmbeddingLayer))


def step_2() -> str:
    """Run a full-size GPT-2 feed-forward network on 5 random token vectors."""

    def show():
        ffn = FeedForward(d_model=D_MODEL, d_ff=D_FF).eval()
        x = torch.randn(1, 5, D_MODEL)
        with torch.no_grad():
            out = ffn(x)
        print(f"Input {tuple(x.shape)} -> output {tuple(out.shape)}, widening to d_ff={D_FF} in between")
        print(f"Parameters: {count_parameters(ffn):,}")

    return run_step("Step 2: FeedForward.forward()", show, lambda: check_feed_forward(FeedForward))


def step_3() -> str:
    """Run a full-size GPT-2 block (attention + your FFN) on 5 random token vectors."""

    def show():
        _PROBES["3"]()  # your block on a tiny input first, so an unfinished Step 3 reports its own TODO
        _require("2", "run a full-size block")  # the block calls your FFN
        block = TransformerBlock(d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF).eval()
        x = torch.randn(1, 5, D_MODEL)
        with torch.no_grad():
            out = block(x)
        print(f"Input {tuple(x.shape)} -> output {tuple(out.shape)}")
        print(f"Parameters: {count_parameters(block):,} per block, "
              f"{count_parameters(block.ffn):,} of them in the FFN")

    return run_step("Step 3: TransformerBlock.forward()", show,
                    lambda: check_transformer_block(TransformerBlock))


def step_4(results: dict) -> str:
    """Load the real GPT-2 weights into your model and score the prompt."""

    def show():
        _PROBES["4"]()  # your wiring on a tiny model first, so an unfinished Step 4 reports its own TODO
        _require("123", "run the real GPT-2")
        model, tokenizer = _gpt2(results)
        ids = tokenizer.encode(PROMPT, return_tensors="pt")
        with torch.no_grad():
            logits = model(ids)
            reference = results["hf_model"](ids).logits  # Hugging Face's GPT-2, same weights
        same_shape = logits.shape == reference.shape
        results["hf_logit_diff"] = float((logits - reference).abs().max()) if same_shape else float("inf")

        # GPT-2 small is usually quoted at 124M because it reuses the token
        # embedding as the LM head; this model keeps a separate lm_head
        total = count_parameters(model)
        untied = total - model.lm_head.weight.numel()
        print(f"Parameters: {total:,} ({untied:,} if lm_head shared the token embedding)")
        print(f"Logits: {tuple(logits.shape)}, one score per vocabulary token at each position")

        # The last position's logits are the model's guess for the next token
        probs = torch.softmax(logits[0, -1], dim=-1)
        top = torch.topk(probs, k=15)
        tokens = [tokenizer.decode([i]) for i in top.indices.tolist()]
        best = ", ".join(f"{t!r} {p:.1%}" for t, p in zip(tokens[:5], top.values.tolist()[:5]))
        print(f"Most likely next tokens: {best}")
        plot_token_probs(tokens, top.values.numpy(), title=f"Next-token probabilities after: '{PROMPT}'",
                         filepath=str(OUTPUT_DIR / "token_probs.png"))

    return run_step("Step 4: GPT2Model.forward()", show,
                    lambda: check_gpt2_forward(GPT2Model) + check_matches_hugging_face(results))


class _LogitsOnly(torch.nn.Module):
    """Wrap Hugging Face's GPT-2 so that calling it returns plain logits, like yours."""

    def __init__(self, hf_model: torch.nn.Module) -> None:
        super().__init__()
        self.hf_model = hf_model

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.hf_model(token_ids).logits


def step_5(results: dict) -> str:
    """Greedy decoding with your GPT-2."""

    def show():
        _PROBES["5"]()  # a fake model first, so an unfinished Step 5 reports its own TODO
        _require("1234", "generate text with your GPT-2")
        model, tokenizer = _gpt2(results)
        text = greedy_decode(model, tokenizer, PROMPT, max_new=10)
        print(f'Prompt:  "{PROMPT}"')
        print(f'Greedy:  "{text}"')

        # For the tests: your loop driving Hugging Face's GPT-2, and Hugging Face's own greedy search
        hf_model = results["hf_model"]
        results["greedy_on_hf"] = greedy_decode(_LogitsOnly(hf_model), tokenizer, PROMPT, max_new=10)
        ids = tokenizer.encode(PROMPT, return_tensors="pt")
        out = hf_model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=10,
                                do_sample=False, pad_token_id=tokenizer.eos_token_id)
        results["hf_greedy"] = tokenizer.decode(out[0])

    return run_step("Step 5: greedy_decode()", show,
                    lambda: check_greedy_decode(greedy_decode) + check_greedy_on_gpt2(results))


def step_6(results: dict) -> str:
    """Temperature + top-k sampling with your GPT-2, at a low and a high temperature."""

    def show():
        _PROBES["6"]()  # a fake model first, so an unfinished Step 6 reports its own TODO
        _require("1234", "generate text with your GPT-2")
        model, tokenizer = _gpt2(results)
        print(f'Prompt:  "{PROMPT}"')
        for temperature in (0.8, 1.4):
            torch.manual_seed(0)  # the same random draws on every run, so the output repeats
            text = sample_with_temperature_topk(model, tokenizer, PROMPT, max_new=10,
                                                temperature=temperature, top_k=40)
            print(f'T={temperature}, top_k=40:  "{text}"')

    return run_step("Step 6: sample_with_temperature_topk()", show,
                    lambda: check_temperature(sample_with_temperature_topk))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

STEP_CHOICES = ["1", "2", "3", "4", "5", "6"]


def main():
    parser = argparse.ArgumentParser(description="Assemble GPT-2 and generate text")
    parser.add_argument("--step", choices=[*STEP_CHOICES, "all"], default="all",
                        help="Which step to run (default: all)")
    args = parser.parse_args()
    steps = STEP_CHOICES if args.step == "all" else [args.step]

    OUTPUT_DIR.mkdir(exist_ok=True)
    results: dict = {}  # the tokenizer, GPT-2 and test inputs, shared between steps

    for step in steps:
        if step == "1":
            step_1(results)
        elif step == "2":
            step_2()
        elif step == "3":
            step_3()
        elif step == "4":
            step_4(results)
        elif step == "5":
            step_5(results)
        elif step == "6":
            step_6(results)


if __name__ == "__main__":
    main()
