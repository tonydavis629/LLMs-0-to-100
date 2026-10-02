"""
Module 6 Exercise runner: Finetuning NanoGPT into an instruct model

Run with:
    uv run python module_06_finetuning/src/main.py

Loads the bundled Module 5 base checkpoint, injects LoRA adapters, finetunes on
toy instruction-response pairs, and shows the behavioral flip on one prompt:
the base model continues text and ignores the instruction; the finetuned model
answers it. Read top to bottom, the steps follow the finetuning recipe:

    1. wrap a prompt and response in the chat template
    2. mask the prompt so only the response is trained
    3. score the response tokens with a masked loss
    4. build the optimizer
    5. the LoRA adapter's forward pass
    6. freeze the base model so only the adapters train
    7. the training step, then the finetuning loop
    8. count how few parameters LoRA trains
    9. prompt the model for an answer, before and after finetuning
   10. merge the adapters back into plain weight matrices

Add --step N to run one step (1-10).
Add --solution to run the finished answers from solution/exercise.py.
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path

import torch

# Make the module root (parent of src/) importable so we can `from exercise import ...`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# With --solution, swap in solution/exercise.py before anything imports `exercise`
from src.solution import use_solution_if_requested

use_solution_if_requested()

from exercise import (
    build_generation_prompt,
    build_optimizer,
    build_targets,
    count_trainable_params,
    format_example,
    freeze_base_param,
    lora_forward_delta,
    masked_cross_entropy,
    merge_lora_weight,
    sft_train_step,
)
from src.data import load_dataset, pad, prompt_span
from src.model import LoRALinear, freeze_base_, generate, inject_lora, load_base_model, merge_lora
from src.prerequisites import require
from src.reporting import run_step
from src.tokenizer import SPECIAL_TOKENS, build_vocab, decode, encode

# One test file per step lives in tests/
from tests.test_step1_format_example import check_format_example
from tests.test_step2_build_targets import check_build_targets
from tests.test_step3_masked_loss import check_masked_cross_entropy
from tests.test_step4_optimizer import check_build_optimizer
from tests.test_step5_lora_delta import check_lora_forward_delta
from tests.test_step6_freeze import check_freeze_base_param, check_freeze_on_model
from tests.test_step7_train_step import check_sft_train_step, check_sft_training
from tests.test_step8_count_params import check_count_on_model, check_count_trainable_params
from tests.test_step9_generation_prompt import check_build_generation_prompt
from tests.test_step10_merge import check_merge_lora_weight, check_merge_on_model

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ---------------------------------------------------------------------------
# Hyperparameters (small enough to finetune on a laptop CPU in seconds)
# ---------------------------------------------------------------------------
BLOCK_SIZE = 128       # the base model's context length in characters
PAD_LEN = 36           # pad each pair to 36 tokens (the longest is 33), not the full 128
BATCH_SIZE = 16        # examples per batch
MAX_STEPS = 150        # total finetuning steps (the loss levels off near 0.3 by then)
EVAL_INTERVAL = 25     # report the loss every this many steps
LR = 1e-3              # small finetuning learning rate (step 4)
GRAD_CLIP = 1.0        # largest gradient norm allowed in one update
RANK = 8               # LoRA rank r
ALPHA = 32.0           # LoRA alpha (scale = alpha / r)
DROPOUT = 0.0          # no dropout inside the adapters
SEED = 1337            # fixed seed, so every run gives the same numbers

NEW_VOCAB_SIZE = 65 + len(SPECIAL_TOKENS)  # 65 base chars + 4 specials = 69
SAMPLE_PROMPT = "uppercase: hello"          # the one prompt we flip before/after
SAMPLE_TOKENS = 18                          # tokens to generate per sample
SAMPLE_TEMPERATURE = 0.4                    # low temp: the toy tasks are deterministic
DEMO_PAIR = 2                               # the dataset pair Steps 1 and 2 print

# ---------------------------------------------------------------------------
# Setup: the base model, the vocabulary, and the dataset, shared by every step
# ---------------------------------------------------------------------------
torch.manual_seed(SEED)

# data/base_model.pt ships with the repo. It is the frozen Module 5 base model:
# TinyGPT pretrained on tinyshakespeare by src/make_base_checkpoint.py.
model, base_stoi, _ = load_base_model(DATA_DIR / "base_model.pt", NEW_VOCAB_SIZE)
stoi, itos = build_vocab(base_stoi)  # the 65 base characters plus 4 special tokens

# An untouched copy of the base model, for the "before finetuning" sample
base = copy.deepcopy(model)

# Wrap the attention and MLP layers of `model` with LoRA adapters (provided).
# Every B starts at zero, so the wrapped model still behaves exactly like the base.
inject_lora(model, r=RANK, alpha=ALPHA, dropout=DROPOUT)

dataset = load_dataset(DATA_DIR / "sft_pairs.jsonl")    # list of {"prompt": ..., "response": ...}
special = {tok: stoi[tok] for tok in SPECIAL_TOKENS}    # e.g. "<|user|>" -> 65


def enc(text: str) -> list[int]:
    """Text -> token ids."""
    return encode(text, stoi)


def dec(ids) -> str:
    """Token ids -> text."""
    return decode(ids, itos)


# Step 7 fills this with the loss at every report, so later steps can tell
# whether the adapters were trained in this run
training_losses: list[float] = []


def build_batch(generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one padded (x, y) batch from random prompt-response pairs (uses Steps 1 and 2).

    Targets are built from the unpadded ids so pad tokens never become targets;
    both x and y are then padded to PAD_LEN with -100 padding on the targets.
    """
    idxs = torch.randperm(len(dataset), generator=generator)[:BATCH_SIZE].tolist()
    xs, ys = [], []
    for i in idxs:
        ids = format_example(dataset[i]["prompt"], dataset[i]["response"], special, enc)  # Step 1
        targets = build_targets(ids, prompt_span(dataset[i]["prompt"], enc))              # Step 2
        xs.append(pad(ids, PAD_LEN, special["<|pad|>"]))
        ys.append(pad(targets, PAD_LEN, -100))
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def sample(some_model) -> tuple[str, str]:
    """Generate a completion for SAMPLE_PROMPT (uses Step 9). Returns (full_text, response_only)."""
    ids = build_generation_prompt(SAMPLE_PROMPT, special, enc)  # Step 9
    idx = torch.tensor([ids], dtype=torch.long)
    gen = torch.Generator().manual_seed(SEED)
    out = generate(some_model, idx, SAMPLE_TOKENS, BLOCK_SIZE, temperature=SAMPLE_TEMPERATURE, generator=gen)
    full = dec(out[0])
    # The response is what follows the assistant marker, up to the first <|end|>.
    after = full.split("<|assistant|>", 1)[-1]
    response = after.split("<|end|>", 1)[0]
    return full, response


# ---------------------------------------------------------------------------
# Step 1: wrap a prompt and response in the chat template
# ---------------------------------------------------------------------------


def step_1() -> str:
    def show():
        pair = dataset[DEMO_PAIR]
        ids = format_example(pair["prompt"], pair["response"], special, enc)
        print(f"{len(dataset)} instruction pairs loaded from sft_pairs.jsonl")
        print(f"Example pair: {pair['prompt']!r} -> {pair['response']!r}")
        print(f"  {len(ids)} token ids: {ids}")
        if all(isinstance(i, int) for i in ids):  # a nested list cannot be decoded
            print(f"  decoded: {dec(ids)!r}")

    return run_step("Step 1: format_example()", show,
                    lambda: check_format_example(format_example, special, enc, dec))


# ---------------------------------------------------------------------------
# Step 2: mask the prompt so only the response is trained
# ---------------------------------------------------------------------------


def step_2() -> str:
    def show():
        require("2", needs={"1": "format the pair it masks"})
        pair = dataset[DEMO_PAIR]
        ids = format_example(pair["prompt"], pair["response"], special, enc)
        span = prompt_span(pair["prompt"], enc)
        targets = build_targets(ids, span)
        trained = [t for t in targets if t != -100]
        print(f"Example pair: {pair['prompt']!r} -> {pair['response']!r}")
        print(f"  {len(ids)} tokens, the first {span} are the prompt (prompt_span = {span})")
        print(f"  masked to -100: {len(targets) - len(trained)} positions")
        print(f"  trained:        {len(trained)} positions, targets {dec(trained)!r}")

    return run_step("Step 2: build_targets()", show, lambda: check_build_targets(build_targets))


# ---------------------------------------------------------------------------
# Step 3: score the response tokens with a masked loss
# ---------------------------------------------------------------------------


def step_3() -> str:
    def show():
        require("3", needs={"1": "build a batch of formatted pairs",
                            "2": "build the batch's masked targets"})
        # Same seed as the training loop, so this is the batch that training scores at step 0
        xb, yb = build_batch(torch.Generator().manual_seed(SEED))
        with torch.no_grad():
            loss = masked_cross_entropy(base(xb), yb)
        n_trained = int((yb != -100).sum())
        uniform = math.log(NEW_VOCAB_SIZE)  # loss of a model that guesses every token equally
        print(f"Base model on one batch of {BATCH_SIZE} pairs ({n_trained} response tokens scored)")
        print(f"  masked loss:    {float(loss):.4f}")
        print(f"  uniform guess:  {uniform:.4f} (ln {NEW_VOCAB_SIZE})")

    return run_step("Step 3: masked_cross_entropy()", show,
                    lambda: check_masked_cross_entropy(masked_cross_entropy))


# ---------------------------------------------------------------------------
# Step 4: build the optimizer (used for real in Step 7, so no demo here)
# ---------------------------------------------------------------------------


def step_4() -> str:
    return run_step("Step 4: build_optimizer()", lambda: None, lambda: check_build_optimizer(build_optimizer))


# ---------------------------------------------------------------------------
# Step 5: the LoRA adapter's forward pass
# ---------------------------------------------------------------------------


def step_5() -> str:
    def show():
        layers = [m for m in model.modules() if isinstance(m, LoRALinear)]
        idx = torch.tensor([enc(SAMPLE_PROMPT)], dtype=torch.long)
        with torch.no_grad():
            diff = (model(idx) - base(idx)).abs().max().item()  # each LoRALinear calls your function
        print(f"LoRA (r={RANK}, alpha={ALPHA:g}, scale={ALPHA / RANK:g}) wraps {len(layers)} linear layers")
        print(f"Every B starts at zero, so on {SAMPLE_PROMPT!r} the wrapped model matches the base model:")
        print(f"  max logit difference {diff:.2e}")

    return run_step("Step 5: lora_forward_delta()", show,
                    lambda: check_lora_forward_delta(lora_forward_delta))


# ---------------------------------------------------------------------------
# Step 6: freeze the base model so only the adapters train
# ---------------------------------------------------------------------------


def step_6() -> str:
    def show():
        freeze_base_(model)  # calls your freeze_base_param() on every base tensor
        # Split the tensors into LoRA adapters (named .A and .B) and everything else
        lora = [p for name, p in model.named_parameters() if name.endswith((".A", ".B"))]
        frozen = [p for name, p in model.named_parameters() if not name.endswith((".A", ".B"))]
        print(f"Base tensors frozen:     {sum(not p.requires_grad for p in frozen)} of {len(frozen)} "
              "(embeddings, layer norms, and every wrapped Linear)")
        print(f"LoRA tensors trainable:  {sum(p.requires_grad for p in lora)} of {len(lora)} "
              f"(A and B in each of the {len(lora) // 2} LoRA layers)")

    return run_step("Step 6: freeze_base_param()", show,
                    lambda: check_freeze_base_param(freeze_base_param) + check_freeze_on_model(model))


# ---------------------------------------------------------------------------
# Step 7: the training step, then the finetuning loop (the only slow step)
# ---------------------------------------------------------------------------


def step_7() -> str:
    own_checks = []  # this step's own tests, run before training (see below)

    def show():
        # Run this step's own tests first, so an unfinished sft_train_step()
        # reports its own TODO rather than one from an earlier step
        own_checks.extend(check_sft_train_step(sft_train_step))
        require(needs={"1": "build training batches",
                       "2": "build the masked training targets",
                       "3": "compute the training loss",
                       "4": "build the optimizer",
                       "5": "run the LoRA layers",
                       "6": "freeze the base model before training"})

        freeze_base_(model)                       # no-op if Step 6 already froze it
        optimizer = build_optimizer(model, LR)    # AdamW over the adapters only
        batch_gen = torch.Generator().manual_seed(SEED)
        print(f"{'step':>6}  {'loss':>8}")
        for step in range(MAX_STEPS + 1):
            if step % EVAL_INTERVAL == 0:
                # Report the loss on a fresh batch, without training on it
                with torch.no_grad():
                    xb, yb = build_batch(batch_gen)
                    loss = float(masked_cross_entropy(model(xb), yb))
                training_losses.append(loss)
                print(f"{step:>6}  {loss:>8.4f}")
            if step == MAX_STEPS:
                break
            xb, yb = build_batch(batch_gen)
            sft_train_step(model, optimizer, xb, yb, GRAD_CLIP)  # one update of the adapters

    return run_step("Step 7: sft_train_step()", show,
                    lambda: own_checks + check_sft_training(training_losses))


# ---------------------------------------------------------------------------
# Step 8: count how few parameters LoRA trains
# ---------------------------------------------------------------------------


def step_8() -> str:
    def show():
        require("8", needs={"6": "freeze the base before counting what still trains"})
        freeze_base_(model)  # no-op if Step 6 already froze it
        trainable = count_trainable_params(model)
        total = model.num_params()
        print(f"Trainable (LoRA adapters):  {trainable:,}")
        print(f"Total (base + adapters):    {total:,}")
        print(f"Fraction trainable:         {trainable / total:.2%}")

    return run_step("Step 8: count_trainable_params()", show,
                    lambda: check_count_trainable_params(count_trainable_params)
                    + check_count_on_model(count_trainable_params, model))


# ---------------------------------------------------------------------------
# Step 9: prompt the model for an answer, before and after finetuning
# ---------------------------------------------------------------------------


def step_9() -> str:
    def show():
        # The base model has no adapters, so it only needs this step's prompt builder
        full, response = sample(base)
        print(f"prompt: {SAMPLE_PROMPT!r}")
        print("Base model (before finetuning):")
        print(f"  full:     {full!r}")
        print(f"  response: {response!r}   <- ignores the instruction")

        print("Finetuned model (after Step 7):")
        if not training_losses:
            print("  not finetuned in this run (Step 7 has not trained the adapters)")
            return
        full, response = sample(model)
        print(f"  full:     {full!r}")
        print(f"  response: {response!r}   <- answers the instruction")

    return run_step("Step 9: build_generation_prompt()", show,
                    lambda: check_build_generation_prompt(build_generation_prompt, special, enc, dec))


# ---------------------------------------------------------------------------
# Step 10: merge the adapters back into plain weight matrices
# ---------------------------------------------------------------------------


def step_10() -> str:
    idx = torch.tensor([enc(SAMPLE_PROMPT)], dtype=torch.long)

    def show():
        require("10", needs={"5": "run the adapter model the merge is compared against"})
        # Merge a deep copy so the adapter model is left intact for the comparison
        merged = merge_lora(copy.deepcopy(model))  # calls your merge_lora_weight() on every layer
        n_layers = sum(isinstance(m, LoRALinear) for m in model.modules())
        with torch.no_grad():
            diff = (model(idx) - merged(idx)).abs().max().item()
        print(f"Merged {n_layers} LoRA layers into plain nn.Linear weights: "
              f"{model.num_params():,} -> {merged.num_params():,} parameters")
        print(f"  Max logit difference (adapter vs merged): {diff:.2e}")
        if not training_losses:
            print("  (Step 7 has not trained the adapters in this run, so every B is still zero)")

    return run_step("Step 10: merge_lora_weight()", show,
                    lambda: check_merge_lora_weight(merge_lora_weight)
                    + check_merge_on_model(model, merge_lora(copy.deepcopy(model)), idx))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

STEPS = {"1": step_1, "2": step_2, "3": step_3, "4": step_4, "5": step_5,
         "6": step_6, "7": step_7, "8": step_8, "9": step_9, "10": step_10}


def main():
    parser = argparse.ArgumentParser(description="Finetuning NanoGPT into an instruct model")
    parser.add_argument("--step", choices=[*STEPS, "all"], default="all",
                        help="Which step to run (default: all)")
    args = parser.parse_args()

    for number, step in STEPS.items():
        if args.step in ("all", number):
            step()


if __name__ == "__main__":
    main()
