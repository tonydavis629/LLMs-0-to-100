"""
Module 6 Exercise runner: Finetuning NanoGPT into an instruct model

Run with:
    uv run python module_06_finetuning/src/main.py

Loads the bundled Module 5 base checkpoint, injects LoRA adapters, finetunes on
toy instruction-response pairs, and prints the behavioral flip on one prompt:
the base model continues text and ignores the instruction; the finetuned model
answers it. Also reports the trainable-vs-total parameter savings and verifies
that merging the LoRA update back into the weights changes nothing.

Any step in exercise.py that still raises NotImplementedError is detected and
skipped, so you can implement one step at a time and re-run immediately.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

# Make the module root (parent of src/) importable so we can `from exercise import ...`,
# and src/ importable so we can grab the provided model / tokenizer / data helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exercise import (  # noqa: E402  (import after sys.path edits)
    format_example,
    build_targets,
    masked_cross_entropy,
    build_optimizer,
    lora_forward_delta,
    freeze_base_param,
    sft_train_step,
    count_trainable_params,
    build_generation_prompt,
    merge_lora_weight,
)
from model import (  # noqa: E402
    GPTConfig,
    TinyGPT,
    inject_lora,
    freeze_base_,
    merge_lora,
    load_base_model,
    generate,
)
from tokenizer import build_vocab, encode, decode, SPECIAL_TOKENS  # noqa: E402
from data import load_dataset  # noqa: E402


# ---------------------------------------------------------------------------
# Hyperparameters (small enough to finetune on a laptop CPU in a couple minutes)
# ---------------------------------------------------------------------------
BLOCK_SIZE = 128       # context length in characters
BATCH_SIZE = 16        # examples per batch
MAX_STEPS = 1000       # total finetuning steps
EVAL_INTERVAL = 100    # report the loss every this many steps
LR = 1e-3              # small finetuning learning rate (step 4)
GRAD_CLIP = 1.0
RANK = 8               # LoRA rank r
ALPHA = 32.0           # LoRA alpha (scale = alpha / r)
DROPOUT = 0.0
SEED = 1337

NEW_VOCAB_SIZE = 65 + len(SPECIAL_TOKENS)  # 65 base chars + 4 specials = 69
SAMPLE_PROMPT = "uppercase: hello"          # the one prompt we flip before/after
SAMPLE_TOKENS = 18                          # tokens to generate per sample
SAMPLE_TEMPERATURE = 0.4                    # low temp: the toy tasks are deterministic

_THIS_DIR = Path(__file__).resolve().parent


def _find_data_file(name: str) -> Path:
    """Walk up from this file to find data/<name> (works from src/ or solution/src/)."""
    for parent in _THIS_DIR.parents:
        candidate = parent / "data" / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate data/{name}")


def _is_implemented(fn, *args, **kwargs) -> bool:
    """Return True if fn runs without raising NotImplementedError (other errors aside)."""
    try:
        fn(*args, **kwargs)
        return True
    except NotImplementedError:
        return False
    except Exception:
        # Any other exception still means the student wrote *something*; treat as implemented.
        return True


def _probe_steps(special: dict[str, int], enc) -> dict[str, bool]:
    """Detect which exercise.py steps are implemented, using throwaway inputs.

    A tiny scratch model is used for anything that runs a forward/backward pass,
    so probing never disturbs the real model we are about to finetune.
    """
    import torch.nn as nn

    logits = torch.randn(2, 5, NEW_VOCAB_SIZE)
    targets = torch.tensor([[-100, 1, 2, -100, 3], [1, 2, 3, 4, 5]])
    ids = [special["<|user|>"], 1, 2, special["<|end|>"], special["<|assistant|>"], 3, special["<|end|>"]]

    x = torch.randn(1, 3, 8)
    A = nn.Parameter(torch.zeros(2, 8))
    B = nn.Parameter(torch.zeros(8, 2))
    Wm, Am, Bm = torch.zeros(8, 8), torch.zeros(2, 8), torch.zeros(8, 2)

    scratch = TinyGPT(GPTConfig(vocab_size=8, block_size=8, n_layer=1, n_head=2, n_embd=16))
    scratch_opt = torch.optim.AdamW(scratch.parameters(), lr=1e-3)
    xb = torch.zeros(1, 4, dtype=torch.long)
    yb = torch.tensor([[-100, 2, 3, 4]])

    return {
        "format_example": _is_implemented(format_example, "hi", "HO", special, enc),
        "build_targets": _is_implemented(build_targets, ids, 4),
        "masked_cross_entropy": _is_implemented(masked_cross_entropy, logits, targets),
        "build_optimizer": _is_implemented(build_optimizer, scratch, LR),
        "lora_forward_delta": _is_implemented(lora_forward_delta, x, A, B, 1.0, nn.Identity()),
        "freeze_base_param": _is_implemented(freeze_base_param, nn.Parameter(torch.zeros(3))),
        "sft_train_step": _is_implemented(sft_train_step, scratch, scratch_opt, xb, yb),
        "count_trainable": _is_implemented(count_trainable_params, scratch),
        "build_generation_prompt": _is_implemented(build_generation_prompt, "hi", special, enc),
        "merge_lora_weight": _is_implemented(merge_lora_weight, Wm, Am, Bm, 1.0),
    }


def _heading(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)


def _pad(seq: list[int], length: int, pad_id: int) -> list[int]:
    """Pad (or truncate) a sequence to exactly `length`."""
    if len(seq) >= length:
        return seq[:length]
    return seq + [pad_id] * (length - len(seq))


def _build_batch(pairs, special, enc, generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one padded (x, y) batch from random prompt-response pairs.

    Targets are built from the unpadded ids so pad tokens never become targets;
    both x and y are then padded to BLOCK_SIZE with -100 padding on the targets.
    """
    idxs = torch.randperm(len(pairs), generator=generator)[:BATCH_SIZE].tolist()
    xs, ys = [], []
    for i in idxs:
        ids = format_example(pairs[i]["prompt"], pairs[i]["response"], special, enc)
        # prompt span = user marker + prompt text + end marker + assistant marker
        prompt_span = 1 + len(enc(pairs[i]["prompt"])) + 1 + 1
        targets = build_targets(ids, prompt_span)
        xs.append(_pad(ids, BLOCK_SIZE, special["<|pad|>"]))
        ys.append(_pad(targets, BLOCK_SIZE, -100))
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def _sample(model, prompt: str, special, enc, itos, generator) -> tuple[str, str]:
    """Generate a completion for `prompt`. Returns (full_text, response_only)."""
    ids = build_generation_prompt(prompt, special, enc)
    idx = torch.tensor([ids], dtype=torch.long)
    out = generate(model, idx, SAMPLE_TOKENS, BLOCK_SIZE,
                   temperature=SAMPLE_TEMPERATURE, generator=generator)
    full = decode(out[0], itos)
    # The response is what follows the assistant marker, up to the first <|end|>.
    after = full.split("<|assistant|>", 1)[-1]
    response = after.split("<|end|>", 1)[0]
    return full, response


def main() -> None:
    torch.manual_seed(SEED)

    # data/base_model.pt ships with the repo. It is the frozen Module 5 base model:
    # TinyGPT pretrained on tinyshakespeare by solution/src/make_base_checkpoint.py.
    base_ckpt = _find_data_file("base_model.pt")
    dataset = load_dataset(_find_data_file("sft_pairs.jsonl"))

    # Load the frozen base model and expand its vocabulary for the special tokens.
    model, base_stoi, _ = load_base_model(base_ckpt, NEW_VOCAB_SIZE)
    stoi, itos = build_vocab(base_stoi)
    special = {tok: stoi[tok] for tok in SPECIAL_TOKENS}
    enc = lambda s: encode(s, stoi)  # noqa: E731

    steps = _probe_steps(special, enc)

    _heading("MODULE 6: Finetuning NanoGPT")
    print(f"TinyGPT: {model.cfg.n_layer} layers, {model.cfg.n_head} heads, width {model.cfg.n_embd}")
    print(f"Base parameters: {model.num_params():,}   Vocabulary: {NEW_VOCAB_SIZE} "
          f"({len(SPECIAL_TOKENS)} special tokens)")
    print(f"Instruction pairs loaded: {len(dataset):,}")
    print()

    # ------------------------------------------------------------------
    # BEFORE: sample the base model on the instruction (needs build_generation_prompt).
    # This runs on the plain model, before any adapter is injected.
    # ------------------------------------------------------------------
    _heading("SAMPLE BEFORE FINETUNING (base model)")
    if steps["build_generation_prompt"]:
        gen = torch.Generator().manual_seed(SEED)
        full, resp = _sample(model, SAMPLE_PROMPT, special, enc, itos, gen)
        print(f"  prompt:   {SAMPLE_PROMPT!r}")
        print(f"  full:     {full!r}")
        print(f"  response: {resp!r}   <- base model ignores the instruction")
    else:
        print("  [skipped: implement build_generation_prompt()]")
    print()

    # ------------------------------------------------------------------
    # Inject LoRA adapters (provided), then freeze the base weights (step 6).
    # ------------------------------------------------------------------
    inject_lora(model, r=RANK, alpha=ALPHA, dropout=DROPOUT)

    _heading("PARAMETER COUNTS (LoRA)")
    if steps["freeze_base_param"]:
        freeze_base_(model)
    else:
        print("  [base not frozen yet: implement freeze_base_param()]")
    if steps["count_trainable"]:
        trainable = count_trainable_params(model)
        total = model.num_params()
        print(f"  Trainable (LoRA adapters):  {trainable:,}")
        print(f"  Total (base + adapters):    {total:,}")
        print(f"  Fraction trainable:         {trainable / total:.2%}")
    else:
        print("  [skipped: implement count_trainable_params()]")
    print()

    # ------------------------------------------------------------------
    # TRAINING: needs the data path (1-2), the loss (3), the optimizer (4),
    # the LoRA delta (5), the freeze (6), and the step (7).
    # ------------------------------------------------------------------
    core = ["format_example", "build_targets", "masked_cross_entropy", "build_optimizer",
            "lora_forward_delta", "freeze_base_param", "sft_train_step"]
    missing = [s for s in core if not steps[s]]

    _heading("FINETUNING")
    if missing:
        print(f"  [skipped: implement {', '.join(missing)} to finetune]")
        print()
    else:
        optimizer = build_optimizer(model, LR)
        batch_gen = torch.Generator().manual_seed(SEED)
        print(f"{'step':>6}  {'loss':>8}")
        for step in range(MAX_STEPS + 1):
            if step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
                model.eval()
                with torch.no_grad():
                    xb, yb = _build_batch(dataset, special, enc, batch_gen)
                    loss = masked_cross_entropy(model(xb), yb)
                model.train()
                print(f"{step:>6}  {loss.item():>8.4f}")
            if step == MAX_STEPS:
                break
            xb, yb = _build_batch(dataset, special, enc, batch_gen)
            sft_train_step(model, optimizer, xb, yb, GRAD_CLIP)
        print()

    # ------------------------------------------------------------------
    # AFTER: same prompt, finetuned model (needs the same prompt builder + a run).
    # ------------------------------------------------------------------
    _heading("SAMPLE AFTER FINETUNING (instruct model)")
    if not missing and steps["build_generation_prompt"]:
        gen = torch.Generator().manual_seed(SEED)
        full, resp = _sample(model, SAMPLE_PROMPT, special, enc, itos, gen)
        print(f"  prompt:   {SAMPLE_PROMPT!r}")
        print(f"  full:     {full!r}")
        print(f"  response: {resp!r}   <- finetuned model answers the instruction")
    else:
        print("  [skipped: finetuning did not run or build_generation_prompt missing]")
    print()

    # ------------------------------------------------------------------
    # MERGE EQUALITY: merging B A back into W must not change the output (step 10).
    # Compare on a deep copy so the adapter model is left intact.
    # ------------------------------------------------------------------
    _heading("MERGE-EQUALITY CHECK")
    if not missing and steps["merge_lora_weight"]:
        merged = merge_lora(copy.deepcopy(model))
        ids = build_generation_prompt(SAMPLE_PROMPT, special, enc)
        idx = torch.tensor([ids], dtype=torch.long)
        model.eval()
        merged.eval()
        with torch.no_grad():
            max_diff = (model(idx) - merged(idx)).abs().max().item()
        print(f"  Max logit difference (adapter vs merged): {max_diff:.2e}")
        print("  Merged model matches the adapter model: "
              + ("PASS" if max_diff < 1e-4 else "FAIL"))
    else:
        print("  [skipped: implement merge_lora_weight() and finetune first]")
    print()

    _heading("Done")
    print("Run after each step; unfinished steps are skipped automatically.")


if __name__ == "__main__":
    main()
