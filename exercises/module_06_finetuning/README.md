# Module 6: Finetuning NanoGPT into an Instruct Model

## Overview

Take the tiny base model from Module 5 and **finetune** it into a small instruct
model on prompt-response pairs. The model, the LoRA plumbing, the tokenizer, and
the dataset are all provided &mdash; your job is the **finetuning loop**: format a
chat template, mask the loss to the response, compute masked cross-entropy,
implement a LoRA adapter, freeze the base, run a supervised-finetuning step, count
the trainable parameters, and merge the adapter back in.

The goal is not a useful assistant. The goal is to make finetuning **visible**: the
loss mask, the LoRA adapter, the trainable-parameter savings, and the behavioral
flip from "continues text" to "answers the instruction" on the **same prompt**
before and after. With the bundled base model, `uppercase: hello` flips from a
Shakespeare-style continuation to `HELLO`, training under 8% of the parameters.

## Setup

There is one shared environment for the whole repo. From the repo root:

```bash
uv sync
```

## Running

```bash
uv run python exercises/module_06_finetuning/src/main.py
```

The runner detects which steps you have implemented and skips the rest, so you can
fill in one step at a time and re-run immediately. It prints the base model's
completion of a sample instruction (**before**), the trainable-vs-total parameter
count, the finetuning loss, the finetuned completion of the same instruction
(**after**), and a merge-equality check (the merged model must match the adapter
model).

## What to implement

Open `exercise.py` and fill in each `raise NotImplementedError(...)` line. Each
needs only one expression or one short block.

| Step | Function | What it does |
|------|----------|--------------|
| 1 | `format_example()` | Assemble the chat template: `[user] + prompt + [end] + [assistant] + response + [end]` |
| 2 | `build_targets()` | Next-token targets with the prompt predictions masked to `-100` |
| 3 | `masked_cross_entropy()` | Cross-entropy over response tokens only (`ignore_index=-100`) |
| 4 | `build_optimizer()` | AdamW over only the trainable (adapter) parameters |
| 5 | `lora_forward_delta()` | The low-rank update `scale * (dropout(x) @ A.t() @ B.t())` |
| 6 | `freeze_base_param()` | Freeze one base parameter (`requires_grad = False`) |
| 7 | `sft_train_step()` | Zero gradients, backpropagate (clip + step provided) |
| 8 | `count_trainable_params()` | Count parameters with `requires_grad=True` |
| 9 | `build_generation_prompt()` | The template up to the assistant marker (no response) |
| 10 | `merge_lora_weight()` | The merged weight `base_W + scale * (B @ A)` |

The model (`src/model.py`), tokenizer (`src/tokenizer.py`), dataset builder
(`src/data.py`), and runner (`src/main.py`) are all provided. The LoRA injection,
freezing loop, and merge loop live in `src/model.py` and call back into the three
functions you write (steps 5, 6, 10). You only edit `exercise.py`.

## Data

- `data/base_model.pt` &mdash; the frozen Module 5 base checkpoint (the TinyGPT
  config, weights, and character vocabulary). The runner loads it, expands the
  vocabulary by four special tokens, and finetunes on top. It is never
  re-pretrained. (Regenerate it with `solution/src/make_base_checkpoint.py`.)
- `data/sft_pairs.jsonl` &mdash; ~350 toy instruction-response pairs across four
  learnable tasks: uppercase, fixed question-answer, repeat, and reverse. Toy and
  deterministic so a tiny model shows a crisp flip in a few hundred CPU steps.

The tokenizer is the Module 5 65-character vocabulary plus four **atomic** special
tokens (`<|user|>`, `<|assistant|>`, `<|end|>`, `<|pad|>`), each one token id, for
a vocabulary of 69.

## Extra credit

- **Full finetuning vs LoRA.** Skip the freeze step (or unfreeze everything),
  compare the trainable-parameter count and the sample quality against the LoRA run.
- **Vary the rank `r`.** Try `r = 1, 2, 4, 16`. Watch the quality-vs-size trade-off:
  how small can the adapter get before the flip stops working?
- **Catastrophic-forgetting probe.** After finetuning, feed a raw base-style prompt
  (e.g. `To be, or not to be`) and check whether the model still continues text.
- **Loss-mask ablation.** Build targets **without** masking the prompt, retrain, and
  watch the model start hallucinating its own `<|user|>` prompts.
