# Module 6: Finetuning NanoGPT into an Instruct Model

## Overview

Take the tiny base model from Module 5 and **finetune** it into a small instruct
model on prompt-response pairs. The model, the LoRA plumbing, the tokenizer, and
the dataset are all provided &mdash; your job is the **finetuning loop**: format a
chat template, mask the loss to the response, compute masked cross-entropy,
implement a LoRA adapter, run a supervised-finetuning step, count the trainable
parameters, and merge the adapter back in.

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

From the `exercises/` directory:

```bash
uv run python module_06_finetuning/src/main.py
```

The runner goes through the nine steps in order. Each step's header line carries a tag, and the step's output follows: what your code produced on the real model, then one line per test from `tests/`. The tags are:

| Tag | Meaning |
|-----|---------|
| `CORRECT` | every test for the step passed |
| `INCORRECT` | your code ran but a test failed; the expected and actual values are printed under it |
| `INCOMPLETE` | the function still raises `NotImplementedError` |

```
=== Step 7: count_trainable_params() === CORRECT
Frozen base tensors:        52 (embeddings, layer norms, every wrapped Linear)
Trainable LoRA tensors:     32 (A and B in each of the 16 LoRA layers)
Trainable (LoRA adapters):  65,536
Total (base + adapters):    884,096
Fraction trainable:         7.41%
  CORRECT    counts every element: nn.Linear(3, 2) has 2*3 + 2 = 8 trainable numbers
  CORRECT    skips frozen tensors: a rank-2 adapter on a frozen 4 -> 3 layer gives 2*4 + 3*2 = 14
  CORRECT    a fully frozen layer has 0 trainable parameters
  CORRECT    real model: the count is exactly the LoRA A and B matrices, 65,536 numbers
```

Step 6 finetunes the model for 150 steps, which takes under 30 seconds on a laptop CPU. Every other step takes a few seconds. Run a single step with `--step` (1 to 9):

```bash
uv run python module_06_finetuning/src/main.py --step 3
```

Step 8 samples the base model and, if Step 6 finetuned it in the same run, the finetuned model too. Step 9 merges whatever adapters the run has trained. Run on their own, both steps note that nothing has been finetuned yet.

`exercise.py` at the module root is the only file you edit. Everything already written for you lives in `src/`. Run the finished answers with `--solution`:

```bash
uv run python module_06_finetuning/src/main.py --solution
```

## What to implement

Open `exercise.py` and fill in each `raise NotImplementedError(...)` line. Each
needs only one expression or one short block.

| Step | Function | What it does |
|------|----------|--------------|
| 1 | `format_example()` | Assemble the chat template: `[user] + prompt + [end] + [assistant] + response + [end]` |
| 2 | `build_targets()` | Next-token targets with the prompt predictions masked to `-100` |
| 3 | `masked_cross_entropy()` | Cross-entropy over response tokens only (`ignore_index=-100`) |
| 4 | `build_optimizer()` | AdamW over only the trainable (adapter) parameters |
| 5 | `lora_forward_delta()` | The low-rank update `scale * B A x`, written for a batch of row vectors |
| 6 | `sft_train_step()` | Zero gradients, backpropagate (clip + step provided) |
| 7 | `count_trainable_params()` | Count parameters with `requires_grad=True` |
| 8 | `build_generation_prompt()` | The template up to the assistant marker (no response) |
| 9 | `merge_lora_weight()` | The merged weight `base_W + scale * (B @ A)` |

The model (`src/model.py`), tokenizer (`src/tokenizer.py`), dataset builder
(`src/data.py`), and runner (`src/main.py`, which reads top to bottom as the nine
steps) are all provided. The runner's bookkeeping lives in `src/reporting.py` (the
CORRECT/INCORRECT/INCOMPLETE output), `src/prerequisites.py` (the "needs Step N"
checks) and `src/solution.py` (the `--solution` switch). The LoRA injection,
base freezing, and merge loop live in `src/model.py`; the LoRA layer and the merge
loop call back into two functions you write (steps 5 and 9). You only edit `exercise.py`.

The tests live in `tests/`, one file per step (`test_step1_format_example.py`
through `test_step9_merge.py`). Each calls your function on small, hand-made
inputs with a known answer, so you can read the test for the step you are on to
see exactly what is expected. Some steps add one check on the real model, such as
the trainable-parameter count. The Step 6 tests use PyTorch's loss in place of
your Step 3 function, so Step 6 is graded on its own.

## Data

- `data/base_model.pt` &mdash; the frozen Module 5 base checkpoint (the TinyGPT
  config, weights, and character vocabulary). The runner loads it, expands the
  vocabulary by four special tokens, and finetunes on top. It is never
  re-pretrained. (Regenerate it with `src/make_base_checkpoint.py`.)
- `data/sft_pairs.jsonl` &mdash; ~350 toy instruction-response pairs across four
  learnable tasks: uppercase, fixed question-answer, repeat, and reverse. Toy and
  deterministic so a tiny model shows a crisp flip in 150 CPU steps.

The tokenizer is the Module 5 65-character vocabulary plus four **atomic** special
tokens (`<|user|>`, `<|assistant|>`, `<|end|>`, `<|pad|>`), each one token id, for
a vocabulary of 69.

## Extra credit

- **Full finetuning vs LoRA.** In `src/main.py`, replace `freeze_base_(model)` with
  `model.requires_grad_(True)` so every tensor trains, then compare the trainable-parameter count and the sample quality against the LoRA run.
- **Vary the rank `r`.** Try `r = 1, 2, 4, 16`. Watch the quality-vs-size trade-off:
  how small can the adapter get before the flip stops working?
- **Catastrophic-forgetting probe.** After finetuning, feed a raw base-style prompt
  (e.g. `To be, or not to be`) and check whether the model still continues text.
- **Loss-mask ablation.** Build targets **without** masking the prompt, retrain, and
  watch the model start hallucinating its own `<|user|>` prompts.
