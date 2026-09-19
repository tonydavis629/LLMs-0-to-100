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

From the `exercises/` directory:

```bash
uv run python module_06_finetuning/src/main.py
```

The runner goes through the ten steps in order. Each step's header line carries a tag, and the step's output follows: what your code produced on the real model, then one line per test from `tests/`. The tags are:

| Tag | Meaning |
|-----|---------|
| `CORRECT` | every test for the step passed |
| `INCORRECT` | your code ran but a test failed; the expected and actual values are printed under it |
| `INCOMPLETE` | the function still raises `NotImplementedError` |

```
=== Step 8: count_trainable_params() === CORRECT
Trainable (LoRA adapters):  65,536
Total (base + adapters):    884,096
Fraction trainable:         7.41%
  CORRECT    counts every element: nn.Linear(3, 2) has 2*3 + 2 = 8 trainable numbers
  CORRECT    skips frozen tensors: a rank-2 adapter on a frozen 4 -> 3 layer gives 2*4 + 3*2 = 14
  CORRECT    a fully frozen layer has 0 trainable parameters
  CORRECT    real model: the count is exactly the LoRA A and B matrices, 65,536 numbers
```

Step 7 finetunes the model, which takes about two minutes on a laptop CPU. Every other step takes a few seconds. Run a single step with `--step` (1 to 10):

```bash
uv run python module_06_finetuning/src/main.py --step 3
```

Step 9 samples the base model and, if Step 7 finetuned it in the same run, the finetuned model too. Step 10 merges whatever adapters the run has trained. Run on their own, both steps note that nothing has been finetuned yet.

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

The tests live in `tests/`, one file per step (`test_step1_format_example.py`
through `test_step10_merge.py`). Each calls your function on small, hand-made
inputs with a known answer, so you can read the test for the step you are on to
see exactly what is expected. Some steps add one check on the real model, such as
the trainable-parameter count. The Step 7 tests use PyTorch's loss in place of
your Step 3 function, so Step 7 is graded on its own.

## Data

- `data/base_model.pt` &mdash; the frozen Module 5 base checkpoint (the TinyGPT
  config, weights, and character vocabulary). The runner loads it, expands the
  vocabulary by four special tokens, and finetunes on top. It is never
  re-pretrained. (Regenerate it with `src/make_base_checkpoint.py`.)
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
