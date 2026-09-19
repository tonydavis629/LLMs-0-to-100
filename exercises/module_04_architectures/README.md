# Module 4: LLM Architectures

## Overview

Assemble a complete decoder-only transformer, load real GPT-2 weights from HuggingFace, and generate text. You will build the embedding layer, feed-forward network, transformer block, and full model stack, then implement greedy decoding and temperature-based sampling.

No training yet (that is Module 5); the goal is to see a model you assembled from scratch produce real output.

## Setup

From the `exercises/` directory:

```bash
uv sync
```

This module adds the `transformers` library so we can download pretrained weights and tokenizers.

## Running

```bash
uv run python module_04_architectures/src/main.py
```

The runner goes through the steps in order. Each step's header line carries a tag, and the step's output follows: what your code produced, then one line per test from `tests/`. The tags are:

| Tag | Meaning |
|-----|---------|
| `CORRECT` | every test for the step passed |
| `INCORRECT` | your code ran but a test failed; the expected and actual values are printed under it |
| `INCOMPLETE` | the function still raises `NotImplementedError` |

```
=== Step 3: TransformerBlock.forward() === CORRECT
Input (1, 5, 768) -> output (1, 5, 768)
Parameters: 7,087,872 per block, 4,722,432 of them in the FFN
  CORRECT    keeps the shape: (2, 5, 8) in gives (2, 5, 8) out
  CORRECT    adds each sub-layer back to its input: attn(x)=2x, ffn(x)=10x turn x=1 into 33
  CORRECT    normalizes before each sub-layer: x + attn(ln1(x)), then x + ffn(ln2(x))
```

Run a single step with `--step` (1 to 6):

```
uv run python module_04_architectures/src/main.py --step 3
```

Step 4 loads the real GPT-2 checkpoint into your model (downloaded from HuggingFace on the first run, then cached) and saves a plot of the next-token probabilities to `module_04_architectures/output/token_probs.png`. Steps 5 and 6 generate text with that model, so their demos wait until Steps 1 to 4 are done. A step that is waiting says which earlier step it needs.

`exercise.py` at the module root is the only file you edit. Everything already written for you lives in `src/`. The Module 3 attention block (`src/attention.py`), the GPT-2 weight loader (`src/pretrained.py`) and the top-k sampling helper (`src/sampling.py`) are provided. Run the finished answers with `--solution`:

```
uv run python module_04_architectures/src/main.py --solution
```

## What to implement

Open `exercise.py` and fill in each `raise NotImplementedError(...)` line. Each requires only one line of code.

| Step | Function | What it does |
|------|----------|--------------|
| 1 | `EmbeddingLayer.forward()` | Token embedding lookup + positional embedding addition |
| 2 | `FeedForward.forward()` | Linear -> GELU -> Linear -> dropout |
| 3 | `TransformerBlock.forward()` | Pre-norm attention with residual, then pre-norm FFN with residual |
| 4 | `GPT2Model.forward()` | Full forward pass: embed -> N blocks -> final norm -> LM head |
| 5 | `greedy_decode()` | Argmax next token, append, repeat |
| 6 | `sample_with_temperature_topk()` | Scale logits by temperature, truncate to top-k, then sample |

In Step 4 the runner loads the real GPT-2 checkpoint into your model for you.

Everything else is provided: `src/attention.py` (the Module 3 attention block), `src/pretrained.py` (`load_gpt2_weights()`), `src/sampling.py` (the top-k helper), `src/visualization.py` (plots) and `src/main.py` (the runner). The tests live in `tests/`, one file per step (`test_step1_embedding.py` through `test_step6_temperature.py`). Each calls your code on small inputs with a known answer. Where a step depends on other parts of the model, its tests swap those parts for simple stand-ins (see `tests/fakes.py`), so each step is tested on its own. Steps 4 and 5 also compare your work against Hugging Face's own GPT-2. You only edit `exercise.py`.

## Extra credit

- Implement **top-p (nucleus) sampling**: instead of keeping a fixed number of tokens (top-k), keep the smallest set of tokens whose cumulative probability exceeds p.
- **Tie the embedding and output weights**: set `self.lm_head.weight = self.embed.token_embed.weight` and verify the parameter count drops.
- **Swap the causal mask for a bidirectional one** in `CausalSelfAttention` and observe how open-ended generation breaks (no single-token history to condition on).
- Implement a **tiny BPE training loop** on a short string (a handful of merges) to see tokenization from the inside.
