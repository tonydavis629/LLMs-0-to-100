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

The runner gracefully skips any step that still raises `NotImplementedError`, so you can run after each fill-in.

## What to implement

Open `exercise.py` and fill in each `raise NotImplementedError(...)` line. Each requires only one line of code.

| Step | Function | What it does |
|------|----------|--------------|
| 1 | `EmbeddingLayer.forward()` | Token embedding lookup + positional embedding addition |
| 2 | `FeedForward.forward()` | Linear -> GELU -> Linear -> dropout |
| 3 | `TransformerBlock.forward()` | Pre-norm attention with residual, then pre-norm FFN with residual |
| 4 | `GPT2Model.forward()` | Full forward pass: embed -> N blocks -> final norm -> LM head |
| 5 | (provided) | `load_gpt2_weights()` maps and copies pretrained tensors |
| 6 | `greedy_decode()` | Argmax next token, append, repeat |
| 7 | `sample_with_temperature_topk()` | Scale logits by temperature, truncate to top-k, then sample |

`src/main.py` is the runner and `src/visualization.py` holds plotting helpers &mdash; both are provided. You should only need to edit `exercise.py`.

## Extra credit

- Implement **top-p (nucleus) sampling**: instead of keeping a fixed number of tokens (top-k), keep the smallest set of tokens whose cumulative probability exceeds p.
- **Tie the embedding and output weights**: set `self.lm_head.weight = self.embed.token_embed.weight` and verify the parameter count drops.
- **Swap the causal mask for a bidirectional one** in `CausalSelfAttention` and observe how open-ended generation breaks (no single-token history to condition on).
- Implement a **tiny BPE training loop** on a short string (a handful of merges) to see tokenization from the inside.
