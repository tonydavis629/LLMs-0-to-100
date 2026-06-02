:::divider id="title" title="LLMs 0 to 100" sub="Module 4: LLM Architectures"
From the Transformer to GPT-2
:::

---

<!-- .slide: id="review-1" -->

## Review: Module 3

:::columns cols="2" gap="30px"
**Self-Attention as a Learned Lookup**

Compare every token (query) against every other token (key), normalize with softmax, and retrieve a weighted sum of values:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Each output token is a weighted average of value vectors, with weights from query-key compatibility.
+++
**The Missing Pieces**

- A single attention layer is not a model; it needs embeddings, depth (stacked blocks), and an output head
- Positional embeddings inject order into a bag of attention scores
- Causal masking ensures autoregressive generation: each token only attends to itself and previous tokens
:::

---

<!-- .slide: id="review-2" -->

## Review: The Full Transformer Picture

Three design decisions that define what a transformer can do:

1. **Attention direction** &mdash; causal (autoregressive) or bidirectional (encoding)
2. **Architecture family** &mdash; encoder-decoder, encoder-only, or decoder-only
3. **Output objective** &mdash; next-token prediction, masked-token prediction, or sequence-to-sequence

These choices determine whether the model generates, classifies, or translates. This module explores each family in depth.
