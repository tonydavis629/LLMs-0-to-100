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

## Review: From Attention to Models

:::columns cols="2" gap="30px"
**What Module 3 gave us**

- Queries choose what each token is looking for
- Keys describe what each token offers
- Values carry the information to copy forward
- A causal mask blocks attention to future tokens
+++
**What is still missing**

- Token IDs must become vectors
- Attention must be wrapped in residuals, normalization, and feed-forward networks
- Many blocks must stack before logits become text
- Architecture choices determine what the model can do
:::
