:::divider id="divider-masks" title="Attention Masks" sub="Controlling what each token can see"
:::

---

<!-- .slide: id="padding-masks" -->

## Padding Masks

In a batch, sequences have different lengths. Shorter sequences are padded with special tokens. A **padding mask** prevents attention to these fake tokens:

:::columns cols="2" gap="30px"
**Without mask:**

The model wastes capacity attending to padding tokens, which carry no information.
+++
**With mask:**

Set padded positions to $-\infty$ before softmax, so they receive zero weight after normalization.
:::

Padding masks are a practical necessity for batched training.

---

<!-- .slide: id="causal-masks" -->

## Causal Masks

A **causal mask** prevents each token from attending to **future** tokens. This is essential for autoregressive generation (like GPT).

:::columns cols="2" gap="30px"
**Bidirectional attention** (BERT-style)

Every token can attend to every other token. Useful for understanding the full context.

$$\begin{bmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{bmatrix}$$
+++
**Causal attention** (GPT-style)

Each token can only attend to itself and earlier tokens.

$$\begin{bmatrix} 1 & 0 & 0 \\ 1 & 1 & 0 \\ 1 & 1 & 1 \end{bmatrix}$$
:::

---

<!-- .slide: id="why-causal" -->

## Why Causal Masking?

In a language model, we predict the next token given all previous tokens:

$$P(w_t \mid w_1, w_2, \ldots, w_{t-1})$$

:::note
If token $t$ could see token $t+1$, the model would be **cheating** during training &mdash; it could copy the answer instead of learning to predict. Causal masking enforces the same information constraint during training that the model will face during inference, when future tokens do not exist yet.
:::
