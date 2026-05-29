:::divider id="divider-kv-cache" title="KV Cache" sub="Reusing work during generation"
:::

---

<!-- .slide: id="kv-cache-idea" -->

## The KV Cache

During autoregressive inference, the model generates one token at a time. Without caching, each step recomputes keys and values for **all** previous tokens.

:::columns cols="2" gap="30px"
**Without KV cache**

Each generation step recomputes K and V for every previous token. Cost per step grows linearly with sequence length. Total cost is $O(n^2)$.
+++
**With KV cache**

Previous keys and values are stored and reused. Each new token only computes its own K and V. Cost per step is constant. Total cost is $O(n)$.
:::

---

<!-- .slide: id="kv-cache-tradeoffs" -->

## KV Cache Tradeoffs

The KV cache is a classic memory-vs-compute tradeoff:

:::columns cols="2" gap="30px"
**Speed win**

Each generation step does $O(1)$ work instead of $O(n)$. Generation is dramatically faster, especially for long sequences.
+++
**Memory cost**

The cache stores $2 \times n_{\text{layers}} \times n_{\text{heads}} \times n_{\text{tokens}} \times d_k$ floats. For a 70B model with 128K context, the KV cache can be tens of gigabytes.
:::

:::note
**KV cache memory is a major limit** for long context and large-batch serving. This is why MQA and GQA exist &mdash; fewer unique K and V heads means a smaller cache.
:::

---

<!-- .slide: id="side-quest-registers" -->

## Side Quest: Register Tokens in Vision Transformers

Darcet, Oquab, Mairal, and Bojanowski (2023) observed **high-norm artifact tokens** in ViT feature maps, often corresponding to low-information background patches.

**Interpretation:** the model repurposes some patch tokens as internal scratch space for computation &mdash; they are not representing the image patch they correspond to, but storing global information.

**Register tokens** add extra learned token positions that the model can use as dedicated scratch space, so image tokens are freed to represent their actual patches.

:::note
This is a concrete example of token positions having roles beyond "word at position $i$." The model invents its own internal bookkeeping.
:::

---

<!-- .slide: id="side-quest-attention-maps" -->

## Side Quest: Attention Maps Are Not Always Explanations

Attention weights can be useful visual diagnostics &mdash; they show where the model is looking. But:

:::columns cols="2" gap="30px"
**Attention weights can suggest hypotheses**

- A pronoun attending to its antecedent is a real pattern
- Useful for debugging and building intuition
+++
**High attention does not prove causal importance**

- Attention is one of many operations in the network
- Ablation tests are needed for stronger claims
- The model may compensate through other pathways
:::

**Distinction:** visualization can suggest hypotheses, but ablation tests are needed for stronger claims.
