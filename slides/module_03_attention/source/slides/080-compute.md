:::divider id="divider-compute" title="Memory and Compute" sub="The cost of every token seeing every other token"
:::

---

:::figure img="images/dao.jpg" name="Tri Dao" kicker="Made Exact Attention Fast"
- Tri Dao and collaborators (2022): FlashAttention
- Observed that the $n \times n$ attention matrix is the bottleneck
- FlashAttention computes exact attention without materializing the full $n \times n$ matrix in GPU HBM
- IO-aware: tiles the computation to keep intermediate results in fast SRAM
- Exact same result as standard attention, but 2-4x faster and 5-20x less memory
- FlashAttention-2 and -3 pushed further gains on newer GPU architectures
:::

---

<!-- .slide: id="n-squared-cost" -->

## The $O(n^2)$ Cost

Attention creates an $n \times n$ score matrix for $n$ tokens. Every token interacts with every other token.

:::columns cols="2" gap="30px"
**Compute:** $O(n^2 \cdot d_k)$ for the matrix multiplications

**Memory:** $O(n^2)$ to store the attention weights

For 128K tokens, the attention matrix has 16 billion entries.
+++
**Why this matters:**

- Doubling the context length quadruples the attention cost
- Long context is expensive because every token can see every other token
- The $O(n^2)$ cost is the main barrier to very long contexts
:::

---

<!-- .slide: id="mitigations" -->

## Modern Mitigations

Several approaches reduce the $O(n^2)$ cost without sacrificing too much quality:

:::columns cols="2" gap="30px"
**FlashAttention**

Computes exact attention without materializing the full matrix. Same result, less memory traffic.
+++
**Sliding-window attention**

Each token attends only to a local window of neighbors (plus a few global tokens). Reduces the effective $n$.
+++
**Multi-query attention (MQA)**

All heads share the same K and V projections. Only Q is per-head. Reduces KV cache size.
+++
**Grouped-query attention (GQA)**

A compromise: heads are divided into groups that share K and V. Used in LLaMA 2/3.
:::
