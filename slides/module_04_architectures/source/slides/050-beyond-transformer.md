:::divider id="divider-beyond" title="Beyond the Vanilla Transformer" sub="Modern blocks, MoE, and sub-quadratic alternatives"
:::

---

<!-- .slide: id="modern-block" -->

## The Modern Block: 2017 vs Llama-Style

The vanilla transformer block has been refined in nearly every component:

:::columns cols="2" gap="30px"
**Original Transformer (2017)**

- LayerNorm
- ReLU activation in FFN
- Sinusoidal positional embeddings
- Post-norm (normalize after sub-layer)
- Bias terms in all linear layers
+++
**Modern Llama-Style**

- RMSNorm (simpler, faster)
- SwiGLU activation (smoother, better gradients)
- RoPE (rotary positional embeddings, callback to Module 3)
- Pre-norm (normalize before sub-layer)
- **Dropped bias terms** in linear layers (saves parameters)
:::

These changes are mostly independent but together produce a more stable and more efficient training run.

---

<!-- .slide: id="moe" -->

## Mixture of Experts

Replace the single FFN with **many expert FFNs** plus a router that activates only a few per token:

$$\text{MoE}(\mathbf{x}) = \sum_{i=1}^{E} g(\mathbf{x})_i \cdot \text{FFN}_i(\mathbf{x})$$

where $g(\mathbf{x})$ is the router output (a sparse gate) and only the top-$k$ experts receive nonzero weight.

This buys far more parameters at roughly the same compute per token: a 176B-parameter MoE model may only activate 33B parameters on any given forward pass. The serving and memory implications come later (forward reference to Module 9).

---

:::figure img="images/shazeer.jpg" name="Noam Shazeer" kicker="Co-Author of the Original Transformer; Lead on MoE"
- Shazeer co-authored "Attention Is All You Need" and led the sparsely-gated Mixture-of-Experts layer
- He also introduced SwiGLU and other modern block improvements
- Most of the transformer ecosystem traces back to his architectural innovations
- Shazeer et al., <https://arxiv.org/abs/1701.06538>; Touvron et al., <https://arxiv.org/abs/2302.13971>
:::

---

<!-- .slide: id="sub-quadratic" -->

## Sub-Quadratic Alternatives

Attention is $O(n^2)$ in sequence length. For very long contexts, this is the bottleneck. Active research explores alternatives:

- **State-space models** (Mamba): a recurrent update with input-dependent gates. Linear in sequence length, but the state is a fixed-size compression.
- **Linear attention:** kernelize the softmax to avoid the $n \times n$ matrix entirely. Exact in some variants, approximate in others.
- **RWKV:** combines a recurrent update with linear attention-like weights, aiming to get the best of both worlds.

Hybrids like **Jamba** mix attention layers with Mamba layers: attention for short-range precision, state-space for long-range compression.

These are clearly labeled active research. The transformer remains the dominant architecture, but the $O(n^2)$ cost ensures that alternatives will continue to be explored.
