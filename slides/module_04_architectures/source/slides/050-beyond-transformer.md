:::divider id="divider-beyond" title="Beyond the Vanilla Transformer" sub="Modern blocks, MoE, and sub-quadratic alternatives"
:::

---

<!-- .slide: id="modern-block" -->

## The Modern Block: 2017 vs Llama-Style

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

Mostly independent changes; together they make training more stable and efficient.

---

<!-- .slide: id="moe" -->

## Mixture of Experts

<div class="moe-diagram">
  <div class="router-box">router<br><span>top-k experts per token</span></div>
  <div class="expert-grid">
    <div class="expert active">expert 3<br><span>GPU active</span></div>
    <div class="expert">expert 8<br><span>CPU or cold GPU</span></div>
    <div class="expert active">expert 17<br><span>GPU active</span></div>
    <div class="expert">expert 42<br><span>CPU or cold GPU</span></div>
  </div>
</div>

$$\text{MoE}(\mathbf{x}) = \sum_{i=1}^{E} g(\mathbf{x})_i \cdot \text{FFN}_i(\mathbf{x})$$

- $g(\mathbf{x})$ is the router: a sparse gate, nonzero only for the top-$k$ experts
- Sparse activation buys far more parameters at roughly the same compute per token
- Serving trade-off: active experts need fast GPU access; inactive ones can sit in CPU memory or on other devices

---

<!-- .slide: id="sub-quadratic" -->

## Sub-Quadratic Alternatives

Attention is $O(n^2)$ in sequence length; the bottleneck for long contexts. Alternatives under active research:

- **State-space models** (Mamba): recurrent update with input-dependent gates; linear in length, fixed-size state
- **Linear attention:** kernelize the softmax; no $n \times n$ matrix
- **RWKV:** recurrent update with linear attention-like weights
- **Hybrids** (Jamba): attention layers for short-range precision, Mamba layers for long-range compression

The transformer still dominates; the $O(n^2)$ cost keeps the search alive.
