:::divider id="divider-multi-head" title="Multi-Head Attention" sub="Several lookup patterns running in parallel"
:::

---

<!-- .slide: id="why-multi-head" -->

## Why Multiple Heads?

A single attention head produces one weighted average per token. But a token might need to attend to different things for different reasons:

:::columns cols="2" gap="30px"
**Head 1** might learn syntactic attention &mdash; verbs attend to their subjects

**Head 2** might learn coreference &mdash; pronouns attend to their antecedents

**Head 3** might learn positional attention &mdash; each token attends to its immediate neighbor
+++
Multiple heads let the model learn **different ways to compare tokens** simultaneously, rather than forcing one pattern to serve all purposes.
:::

---

<!-- .slide: id="multi-head-mechanics" -->

## Multi-Head Mechanics

Each head $h$ has its own projections:

$$Q_h = XW_Q^h, \quad K_h = XW_K^h, \quad V_h = XW_V^h$$

Each head independently computes scaled dot-product attention:

$$\text{head}_h = \text{Attention}(Q_h, K_h, V_h)$$

The heads are **concatenated** and projected back:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) \, W_O$$

:::note
**Compute budget:** total projection size across all heads is typically kept equal to $d_{\text{model}}$. With 8 heads and $d_{\text{model}} = 512$, each head uses $d_k = 64$. Multi-head attention is not more expensive than single-head &mdash; it is the same compute, split into independent patterns.
:::
