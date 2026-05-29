:::divider id="divider-scaled-dot-product" title="Scaled Dot-Product Attention" sub="The operation at the heart of every Transformer"
:::

---

<!-- .slide: id="scaling-problem" -->

## Why Scale by $\sqrt{d_k}$?

Dot products grow with dimension. For $d_k = 64$, the expected magnitude of $\mathbf{q} \cdot \mathbf{k}$ is about 4; for $d_k = 1024$, it is about 16.

When inputs to softmax are large, the output becomes **sharply peaked** &mdash; one entry near 1, the rest near 0. This produces tiny gradients that make training unstable.

:::note
Dividing by $\sqrt{d_k}$ keeps the variance of the dot product roughly constant regardless of dimension. This preserves gradient flow and stabilizes training.
:::

---

<!-- .slide: id="scaled-softmax" -->

## Scaled Softmax

The full attention weight computation:

$$\alpha_{ij} = \frac{\exp\!\left(\mathbf{q}_i \cdot \mathbf{k}_j / \sqrt{d_k}\right)}{\sum_{j'} \exp\!\left(\mathbf{q}_i \cdot \mathbf{k}_{j'} / \sqrt{d_k}\right)}$$

Each row of the attention matrix sums to 1. The weights form a **soft selection** over all tokens.

:::note
**Without scaling:** for $d_k = 64$, softmax saturates and gradients vanish. With scaling, the distribution stays smooth and learning proceeds. This is a small but critical engineering detail.
:::

---

<!-- .slide: id="attention-output" -->

## The Attention Output

The output is a weighted average of value vectors:

$$\text{output}_i = \sum_j \alpha_{ij} \, \mathbf{v}_j$$

:::columns cols="2" gap="30px"
**Interpretation:**

- Each output token is a mixture of all value vectors
- The mixture weights come from query-key compatibility
- If token $i$ attends strongly to token $j$, the output at position $i$ is dominated by $\mathbf{v}_j$
+++
**Properties:**

- Different tokens can attend to different subsets of the sequence
- The same token can be attended to by many others
- The output dimension matches the value dimension ($d_k$), not the sequence length
:::

---

<!-- .slide: id="full-attention-formula" -->

## The Complete Formula

Scaled dot-product attention in one line:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Compare, normalize, retrieve. Three matrix operations. No recurrence, no convolution, no fixed-size bottleneck. The model learns what to look at.
