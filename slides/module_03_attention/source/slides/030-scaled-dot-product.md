:::divider id="divider-scaled-dot-product" title="Scaled Dot-Product Attention" sub="The operation at the heart of every Transformer"
:::

---

<!-- .slide: id="scaling-problem" -->

## Why Scale by $\sqrt{d_k}$?

Dot products grow with dimension. For $d_k = 64$, the expected magnitude of $\mathbf{q} \cdot \mathbf{k}$ is about 8; for $d_k = 1024$, it is about 32.

When inputs to softmax are large, the output becomes **sharply peaked** &mdash; one entry near 1, the rest near 0. This produces tiny gradients that make training unstable.

:::note
Dividing by $\sqrt{d_k}$ keeps the variance of the dot product roughly constant regardless of dimension. This preserves gradient flow and stabilizes training.
:::

---

<!-- .slide: id="scaled-softmax" -->

## Scaled Softmax

The attention weight from query $i$ to key $j$:

$$\alpha_{ij} = \frac{\exp\left(\mathbf q_i \cdot \mathbf k_j / \sqrt{d_k}\right)}{\sum_{j'} \exp\left(\mathbf q_i \cdot \mathbf k_{j'} / \sqrt{d_k}\right)}$$

Each row of the attention matrix sums to 1. The weights form a **soft selection** over all tokens.

:::note
**Without scaling:** for large $d_k$, the softmax saturates and gradients vanish. With scaling, the distribution stays smooth and learning proceeds. A small but critical engineering detail.
:::

---

<!-- .slide: id="attention-output" -->

## The Attention Output

The output for each token is a weighted average of the value vectors:

$$\mathbf o_i = \sum_j \alpha_{ij} \mathbf v_j$$

:::columns cols="2" gap="30px"
**Interpretation**

- Each output vector $\mathbf o_i$ is a mixture of all value vectors
- The mixture weights $\alpha_{ij}$ come from query-key compatibility
- If token $i$ attends strongly to token $j$, its output is dominated by $\mathbf v_j$
+++
**Properties**

- Different tokens can attend to different subsets of the sequence
- The same token can be attended to by many others
- The output dimension matches the value dimension, not the sequence length
:::

---

<!-- .slide: id="attention-heatmap" -->

:::interactive id="attn-heatmap" widget="attentionHeatmap" title="Attention as a Heatmap"
:::

---

<!-- .slide: id="attention-not-just-text" -->

## Attention Is Not Just for Text

The same mechanism works on any sequence. In a vision-language Transformer, each text token can attend to **regions of an image** &mdash; the bright areas are where attention concentrates.

<div class="video-container" style="flex-direction: column;">
<img src="images/sualization-of-attention-regions-extracted-from-the-first-Transformer-layer-of.webp" alt="Attention regions over an image in a multi-modal Transformer" style="max-height: 470px;">
<p class="text-muted" style="font-size: 12pt; margin-top: 10px;">Attention regions from the first Transformer layer of Pixel-BERT (Huang et al., 2020). Attention concentrates on the objects the text refers to.</p>
</div>

---

<!-- .slide: id="full-attention-formula" -->

## The Complete Formula

Scaled dot-product attention in one line:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Compare, normalize, retrieve. Three matrix operations. No recurrence, no convolution, no fixed-size bottleneck. The model learns what to look at.
