:::divider id="divider-scaled-dot-product" title="Scaled Dot-Product Attention" sub="Compare, scale, normalize, retrieve"
:::

---

<!-- .slide: id="scaling-problem" -->

## Why Scale by $\sqrt{d_k}$?

Dot products grow with dimension. For $d_k = 64$, the expected magnitude of $\mathbf{q} \cdot \mathbf{k}$ is about 8; for $d_k = 1024$, it is about 32.
<div style="text-align: center; margin: 8px 0;">
<svg viewBox="0 0 820 250" width="100%" style="max-height: 230px;">
  <text x="235" y="22" fill="#f5a623" font-size="13" text-anchor="middle" font-weight="600">without scaling: large logits saturate softmax</text>
  <text x="610" y="22" fill="#4a9eff" font-size="13" text-anchor="middle" font-weight="600">with scaling: weights stay usable</text>
  <g font-size="11" text-anchor="middle">
    <text x="95" y="53" fill="#8892a4">scores</text>
    <text x="95" y="144" fill="#8892a4">softmax</text>
    <rect x="135" y="44" width="42" height="70" rx="4" fill="rgba(245,166,35,0.72)"/><text x="156" y="132" fill="#e8eaf0">16</text>
    <rect x="187" y="72" width="42" height="42" rx="4" fill="rgba(245,166,35,0.45)"/><text x="208" y="132" fill="#e8eaf0">8</text>
    <rect x="239" y="94" width="42" height="20" rx="4" fill="rgba(245,166,35,0.24)"/><text x="260" y="132" fill="#e8eaf0">2</text>
    <rect x="291" y="102" width="42" height="12" rx="4" fill="rgba(245,166,35,0.16)"/><text x="312" y="132" fill="#e8eaf0">0</text>
    <rect x="135" y="155" width="42" height="70" rx="4" fill="rgba(231,76,60,0.80)"/><text x="156" y="242" fill="#e8eaf0">1.00</text>
    <rect x="187" y="222" width="42" height="3" rx="2" fill="rgba(74,158,255,0.20)"/><text x="208" y="242" fill="#8892a4">0.00</text>
    <rect x="239" y="224" width="42" height="1" rx="1" fill="rgba(74,158,255,0.20)"/><text x="260" y="242" fill="#8892a4">0.00</text>
    <rect x="291" y="224" width="42" height="1" rx="1" fill="rgba(74,158,255,0.20)"/><text x="312" y="242" fill="#8892a4">0.00</text>
    <text x="425" y="120" fill="#8892a4" font-size="28">&divide;</text>
    <text x="425" y="145" fill="#8892a4" font-size="13">&radic;d_k</text>
    <text x="470" y="53" fill="#8892a4">scores</text>
    <text x="470" y="144" fill="#8892a4">softmax</text>
    <rect x="510" y="82" width="42" height="32" rx="4" fill="rgba(74,158,255,0.52)"/><text x="531" y="132" fill="#e8eaf0">2.0</text>
    <rect x="562" y="96" width="42" height="18" rx="4" fill="rgba(74,158,255,0.34)"/><text x="583" y="132" fill="#e8eaf0">1.0</text>
    <rect x="614" y="108" width="42" height="6" rx="3" fill="rgba(74,158,255,0.18)"/><text x="635" y="132" fill="#e8eaf0">0.25</text>
    <rect x="666" y="114" width="42" height="1" rx="1" fill="rgba(74,158,255,0.12)"/><text x="687" y="132" fill="#e8eaf0">0</text>
    <rect x="510" y="155" width="42" height="44" rx="4" fill="rgba(74,158,255,0.70)"/><text x="531" y="216" fill="#e8eaf0">0.58</text>
    <rect x="562" y="155" width="42" height="16" rx="4" fill="rgba(74,158,255,0.40)"/><text x="583" y="188" fill="#e8eaf0">0.21</text>
    <rect x="614" y="155" width="42" height="10" rx="4" fill="rgba(74,158,255,0.30)"/><text x="635" y="182" fill="#e8eaf0">0.13</text>
    <rect x="666" y="155" width="42" height="7" rx="4" fill="rgba(74,158,255,0.24)"/><text x="687" y="178" fill="#e8eaf0">0.08</text>
  </g>
</svg>
</div>

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

The same mechanism works on any sequence. In a vision-language model, each text token can attend to **regions of an image** &mdash; the bright areas are where attention concentrates.
<div class="video-container" style="flex-direction: column;">
<img src="images/sualization-of-attention-regions-extracted-from-the-first-Transformer-layer-of.webp" alt="Attention regions over an image in a vision-language model" style="max-height: 470px;">
<p class="text-muted" style="font-size: 12pt; margin-top: 10px;">Attention regions from Pixel-BERT (Huang et al., 2020). Attention concentrates on the objects the text refers to.</p>
</div>

---

<!-- .slide: id="full-attention-formula" -->

## The Complete Formula

Scaled dot-product attention in one line:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Compare, normalize, retrieve. Three matrix operations. No recurrence, no convolution, no fixed-size bottleneck. The model learns what to look at.
