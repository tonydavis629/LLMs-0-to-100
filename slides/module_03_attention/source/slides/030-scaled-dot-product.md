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
    <rect x="510" y="181" width="42" height="44" rx="4" fill="rgba(74,158,255,0.70)"/><text x="531" y="242" fill="#e8eaf0">0.58</text>
    <rect x="562" y="209" width="42" height="16" rx="4" fill="rgba(74,158,255,0.40)"/><text x="583" y="242" fill="#e8eaf0">0.21</text>
    <rect x="614" y="215" width="42" height="10" rx="4" fill="rgba(74,158,255,0.30)"/><text x="635" y="242" fill="#e8eaf0">0.13</text>
    <rect x="666" y="218" width="42" height="7" rx="4" fill="rgba(74,158,255,0.24)"/><text x="687" y="242" fill="#e8eaf0">0.08</text>
  </g>
</svg>
</div>

Large softmax inputs produce a **sharply peaked** output: one entry near 1, the rest near 0. Tiny gradients follow, and training destabilizes.

:::note
Dividing by $\sqrt{d_k}$ keeps the dot-product variance roughly constant at any dimension. Gradients keep flowing.
:::

---

<!-- .slide: id="scaled-softmax" -->

## From Scores to the Attention Map

Apply the same softmax to the **scaled** scores, once per query. The weight from query $i$ to key $j$:

$$\alpha_{ij} = \frac{\exp\left(\mathbf q_i \cdot \mathbf k_j / \sqrt{d_k}\right)}{\sum_{j'} \exp\left(\mathbf q_i \cdot \mathbf k_{j'} / \sqrt{d_k}\right)}$$

- Only new ingredient: the $\sqrt{d_k}$ divisor (a fixed temperature)
- Each query $i$ produces one row $\alpha_{i\cdot}$
- Stacked rows form the **attention map**

---

<!-- .slide: id="attention-output" -->

## The Attention Output

The output for each token is a weighted average of the value vectors:

$$\mathbf o_i = \sum_j \alpha_{ij} \mathbf v_j$$

:::columns cols="2" gap="30px"
**Interpretation**

- Each $\mathbf o_i$ is a mixture of all value vectors
- The weights $\alpha_{ij}$ come from query-key compatibility
- Strong attention to token $j$ means $\mathbf v_j$ dominates the output
+++
**Properties**

- Different tokens can attend to different subsets of the sequence
- The same token can be attended to by many others
- The output dimension matches the value dimension, not the sequence length
:::

---

<!-- .slide: id="full-attention-formula" -->

## The Complete Formula

Scaled dot-product attention in one line:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Compare, normalize, retrieve. No recurrence, no convolution, no fixed-size bottleneck. The model learns what to look at.

---

<!-- .slide: id="transformer-stack-preview" -->

## Where Attention Lives

A transformer is a stack of identical blocks: attention, then an MLP, each adding its result back to its input.

<div style="text-align: center; margin: 5px 0;">
<svg viewBox="0 24 880 196" width="100%" style="max-height: 215px;">
  <rect x="130" y="30" width="600" height="184" rx="8" fill="rgba(136,146,164,0.05)" stroke="#8892a4" stroke-width="1.3" stroke-dasharray="6 4"/>
  <text x="146" y="54" fill="#8892a4" font-size="13" font-weight="600">one block</text>
  <text x="714" y="54" fill="#c792ea" font-size="14" font-weight="600" text-anchor="end">repeated N times</text>
  <rect x="16" y="162" width="72" height="36" rx="5" fill="#0d1225" stroke="#e8eaf0" stroke-width="1.5"/>
  <text x="52" y="185" fill="#e8eaf0" font-size="13" text-anchor="middle" font-weight="600">tokens</text>
  <line x1="88" y1="180" x2="396" y2="180" stroke="#e8eaf0" stroke-width="2" marker-end="url(#arrtsw)"/>
  <line x1="424" y1="180" x2="676" y2="180" stroke="#e8eaf0" stroke-width="2" marker-end="url(#arrtsw)"/>
  <line x1="704" y1="180" x2="786" y2="180" stroke="#e8eaf0" stroke-width="2" marker-end="url(#arrtsw)"/>
  <text x="290" y="204" fill="#8892a4" font-size="12" text-anchor="middle">residual connection (input passes straight through)</text>
  <path d="M190 180 L190 104 L230 104" fill="none" stroke="#4a9eff" stroke-width="1.5" marker-end="url(#arrtsb)"/>
  <rect x="232" y="78" width="140" height="52" rx="6" fill="rgba(74,158,255,0.12)" stroke="#4a9eff" stroke-width="1.5"/>
  <text x="302" y="100" fill="#4a9eff" font-size="14" text-anchor="middle" font-weight="600">Attention</text>
  <text x="302" y="119" fill="#8892a4" font-size="11" text-anchor="middle">tokens share information</text>
  <path d="M372 104 L410 104 L410 166" fill="none" stroke="#4a9eff" stroke-width="1.5" marker-end="url(#arrtsb)"/>
  <circle cx="410" cy="180" r="13" fill="#0d1225" stroke="#e8eaf0" stroke-width="1.5"/>
  <text x="410" y="186" fill="#e8eaf0" font-size="18" text-anchor="middle" font-weight="600">+</text>
  <path d="M470 180 L470 104 L510 104" fill="none" stroke="#f5a623" stroke-width="1.5" marker-end="url(#arrtso)"/>
  <rect x="512" y="78" width="140" height="52" rx="6" fill="rgba(245,166,35,0.12)" stroke="#f5a623" stroke-width="1.5"/>
  <text x="582" y="100" fill="#f5a623" font-size="14" text-anchor="middle" font-weight="600">MLP</text>
  <text x="582" y="119" fill="#8892a4" font-size="11" text-anchor="middle">each token on its own</text>
  <path d="M652 104 L690 104 L690 166" fill="none" stroke="#f5a623" stroke-width="1.5" marker-end="url(#arrtso)"/>
  <circle cx="690" cy="180" r="13" fill="#0d1225" stroke="#e8eaf0" stroke-width="1.5"/>
  <text x="690" y="186" fill="#e8eaf0" font-size="18" text-anchor="middle" font-weight="600">+</text>
  <rect x="788" y="162" width="76" height="36" rx="5" fill="#0d1225" stroke="#3fb950" stroke-width="2"/>
  <text x="826" y="185" fill="#e8eaf0" font-size="13" text-anchor="middle" font-weight="600">output</text>
  <defs>
    <marker id="arrtsw" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#e8eaf0"/></marker>
    <marker id="arrtsb" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#4a9eff"/></marker>
    <marker id="arrtso" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#f5a623"/></marker>
  </defs>
</svg>
</div>

$$x \leftarrow x + \text{Attention}(x) \qquad\qquad x \leftarrow x + \text{MLP}(x)$$

- **Attention** is the only place tokens exchange information; the **MLP** (Module 2) works on each token separately
- With the residual connection, each layer only learns an update to its input
- Module 4 builds the full block

---

<!-- .slide: id="attention-heatmap" -->

:::interactive id="attn-heatmap" widget="attentionHeatmap" title="Attention as a Heatmap"
:::

---

<!-- .slide: id="attention-not-just-text" -->

## Attention Is Not Just for Text

The same mechanism works on any sequence. In a vision-language model, each text token attends to **regions of an image**. Bright areas show where attention concentrates.
<div class="video-container" style="flex-direction: column;">
<img src="images/sualization-of-attention-regions-extracted-from-the-first-Transformer-layer-of.webp" alt="Attention regions over an image in a vision-language model" style="max-height: 470px;">
<p class="text-muted" style="font-size: 12pt; margin-top: 10px;">Attention regions from Pixel-BERT (Huang et al., 2020). Attention concentrates on the objects the text refers to.</p>
</div>
