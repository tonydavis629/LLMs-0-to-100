:::divider id="divider-scaled-dot-product" title="Scaled Dot-Product Attention" sub="The compare-and-retrieve layer, with scaling and softmax added"
:::

---

<!-- .slide: id="scaled-softmax" -->

## From Scores to the Attention Map

$QK^T$ holds the raw scores. Divide by $\sqrt{d_k}$, then apply softmax to each row:

$$A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$$

- $QK^T$ is the $n \times n$ grid of scores from before: row $i$ compares token $i$'s query with every key
- The only new ingredient is the $\sqrt{d_k}$ divisor, a fixed temperature
- Softmax runs on each row separately, so every row of $A$ sums to 1
- $A$ is the **attention map**

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

<!-- .slide: id="attention-output" -->

## The Output for One Token

Retrieve, for one token: token $i$ uses row $i$ of the attention map $A$ to average the rows of $V$.

$$\mathbf o_i = \sum_j A_{ij} \mathbf v_j$$

<div style="text-align: center; margin: 4px 0;">
<svg viewBox="0 0 880 212" width="100%" style="max-height: 200px;">
  <text x="20" y="48" fill="#50c878" font-size="12" font-weight="600">j: token</text>
  <text x="20" y="64" fill="#50c878" font-size="12" font-weight="600">attended to</text>
  <text x="190" y="16" fill="#8892a4" font-size="11" text-anchor="middle">j = 1</text>
  <rect x="145" y="24" width="90" height="40" rx="5" fill="rgba(80,200,120,0.12)" stroke="#50c878" stroke-width="1.5"/>
  <text x="190" y="41" fill="#e8eaf0" font-size="13" text-anchor="middle" font-weight="600">the</text>
  <text x="190" y="57" fill="#50c878" font-size="11" text-anchor="middle">v<tspan dy="3" font-size="9">1</tspan></text>
  <line x1="190" y1="88" x2="410" y2="158" stroke="#c792ea" stroke-width="2.2" opacity="0.48"/>
  <text x="190" y="80" fill="#c792ea" font-size="11" text-anchor="middle">A<tspan dy="3" font-size="9">31</tspan><tspan dy="-3"> = 0.10</tspan></text>
  <text x="315" y="16" fill="#8892a4" font-size="11" text-anchor="middle">j = 2</text>
  <rect x="270" y="24" width="90" height="40" rx="5" fill="rgba(80,200,120,0.12)" stroke="#50c878" stroke-width="1.5"/>
  <text x="315" y="41" fill="#e8eaf0" font-size="13" text-anchor="middle" font-weight="600">cat</text>
  <text x="315" y="57" fill="#50c878" font-size="11" text-anchor="middle">v<tspan dy="3" font-size="9">2</tspan></text>
  <line x1="315" y1="88" x2="425" y2="158" stroke="#c792ea" stroke-width="6.4" opacity="0.94"/>
  <text x="315" y="80" fill="#c792ea" font-size="11" text-anchor="middle">A<tspan dy="3" font-size="9">32</tspan><tspan dy="-3"> = 0.45</tspan></text>
  <text x="440" y="16" fill="#8892a4" font-size="11" text-anchor="middle">j = 3</text>
  <rect x="395" y="24" width="90" height="40" rx="5" fill="rgba(80,200,120,0.12)" stroke="#50c878" stroke-width="1.5"/>
  <text x="440" y="41" fill="#e8eaf0" font-size="13" text-anchor="middle" font-weight="600">sat</text>
  <text x="440" y="57" fill="#50c878" font-size="11" text-anchor="middle">v<tspan dy="3" font-size="9">3</tspan></text>
  <line x1="440" y1="88" x2="440" y2="158" stroke="#c792ea" stroke-width="4.0" opacity="0.68"/>
  <text x="440" y="80" fill="#c792ea" font-size="11" text-anchor="middle">A<tspan dy="3" font-size="9">33</tspan><tspan dy="-3"> = 0.25</tspan></text>
  <text x="565" y="16" fill="#8892a4" font-size="11" text-anchor="middle">j = 4</text>
  <rect x="520" y="24" width="90" height="40" rx="5" fill="rgba(80,200,120,0.12)" stroke="#50c878" stroke-width="1.5"/>
  <text x="565" y="41" fill="#e8eaf0" font-size="13" text-anchor="middle" font-weight="600">on</text>
  <text x="565" y="57" fill="#50c878" font-size="11" text-anchor="middle">v<tspan dy="3" font-size="9">4</tspan></text>
  <line x1="565" y1="88" x2="455" y2="158" stroke="#c792ea" stroke-width="1.6" opacity="0.41"/>
  <text x="565" y="80" fill="#c792ea" font-size="11" text-anchor="middle">A<tspan dy="3" font-size="9">34</tspan><tspan dy="-3"> = 0.05</tspan></text>
  <text x="690" y="16" fill="#8892a4" font-size="11" text-anchor="middle">j = 5</text>
  <rect x="645" y="24" width="90" height="40" rx="5" fill="rgba(80,200,120,0.12)" stroke="#50c878" stroke-width="1.5"/>
  <text x="690" y="41" fill="#e8eaf0" font-size="13" text-anchor="middle" font-weight="600">mat</text>
  <text x="690" y="57" fill="#50c878" font-size="11" text-anchor="middle">v<tspan dy="3" font-size="9">5</tspan></text>
  <line x1="690" y1="88" x2="470" y2="158" stroke="#c792ea" stroke-width="2.8" opacity="0.54"/>
  <text x="690" y="80" fill="#c792ea" font-size="11" text-anchor="middle">A<tspan dy="3" font-size="9">35</tspan><tspan dy="-3"> = 0.15</tspan></text>
  <rect x="340" y="160" width="200" height="40" rx="6" fill="rgba(63,185,80,0.15)" stroke="#3fb950" stroke-width="2"/>
  <text x="440" y="185" fill="#e8eaf0" font-size="13" text-anchor="middle" font-weight="600">o<tspan dy="3" font-size="10">3</tspan><tspan dy="-3">: new vector for "sat"</tspan></text>
  <text x="20" y="176" fill="#3fb950" font-size="12" font-weight="600">i: token doing</text>
  <text x="20" y="192" fill="#3fb950" font-size="12" font-weight="600">the attending (i = 3)</text>
</svg>
</div>

- $i$ is the token being updated; $j$ runs over every token it can attend to, itself included
- $A_{ij}$ is how much token $i$ attends to token $j$, and $\mathbf v_j$ is row $j$ of $V$

---

<!-- .slide: id="full-attention-formula" -->

## The Complete Formula

**Scaled dot-product attention** (dot-product scores, divided by $\sqrt{d_k}$), for every token at once:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

<div style="text-align: center; margin: 4px 0;">
<svg viewBox="40 0 720 246" width="100%" style="max-height: 205px;">
  <text x="210" y="16" fill="#c792ea" font-size="13" text-anchor="middle" font-weight="600">attention map A</text>
  <text x="460" y="16" fill="#50c878" font-size="13" text-anchor="middle" font-weight="600">V: one row per token</text>
  <text x="660" y="16" fill="#3fb950" font-size="13" text-anchor="middle" font-weight="600">output</text>
  <text x="130" y="44" fill="#8892a4" font-size="11" text-anchor="middle">the</text>
  <text x="170" y="44" fill="#8892a4" font-size="11" text-anchor="middle">cat</text>
  <text x="210" y="44" fill="#8892a4" font-size="11" text-anchor="middle">sat</text>
  <text x="250" y="44" fill="#8892a4" font-size="11" text-anchor="middle">on</text>
  <text x="290" y="44" fill="#8892a4" font-size="11" text-anchor="middle">mat</text>
  <text x="100" y="72" fill="#8892a4" font-size="12" text-anchor="end">the</text>
  <rect x="110" y="52" width="40" height="30" fill="rgba(199,146,234,0.68)" stroke="#3a4258" stroke-width="1"/>
  <rect x="150" y="52" width="40" height="30" fill="rgba(199,146,234,0.46)" stroke="#3a4258" stroke-width="1"/>
  <rect x="190" y="52" width="40" height="30" fill="rgba(199,146,234,0.30)" stroke="#3a4258" stroke-width="1"/>
  <rect x="230" y="52" width="40" height="30" fill="rgba(199,146,234,0.23)" stroke="#3a4258" stroke-width="1"/>
  <rect x="270" y="52" width="40" height="30" fill="rgba(199,146,234,0.23)" stroke="#3a4258" stroke-width="1"/>
  <text x="392" y="72" fill="#c792ea" font-size="11" text-anchor="end">0.10 &#215;</text>
  <rect x="400" y="52" width="30" height="30" fill="rgba(80,200,120,0.80)" stroke="#50c878" stroke-width="1"/><rect x="430" y="52" width="30" height="30" fill="rgba(80,200,120,0.32)" stroke="#50c878" stroke-width="1"/><rect x="460" y="52" width="30" height="30" fill="rgba(80,200,120,0.48)" stroke="#50c878" stroke-width="1"/><text x="505" y="71" fill="#50c878" font-size="14" text-anchor="middle">&#8230;</text>
  <text x="530" y="72" fill="#8892a4" font-size="11">the</text>
  <rect x="600" y="52" width="30" height="30" fill="rgba(63,185,80,0.48)" stroke="#3a4258" stroke-width="1"/><rect x="630" y="52" width="30" height="30" fill="rgba(63,185,80,0.55)" stroke="#3a4258" stroke-width="1"/><rect x="660" y="52" width="30" height="30" fill="rgba(63,185,80,0.48)" stroke="#3a4258" stroke-width="1"/><text x="705" y="71" fill="#8892a4" font-size="14" text-anchor="middle">&#8230;</text>
  <text x="100" y="102" fill="#8892a4" font-size="12" text-anchor="end">cat</text>
  <rect x="110" y="82" width="40" height="30" fill="rgba(199,146,234,0.38)" stroke="#3a4258" stroke-width="1"/>
  <rect x="150" y="82" width="40" height="30" fill="rgba(199,146,234,0.60)" stroke="#3a4258" stroke-width="1"/>
  <rect x="190" y="82" width="40" height="30" fill="rgba(199,146,234,0.46)" stroke="#3a4258" stroke-width="1"/>
  <rect x="230" y="82" width="40" height="30" fill="rgba(199,146,234,0.23)" stroke="#3a4258" stroke-width="1"/>
  <rect x="270" y="82" width="40" height="30" fill="rgba(199,146,234,0.23)" stroke="#3a4258" stroke-width="1"/>
  <text x="392" y="102" fill="#c792ea" font-size="11" text-anchor="end">0.45 &#215;</text>
  <rect x="400" y="82" width="30" height="30" fill="rgba(80,200,120,0.12)" stroke="#50c878" stroke-width="1"/><rect x="430" y="82" width="30" height="30" fill="rgba(80,200,120,0.84)" stroke="#50c878" stroke-width="1"/><rect x="460" y="82" width="30" height="30" fill="rgba(80,200,120,0.40)" stroke="#50c878" stroke-width="1"/><text x="505" y="101" fill="#50c878" font-size="14" text-anchor="middle">&#8230;</text>
  <text x="530" y="102" fill="#8892a4" font-size="11">cat</text>
  <rect x="600" y="82" width="30" height="30" fill="rgba(63,185,80,0.35)" stroke="#3a4258" stroke-width="1"/><rect x="630" y="82" width="30" height="30" fill="rgba(63,185,80,0.64)" stroke="#3a4258" stroke-width="1"/><rect x="660" y="82" width="30" height="30" fill="rgba(63,185,80,0.51)" stroke="#3a4258" stroke-width="1"/><text x="705" y="101" fill="#8892a4" font-size="14" text-anchor="middle">&#8230;</text>
  <text x="100" y="132" fill="#c792ea" font-size="12" text-anchor="end" font-weight="600">sat</text>
  <rect x="110" y="112" width="40" height="30" fill="rgba(199,146,234,0.23)" stroke="#c792ea" stroke-width="1.5"/><text x="130" y="132" fill="#e8eaf0" font-size="11" text-anchor="middle">0.10</text>
  <rect x="150" y="112" width="40" height="30" fill="rgba(199,146,234,0.76)" stroke="#c792ea" stroke-width="1.5"/><text x="170" y="132" fill="#e8eaf0" font-size="11" text-anchor="middle">0.45</text>
  <rect x="190" y="112" width="40" height="30" fill="rgba(199,146,234,0.46)" stroke="#c792ea" stroke-width="1.5"/><text x="210" y="132" fill="#e8eaf0" font-size="11" text-anchor="middle">0.25</text>
  <rect x="230" y="112" width="40" height="30" fill="rgba(199,146,234,0.16)" stroke="#c792ea" stroke-width="1.5"/><text x="250" y="132" fill="#e8eaf0" font-size="11" text-anchor="middle">0.05</text>
  <rect x="270" y="112" width="40" height="30" fill="rgba(199,146,234,0.30)" stroke="#c792ea" stroke-width="1.5"/><text x="290" y="132" fill="#e8eaf0" font-size="11" text-anchor="middle">0.15</text>
  <text x="392" y="132" fill="#c792ea" font-size="11" text-anchor="end">0.25 &#215;</text>
  <rect x="400" y="112" width="30" height="30" fill="rgba(80,200,120,0.20)" stroke="#50c878" stroke-width="1"/><rect x="430" y="112" width="30" height="30" fill="rgba(80,200,120,0.72)" stroke="#50c878" stroke-width="1"/><rect x="460" y="112" width="30" height="30" fill="rgba(80,200,120,0.80)" stroke="#50c878" stroke-width="1"/><text x="505" y="131" fill="#50c878" font-size="14" text-anchor="middle">&#8230;</text>
  <text x="530" y="132" fill="#8892a4" font-size="11">sat</text>
  <rect x="600" y="112" width="30" height="30" fill="rgba(63,185,80,0.30)" stroke="#3fb950" stroke-width="1.5"/><rect x="630" y="112" width="30" height="30" fill="rgba(63,185,80,0.67)" stroke="#3fb950" stroke-width="1.5"/><rect x="660" y="112" width="30" height="30" fill="rgba(63,185,80,0.52)" stroke="#3fb950" stroke-width="1.5"/><text x="705" y="131" fill="#3fb950" font-size="14" text-anchor="middle">&#8230;</text>
  <text x="100" y="162" fill="#8892a4" font-size="12" text-anchor="end">on</text>
  <rect x="110" y="142" width="40" height="30" fill="rgba(199,146,234,0.23)" stroke="#3a4258" stroke-width="1"/>
  <rect x="150" y="142" width="40" height="30" fill="rgba(199,146,234,0.30)" stroke="#3a4258" stroke-width="1"/>
  <rect x="190" y="142" width="40" height="30" fill="rgba(199,146,234,0.53)" stroke="#3a4258" stroke-width="1"/>
  <rect x="230" y="142" width="40" height="30" fill="rgba(199,146,234,0.38)" stroke="#3a4258" stroke-width="1"/>
  <rect x="270" y="142" width="40" height="30" fill="rgba(199,146,234,0.46)" stroke="#3a4258" stroke-width="1"/>
  <text x="392" y="162" fill="#c792ea" font-size="11" text-anchor="end">0.05 &#215;</text>
  <rect x="400" y="142" width="30" height="30" fill="rgba(80,200,120,0.32)" stroke="#50c878" stroke-width="1"/><rect x="430" y="142" width="30" height="30" fill="rgba(80,200,120,0.80)" stroke="#50c878" stroke-width="1"/><rect x="460" y="142" width="30" height="30" fill="rgba(80,200,120,0.16)" stroke="#50c878" stroke-width="1"/><text x="505" y="161" fill="#50c878" font-size="14" text-anchor="middle">&#8230;</text>
  <text x="530" y="162" fill="#8892a4" font-size="11">on</text>
  <rect x="600" y="142" width="30" height="30" fill="rgba(63,185,80,0.38)" stroke="#3a4258" stroke-width="1"/><rect x="630" y="142" width="30" height="30" fill="rgba(63,185,80,0.59)" stroke="#3a4258" stroke-width="1"/><rect x="660" y="142" width="30" height="30" fill="rgba(63,185,80,0.52)" stroke="#3a4258" stroke-width="1"/><text x="705" y="161" fill="#8892a4" font-size="14" text-anchor="middle">&#8230;</text>
  <text x="100" y="192" fill="#8892a4" font-size="12" text-anchor="end">mat</text>
  <rect x="110" y="172" width="40" height="30" fill="rgba(199,146,234,0.30)" stroke="#3a4258" stroke-width="1"/>
  <rect x="150" y="172" width="40" height="30" fill="rgba(199,146,234,0.53)" stroke="#3a4258" stroke-width="1"/>
  <rect x="190" y="172" width="40" height="30" fill="rgba(199,146,234,0.30)" stroke="#3a4258" stroke-width="1"/>
  <rect x="230" y="172" width="40" height="30" fill="rgba(199,146,234,0.30)" stroke="#3a4258" stroke-width="1"/>
  <rect x="270" y="172" width="40" height="30" fill="rgba(199,146,234,0.46)" stroke="#3a4258" stroke-width="1"/>
  <text x="392" y="192" fill="#c792ea" font-size="11" text-anchor="end">0.15 &#215;</text>
  <rect x="400" y="172" width="30" height="30" fill="rgba(80,200,120,0.64)" stroke="#50c878" stroke-width="1"/><rect x="430" y="172" width="30" height="30" fill="rgba(80,200,120,0.24)" stroke="#50c878" stroke-width="1"/><rect x="460" y="172" width="30" height="30" fill="rgba(80,200,120,0.56)" stroke="#50c878" stroke-width="1"/><text x="505" y="191" fill="#50c878" font-size="14" text-anchor="middle">&#8230;</text>
  <text x="530" y="192" fill="#8892a4" font-size="11">mat</text>
  <rect x="600" y="172" width="30" height="30" fill="rgba(63,185,80,0.39)" stroke="#3a4258" stroke-width="1"/><rect x="630" y="172" width="30" height="30" fill="rgba(63,185,80,0.59)" stroke="#3a4258" stroke-width="1"/><rect x="660" y="172" width="30" height="30" fill="rgba(63,185,80,0.48)" stroke="#3a4258" stroke-width="1"/><text x="705" y="191" fill="#8892a4" font-size="14" text-anchor="middle">&#8230;</text>
  <text x="338" y="134" fill="#e8eaf0" font-size="20" text-anchor="middle">&#215;</text>
  <text x="578" y="134" fill="#e8eaf0" font-size="20" text-anchor="middle">=</text>
  <text x="60" y="238" fill="#8892a4" font-size="11">row = token attending, column = token attended to</text>
  <text x="455" y="238" fill="#8892a4" font-size="11" text-anchor="middle">columns = value dims</text>
  <text x="655" y="238" fill="#8892a4" font-size="11" text-anchor="middle">columns = value dims</text>
  <text x="210" y="220" fill="#c792ea" font-size="13" text-anchor="middle" font-weight="600">n &#215; n</text>
  <text x="445" y="220" fill="#50c878" font-size="13" text-anchor="middle" font-weight="600">n &#215; d<tspan dy="3" font-size="9">v</tspan></text>
  <text x="645" y="220" fill="#3fb950" font-size="13" text-anchor="middle" font-weight="600">n &#215; d<tspan dy="3" font-size="9">v</tspan></text>
</svg>
</div>

- One matrix product produces every output row at once. The highlighted row shows how each one is formed: rows of $V$ scaled by their weights, then added
- Generating text only needs the last row to predict the next token (see the causal mask and KV cache sections)

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
