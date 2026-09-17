:::divider id="divider-neuron" title="The Neuron Model"
:::

---

<!-- .slide: id="biological-analogy" -->

## A Loose Biological Analogy

:::columns cols="2" gap="30px" valign="center"
<div style="text-align: center;">
<p class="text-lg" style="color: var(--primary-color); font-weight: 600; margin-bottom: 6px;">Biological Neuron</p>
<svg viewBox="0 0 300 200" width="100%" style="max-height:240px;">
  <g stroke="#4a9eff" stroke-width="2.5" fill="none" stroke-linecap="round">
    <path d="M16,40 C52,56 70,82 96,96"/>
    <path d="M10,100 C48,100 66,102 94,104"/>
    <path d="M16,162 C52,148 70,124 96,112"/>
    <path d="M40,24 C64,52 80,74 100,92"/>
    <path d="M40,178 C64,150 80,130 100,116"/>
  </g>
  <ellipse cx="122" cy="104" rx="34" ry="30" fill="#0d1225" stroke="#4a9eff" stroke-width="2.5"/>
  <circle cx="122" cy="104" r="9" fill="#4a9eff"/>
  <line x1="156" y1="104" x2="246" y2="104" stroke="#f5a623" stroke-width="4" stroke-linecap="round"/>
  <g stroke="#f5a623" stroke-width="2.5" fill="none" stroke-linecap="round">
    <path d="M246,104 L272,86"/>
    <path d="M246,104 L278,104"/>
    <path d="M246,104 L272,122"/>
  </g>
  <text x="20" y="196" fill="#8892a4" font-size="13">Dendrites</text>
  <text x="100" y="156" fill="#8892a4" font-size="13" text-anchor="middle">Cell body</text>
  <text x="240" y="150" fill="#8892a4" font-size="13" text-anchor="middle">Axon</text>
</svg>
<p class="text-lg" style="color: var(--muted-color); margin-top: 4px;">Dendrites in, cell body integrates, axon fires</p>
</div>
+++
<div style="text-align: center;">
<p class="text-lg" style="color: var(--primary-color); font-weight: 600; margin-bottom: 6px;">Artificial Neuron</p>
<svg viewBox="0 0 340 200" width="100%" style="max-height:240px;">
  <defs>
    <marker id="bioarrow" markerWidth="8" markerHeight="8" refX="5" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="#8892a4"/>
    </marker>
  </defs>
  <circle cx="38" cy="56" r="20" fill="#0d1225" stroke="#4a9eff" stroke-width="2.5"/>
  <text x="38" y="61" fill="#e8eaf0" font-size="15" text-anchor="middle">x₁</text>
  <circle cx="38" cy="148" r="20" fill="#0d1225" stroke="#4a9eff" stroke-width="2.5"/>
  <text x="38" y="153" fill="#e8eaf0" font-size="15" text-anchor="middle">x₂</text>
  <line x1="58" y1="56" x2="142" y2="96" stroke="#8892a4" stroke-width="2" marker-end="url(#bioarrow)"/>
  <line x1="58" y1="148" x2="142" y2="118" stroke="#8892a4" stroke-width="2" marker-end="url(#bioarrow)"/>
  <text x="90" y="66" fill="#f5a623" font-size="14" text-anchor="middle">w₁</text>
  <text x="90" y="146" fill="#f5a623" font-size="14" text-anchor="middle">w₂</text>
  <circle cx="170" cy="107" r="28" fill="#0d1225" stroke="#4a9eff" stroke-width="2.5"/>
  <text x="170" y="114" fill="#e8eaf0" font-size="20" text-anchor="middle">Σ</text>
  <text x="170" y="166" fill="#8892a4" font-size="13" text-anchor="middle">+ bias b</text>
  <line x1="198" y1="107" x2="230" y2="107" stroke="#8892a4" stroke-width="2" marker-end="url(#bioarrow)"/>
  <rect x="232" y="82" width="54" height="50" rx="6" fill="#0d1225" stroke="#3fb950" stroke-width="2.5"/>
  <text x="259" y="114" fill="#e8eaf0" font-size="20" text-anchor="middle">σ</text>
  <line x1="286" y1="107" x2="318" y2="107" stroke="#8892a4" stroke-width="2" marker-end="url(#bioarrow)"/>
  <text x="330" y="112" fill="#e8eaf0" font-size="15" text-anchor="middle">y</text>
</svg>
<p class="text-lg" style="color: var(--muted-color); margin-top: 4px;">Inputs &times; weights, sum + bias, activation out</p>
</div>
:::

:::note
Real neurons use spike timing, dendritic computation, and many neurotransmitters
:::

---

<!-- .slide: id="perceptron-equation" -->

## The Perceptron

A single neuron computes $y = \textcolor{#3fb950}{\sigma}(\textcolor{#f5a623}{\mathbf w} \cdot \textcolor{#4a9eff}{\mathbf x} + \textcolor{#c678dd}{b})$

:::columns grid="0.9fr 1.2fr" gap="30px" valign="start"
<div style="padding-top:36px;">
<p class="text-lg" style="margin:0 0 10px 0; white-space:nowrap;">$\textcolor{#f5a623}{\mathbf w}$ &nbsp; Weights: how much each input matters</p>
<p class="text-lg" style="margin:0 0 10px 0; white-space:nowrap;">$\textcolor{#4a9eff}{\mathbf x}$ &nbsp; Inputs</p>
<p class="text-lg" style="margin:0 0 10px 0; white-space:nowrap;">$\textcolor{#c678dd}{b}$ &nbsp; Bias: shifts the boundary</p>
<p class="text-lg" style="margin:0; white-space:nowrap;">$\textcolor{#3fb950}{\sigma}$ &nbsp; Activation: adds nonlinearity</p>
</div>
+++
<div style="text-align: center;">
<svg viewBox="0 0 340 200" width="100%" style="max-height:240px;">
  <defs>
    <marker id="peqarrow" markerWidth="8" markerHeight="8" refX="5" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="#8892a4"/>
    </marker>
  </defs>
  <circle cx="38" cy="56" r="20" fill="#0d1225" stroke="#4a9eff" stroke-width="2.5"/>
  <text x="38" y="61" fill="#4a9eff" font-size="15" text-anchor="middle">x₁</text>
  <circle cx="38" cy="148" r="20" fill="#0d1225" stroke="#4a9eff" stroke-width="2.5"/>
  <text x="38" y="153" fill="#4a9eff" font-size="15" text-anchor="middle">x₂</text>
  <line x1="58" y1="56" x2="142" y2="96" stroke="#8892a4" stroke-width="2" marker-end="url(#peqarrow)"/>
  <line x1="58" y1="148" x2="142" y2="118" stroke="#8892a4" stroke-width="2" marker-end="url(#peqarrow)"/>
  <text x="90" y="66" fill="#f5a623" font-size="14" text-anchor="middle">w₁</text>
  <text x="90" y="146" fill="#f5a623" font-size="14" text-anchor="middle">w₂</text>
  <circle cx="170" cy="107" r="28" fill="#0d1225" stroke="#4a9eff" stroke-width="2.5"/>
  <text x="170" y="114" fill="#e8eaf0" font-size="20" text-anchor="middle">Σ</text>
  <text x="170" y="166" fill="#c678dd" font-size="13" text-anchor="middle">+ bias b</text>
  <line x1="198" y1="107" x2="230" y2="107" stroke="#8892a4" stroke-width="2" marker-end="url(#peqarrow)"/>
  <rect x="232" y="82" width="54" height="50" rx="6" fill="#0d1225" stroke="#3fb950" stroke-width="2.5"/>
  <text x="259" y="114" fill="#3fb950" font-size="20" text-anchor="middle">σ</text>
  <line x1="286" y1="107" x2="318" y2="107" stroke="#8892a4" stroke-width="2" marker-end="url(#peqarrow)"/>
  <text x="330" y="112" fill="#e8eaf0" font-size="15" text-anchor="middle">y</text>
</svg>
<div class="interactive-host" data-widget="perceptronLine" style="height:150px; display:flex; margin-top:10px;"></div>
</div>
:::
