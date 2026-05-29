:::divider id="divider-neuron" title="The Neuron Model" sub="Weighted sums, activations, and the simplest learning machine"
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
The analogy is real but breaks down fast: real neurons use spike timing, dendritic computation, and many neurotransmitters
:::

---

<!-- .slide: id="perceptron-equation" -->

## The Perceptron

A single neuron computes:

$$y = \sigma(\mathbf{w} \cdot \mathbf{x} + b)$$

:::columns cols="3" gap="20px"
**$\mathbf{w} \cdot \mathbf{x}$**

Dot product of inputs and weights — how much each input matters
+++
**$+ \; b$**

Bias term — shifts the decision boundary
+++
**$\sigma(\cdot)$**

Activation function — introduces nonlinearity
:::

This is a **linear function** followed by a **nonlinearity**. That pattern is the fundamental building block of every neural network.

---

<!-- .slide: id="activation-functions" -->

## Activation Functions

:::columns cols="4" gap="18px"
<div style="text-align:center;">
<p class="text-lg" style="color: var(--secondary-color); font-weight:600; margin:0;">Step</p>
<p style="color: var(--muted-color); margin:2px 0;">$\sigma(z)=\mathbb{1}[z \geq 0]$</p>
<svg viewBox="0 0 240 150" width="100%" style="max-height:150px;">
  <line x1="120" y1="14" x2="120" y2="136" stroke="#2a3450" stroke-width="1.5"/>
  <line x1="18" y1="125" x2="222" y2="125" stroke="#2a3450" stroke-width="1.5"/>
  <polyline points="20,125 120,125" fill="none" stroke="#f5a623" stroke-width="3"/>
  <polyline points="120,25 220,25" fill="none" stroke="#f5a623" stroke-width="3"/>
  <line x1="120" y1="125" x2="120" y2="25" stroke="#f5a623" stroke-width="1.5" stroke-dasharray="4 4"/>
</svg>
<p style="color: var(--muted-color); font-size:13pt;">Binary; not differentiable at 0</p>
</div>
+++
<div style="text-align:center;">
<p class="text-lg" style="color: var(--secondary-color); font-weight:600; margin:0;">Sigmoid</p>
<p style="color: var(--muted-color); margin:2px 0;">$\dfrac{1}{1 + e^{-z}}$</p>
<svg viewBox="0 0 240 150" width="100%" style="max-height:150px;">
  <line x1="120" y1="14" x2="120" y2="136" stroke="#2a3450" stroke-width="1.5"/>
  <line x1="18" y1="125" x2="222" y2="125" stroke="#2a3450" stroke-width="1.5"/>
  <polyline points="20,124.8 30,124.6 40,124.2 50,123.5 60,122.3 70,120.3 80,116.7 90,110.8 100,101.9 110,89.6 120,75.0 130,60.4 140,48.1 150,39.2 160,33.3 170,29.7 180,27.7 190,26.5 200,25.8 210,25.4 220,25.2" fill="none" stroke="#f5a623" stroke-width="3"/>
</svg>
<p style="color: var(--muted-color); font-size:13pt;">(0,1); vanishing gradients</p>
</div>
+++
<div style="text-align:center;">
<p class="text-lg" style="color: var(--secondary-color); font-weight:600; margin:0;">Tanh</p>
<p style="color: var(--muted-color); margin:2px 0;">$\dfrac{e^z - e^{-z}}{e^z + e^{-z}}$</p>
<svg viewBox="0 0 240 150" width="100%" style="max-height:150px;">
  <line x1="120" y1="14" x2="120" y2="136" stroke="#2a3450" stroke-width="1.5"/>
  <line x1="18" y1="75" x2="222" y2="75" stroke="#2a3450" stroke-width="1.5"/>
  <polyline points="20,125.0 40,125.0 60,124.9 70,124.8 80,124.2 90,122.3 100,116.7 110,101.9 120,75.0 130,48.1 140,33.3 150,27.7 160,25.8 170,25.2 180,25.1 200,25.0 220,25.0" fill="none" stroke="#f5a623" stroke-width="3"/>
</svg>
<p style="color: var(--muted-color); font-size:13pt;">Zero-centered (-1,1)</p>
</div>
+++
<div style="text-align:center;">
<p class="text-lg" style="color: var(--secondary-color); font-weight:600; margin:0;">ReLU</p>
<p style="color: var(--muted-color); margin:2px 0;">$\max(0, z)$</p>
<svg viewBox="0 0 240 150" width="100%" style="max-height:150px;">
  <line x1="120" y1="14" x2="120" y2="136" stroke="#2a3450" stroke-width="1.5"/>
  <line x1="18" y1="125" x2="222" y2="125" stroke="#2a3450" stroke-width="1.5"/>
  <polyline points="20,125 120,125 210,35" fill="none" stroke="#f5a623" stroke-width="3"/>
</svg>
<p style="color: var(--muted-color); font-size:13pt;">Cheap; the modern default</p>
</div>
:::

The course MLP uses **ReLU** in its hidden layer and **sigmoid** on the output (for a probability). <!-- .element: class="text-lg" style="text-align:center; color: var(--muted-color); margin-top: 14px;" -->

---

<!-- .slide: id="why-nonlinearity" -->

## Why Nonlinearity Matters

Stack two linear layers with no activation between them:

$$f(\mathbf{x}) = W_2\,(W_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2$$

Multiply it out and it is still just one linear layer:

$$f(\mathbf{x}) = (W_2 W_1)\,\mathbf{x} + (W_2 \mathbf{b}_1 + \mathbf{b}_2) = W'\mathbf{x} + \mathbf{b}'$$

Any number of linear layers collapses to a single linear function, so depth alone gains **nothing**. The nonlinear activation between layers is what lets a network represent complex relationships.
