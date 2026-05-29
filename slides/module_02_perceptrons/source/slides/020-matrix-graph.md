:::divider id="divider-matrix-graph" title="Networks as Matrices" sub="Why GPUs matter"
:::

---

<!-- .slide: id="matrix-natural-graph" -->

## Natural Graphs Become Sparse Matrices

Every graph is a matrix: entry $(i,j)$ is the weight of the edge from node $j$ to node $i$; zero means no edge.

:::columns grid="1.05fr 0.95fr" gap="28px" valign="center"
<svg viewBox="0 0 520 270" width="100%" style="max-height:310px;">
  <defs>
    <marker id="sgar" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="#8892a4"/>
    </marker>
  </defs>
  <g stroke="#8892a4" stroke-width="2.2" fill="none" marker-end="url(#sgar)">
    <path d="M90,64 C145,34 185,42 230,74"/>
    <path d="M90,64 C126,120 126,166 90,216"/>
    <path d="M230,74 C302,70 336,104 376,142"/>
    <path d="M90,216 C150,226 204,206 256,170"/>
    <path d="M256,170 C306,186 338,174 376,142"/>
    <path d="M376,142 C420,112 442,84 464,48"/>
  </g>
  <g fill="#0d1225" stroke="#4a9eff" stroke-width="2.5">
    <circle cx="90" cy="64" r="22"/><circle cx="230" cy="74" r="22"/>
    <circle cx="376" cy="142" r="22"/><circle cx="90" cy="216" r="22"/>
    <circle cx="256" cy="170" r="22"/><circle cx="464" cy="48" r="22"/>
  </g>
  <g fill="#e8eaf0" font-size="16" text-anchor="middle">
    <text x="90" y="70">A</text><text x="230" y="80">B</text><text x="376" y="148">C</text>
    <text x="90" y="222">D</text><text x="256" y="176">E</text><text x="464" y="54">F</text>
  </g>
  <g fill="#f5a623" font-size="13" text-anchor="middle">
    <text x="158" y="48">0.7</text><text x="116" y="142">1.2</text><text x="307" y="84">0.5</text>
    <text x="170" y="223">0.9</text><text x="318" y="184">0.4</text><text x="430" y="88">1.1</text>
  </g>
</svg>
+++
<div style="text-align:center;">
<p class="text-lg" style="color: var(--primary-color); font-weight:600; margin: 0 0 8px 0;">Adjacency / weight matrix</p>
<table style="margin: 0 auto; border-collapse: collapse; color: var(--text-color); font-size: 15pt;">
  <tr><td></td><td>A</td><td>B</td><td>C</td><td>D</td><td>E</td><td>F</td></tr>
  <tr><td>A</td><td>0</td><td style="color:#f5a623;">0.7</td><td>0</td><td style="color:#f5a623;">1.2</td><td>0</td><td>0</td></tr>
  <tr><td>B</td><td>0</td><td>0</td><td style="color:#f5a623;">0.5</td><td>0</td><td>0</td><td>0</td></tr>
  <tr><td>C</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td style="color:#f5a623;">1.1</td></tr>
  <tr><td>D</td><td>0</td><td>0</td><td>0</td><td>0</td><td style="color:#f5a623;">0.9</td><td>0</td></tr>
  <tr><td>E</td><td>0</td><td>0</td><td style="color:#f5a623;">0.4</td><td>0</td><td>0</td><td>0</td></tr>
  <tr><td>F</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
</table>
<p style="color: var(--muted-color); font-size: 14pt; margin-top: 10px;">Zeros mean "no edge"; nonzeros carry edge weights.</p>
</div>
:::

The dense layer case is the same idea with every possible edge present. Sparse graphs just keep the entries that matter.

---

<!-- .slide: id="matrix-graph" -->

## Matrix-Graph Equivalence

A layer's connections **are** a weight matrix: edge weight from input $j$ to neuron $i$ is entry $W_{ij}$.

<div style="text-align:center; margin-top:6px;">
<svg viewBox="0 0 620 220" width="92%" style="max-height:300px;">
  <defs>
    <marker id="mgar" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="#8892a4"/>
    </marker>
  </defs>
  <g stroke="#2a3450" stroke-width="1.8">
    <line x1="72" y1="50" x2="190" y2="85" marker-end="url(#mgar)"/>
    <line x1="72" y1="110" x2="190" y2="88" marker-end="url(#mgar)"/>
    <line x1="72" y1="170" x2="190" y2="92" marker-end="url(#mgar)"/>
    <line x1="72" y1="50" x2="190" y2="148" marker-end="url(#mgar)"/>
    <line x1="72" y1="110" x2="190" y2="151" marker-end="url(#mgar)"/>
    <line x1="72" y1="170" x2="190" y2="154" marker-end="url(#mgar)"/>
  </g>
  <line x1="72" y1="50" x2="190" y2="85" stroke="#f5a623" stroke-width="2.6" marker-end="url(#mgar)"/>
  <g fill="#0d1225" stroke="#4a9eff" stroke-width="2.5">
    <circle cx="55" cy="50" r="17"/><circle cx="55" cy="110" r="17"/><circle cx="55" cy="170" r="17"/>
    <circle cx="208" cy="90" r="17"/><circle cx="208" cy="152" r="17"/>
  </g>
  <g fill="#e8eaf0" font-size="13" text-anchor="middle">
    <text x="55" y="55">x₁</text><text x="55" y="115">x₂</text><text x="55" y="175">x₃</text>
    <text x="208" y="95">h₁</text><text x="208" y="157">h₂</text>
  </g>
  <text x="312" y="124" fill="#8892a4" font-size="30" text-anchor="middle">≡</text>
  <path d="M412,66 L400,66 L400,176 L412,176" fill="none" stroke="#e8eaf0" stroke-width="2"/>
  <path d="M576,66 L588,66 L588,176 L576,176" fill="none" stroke="#e8eaf0" stroke-width="2"/>
  <g font-size="15" text-anchor="middle" fill="#e8eaf0">
    <text x="446" y="108" fill="#f5a623">W₁₁</text><text x="496" y="108">W₁₂</text><text x="546" y="108">W₁₃</text>
    <text x="446" y="156">W₂₁</text><text x="496" y="156">W₂₂</text><text x="546" y="156">W₂₃</text>
  </g>
  <text x="494" y="205" fill="#8892a4" font-size="13" text-anchor="middle">rows = neurons, columns = inputs</text>
</svg>
</div>

:::columns cols="2" gap="36px"
**Forward pass:** $\mathbf{h} = \sigma(W\mathbf{x} + \mathbf{b})$ <!-- .element: class="text-lg" style="color: var(--secondary-color); margin:0;" -->

One matrix multiply per layer. <!-- .element: class="text-lg" style="color: var(--muted-color); margin:4px 0 0 0;" -->
+++
**Why GPUs?** Each output is an independent dot product, so the work is massively parallel — exactly what a GPU's thousands of cores do best. <!-- .element: class="text-lg" style="margin:0;" -->
:::

---

:::columns grid="200px 1fr" gap="30px" valign="start"
<img src="images/minsky.jpg" alt="Marvin Minsky" style="border-radius: 8px; width: 100%; box-shadow: 0 4px 20px rgba(74, 158, 255, 0.15);">
<img src="images/papert.jpg" alt="Seymour Papert" style="border-radius: 8px; width: 100%; margin-top: 10px; box-shadow: 0 4px 20px rgba(74, 158, 255, 0.15);">
+++
## Minsky &amp; Papert <!-- .element: class="fragment fade-in" style="border: none; margin: 0 0 10px 0;" -->

<div class="fragment fade-in">

### Killed Neural Networks for a Decade

- Published *Perceptrons* (1969) &mdash; proved single-layer nets cannot compute XOR, parity, or connectedness
- The result was widely interpreted as showing neural networks were fundamentally limited
- Funding dried up for over a decade &mdash; the first AI winter for connectionism
- Minsky was co-founder of MIT's AI Lab and a leading proponent of symbolic AI
- **They knew multi-layer nets could solve these problems.** The critique was that no one knew how to *train* them.

</div>
:::
