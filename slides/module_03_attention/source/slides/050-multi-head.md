:::divider id="divider-multi-head" title="Multi-Head Attention" sub="Several lookup patterns running in parallel"
:::

---

<!-- .slide: id="why-multi-head" -->

## Why Multiple Heads?

A single attention head produces one weighted average per token. But a token may need to attend to different things for different reasons:

:::columns cols="2" gap="30px"
**Head 1** might learn syntactic attention &mdash; verbs attend to their subjects

**Head 2** might learn coreference &mdash; pronouns attend to their antecedents

**Head 3** might learn positional attention &mdash; each token attends to its immediate neighbor
+++
Multiple heads let the model learn **different ways to compare tokens** at the same time, instead of forcing one pattern to serve every purpose.
:::

---

<!-- .slide: id="multi-head-split" -->

## Splitting the Dimension Across Heads

The model dimension is divided among the heads, so multi-head attention costs the same as a single full-width head. Here: $d_{\text{model}} = 512$, $H = 8$ heads, $d_k = 64$ each.

<div style="text-align: center; margin: 4px 0;">
<svg viewBox="0 0 880 330" width="100%" style="max-height: 320px;">
  <!-- Band 1: split bar -->
  <g font-size="10" text-anchor="middle" font-weight="600">
    <rect x="30" y="24" width="56" height="34" fill="rgba(74,158,255,0.30)" stroke="#0a0e1a"/><text x="58" y="45" fill="#e8eaf0">64</text>
    <rect x="86" y="24" width="56" height="34" fill="rgba(245,166,35,0.30)" stroke="#0a0e1a"/><text x="114" y="45" fill="#e8eaf0">64</text>
    <rect x="142" y="24" width="56" height="34" fill="rgba(80,200,120,0.30)" stroke="#0a0e1a"/><text x="170" y="45" fill="#e8eaf0">64</text>
    <rect x="198" y="24" width="56" height="34" fill="rgba(199,146,234,0.30)" stroke="#0a0e1a"/><text x="226" y="45" fill="#e8eaf0">64</text>
    <rect x="254" y="24" width="56" height="34" fill="rgba(74,158,255,0.30)" stroke="#0a0e1a"/><text x="282" y="45" fill="#e8eaf0">64</text>
    <rect x="310" y="24" width="56" height="34" fill="rgba(245,166,35,0.30)" stroke="#0a0e1a"/><text x="338" y="45" fill="#e8eaf0">64</text>
    <rect x="366" y="24" width="56" height="34" fill="rgba(80,200,120,0.30)" stroke="#0a0e1a"/><text x="394" y="45" fill="#e8eaf0">64</text>
    <rect x="422" y="24" width="56" height="34" fill="rgba(199,146,234,0.30)" stroke="#0a0e1a"/><text x="450" y="45" fill="#e8eaf0">64</text>
  </g>
  <rect x="30" y="24" width="448" height="34" fill="none" stroke="#e8eaf0" stroke-width="1.5"/>
  <text x="520" y="45" fill="#e8eaf0" font-size="13">Q, K, V are split into 8 slices of 64 dims</text>
  <!-- arrows down -->
  <g stroke="#8892a4" stroke-width="1.2" marker-end="url(#arrmh)">
    <line x1="58" y1="58" x2="58" y2="92"/><line x1="114" y1="58" x2="114" y2="92"/><line x1="170" y1="58" x2="170" y2="92"/><line x1="226" y1="58" x2="226" y2="92"/>
    <line x1="282" y1="58" x2="282" y2="92"/><line x1="338" y1="58" x2="338" y2="92"/><line x1="394" y1="58" x2="394" y2="92"/><line x1="450" y1="58" x2="450" y2="92"/>
  </g>
  <!-- Band 2: head boxes -->
  <g font-size="10" text-anchor="middle" font-weight="600">
    <rect x="34" y="94" width="48" height="48" rx="4" fill="rgba(74,158,255,0.12)" stroke="#4a9eff"/><text x="58" y="116" fill="#4a9eff">head</text><text x="58" y="130" fill="#4a9eff">1</text>
    <rect x="90" y="94" width="48" height="48" rx="4" fill="rgba(245,166,35,0.12)" stroke="#f5a623"/><text x="114" y="116" fill="#f5a623">head</text><text x="114" y="130" fill="#f5a623">2</text>
    <rect x="146" y="94" width="48" height="48" rx="4" fill="rgba(80,200,120,0.12)" stroke="#50c878"/><text x="170" y="116" fill="#50c878">head</text><text x="170" y="130" fill="#50c878">3</text>
    <rect x="202" y="94" width="48" height="48" rx="4" fill="rgba(199,146,234,0.12)" stroke="#c792ea"/><text x="226" y="116" fill="#c792ea">head</text><text x="226" y="130" fill="#c792ea">4</text>
    <rect x="258" y="94" width="48" height="48" rx="4" fill="rgba(74,158,255,0.12)" stroke="#4a9eff"/><text x="282" y="116" fill="#4a9eff">head</text><text x="282" y="130" fill="#4a9eff">5</text>
    <rect x="314" y="94" width="48" height="48" rx="4" fill="rgba(245,166,35,0.12)" stroke="#f5a623"/><text x="338" y="116" fill="#f5a623">head</text><text x="338" y="130" fill="#f5a623">6</text>
    <rect x="370" y="94" width="48" height="48" rx="4" fill="rgba(80,200,120,0.12)" stroke="#50c878"/><text x="394" y="116" fill="#50c878">head</text><text x="394" y="130" fill="#50c878">7</text>
    <rect x="426" y="94" width="48" height="48" rx="4" fill="rgba(199,146,234,0.12)" stroke="#c792ea"/><text x="450" y="116" fill="#c792ea">head</text><text x="450" y="130" fill="#c792ea">8</text>
  </g>
  <text x="520" y="122" fill="#e8eaf0" font-size="13">each head runs attention on its own slice</text>
  <!-- arrows down -->
  <g stroke="#8892a4" stroke-width="1.2" marker-end="url(#arrmh)">
    <line x1="58" y1="142" x2="58" y2="176"/><line x1="114" y1="142" x2="114" y2="176"/><line x1="170" y1="142" x2="170" y2="176"/><line x1="226" y1="142" x2="226" y2="176"/>
    <line x1="282" y1="142" x2="282" y2="176"/><line x1="338" y1="142" x2="338" y2="176"/><line x1="394" y1="142" x2="394" y2="176"/><line x1="450" y1="142" x2="450" y2="176"/>
  </g>
  <!-- Band 3: concat bar -->
  <g font-size="10" text-anchor="middle" font-weight="600">
    <rect x="30" y="178" width="56" height="34" fill="rgba(74,158,255,0.30)" stroke="#0a0e1a"/>
    <rect x="86" y="178" width="56" height="34" fill="rgba(245,166,35,0.30)" stroke="#0a0e1a"/>
    <rect x="142" y="178" width="56" height="34" fill="rgba(80,200,120,0.30)" stroke="#0a0e1a"/>
    <rect x="198" y="178" width="56" height="34" fill="rgba(199,146,234,0.30)" stroke="#0a0e1a"/>
    <rect x="254" y="178" width="56" height="34" fill="rgba(74,158,255,0.30)" stroke="#0a0e1a"/>
    <rect x="310" y="178" width="56" height="34" fill="rgba(245,166,35,0.30)" stroke="#0a0e1a"/>
    <rect x="366" y="178" width="56" height="34" fill="rgba(80,200,120,0.30)" stroke="#0a0e1a"/>
    <rect x="422" y="178" width="56" height="34" fill="rgba(199,146,234,0.30)" stroke="#0a0e1a"/>
  </g>
  <rect x="30" y="178" width="448" height="34" fill="none" stroke="#e8eaf0" stroke-width="1.5"/>
  <text x="254" y="200" fill="#e8eaf0" font-size="11" text-anchor="middle">concat = 512</text>
  <text x="520" y="199" fill="#e8eaf0" font-size="13">concatenate the 8 outputs back to 512</text>
  <!-- arrow down -->
  <line x1="254" y1="212" x2="254" y2="246" stroke="#8892a4" stroke-width="1.2" marker-end="url(#arrmh)"/>
  <!-- Band 4: W_O + output -->
  <rect x="170" y="248" width="168" height="34" rx="5" fill="rgba(74,158,255,0.10)" stroke="#4a9eff" stroke-width="1.5"/>
  <text x="254" y="270" fill="#4a9eff" font-size="13" text-anchor="middle" font-weight="600">output projection W_O</text>
  <text x="520" y="269" fill="#e8eaf0" font-size="13">a final linear layer mixes the heads</text>
  <text x="254" y="306" fill="#3fb950" font-size="13" text-anchor="middle" font-weight="600">output: 512 dims</text>
  <defs><marker id="arrmh" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#8892a4"/></marker></defs>
</svg>
</div>

---

<!-- .slide: id="multi-head-mechanics" -->

## Multi-Head Mechanics

Each head $h$ has its own projections and computes attention independently:

$$\text{head}_h = \text{Attention}(XW_Q^h, XW_K^h, XW_V^h)$$

The heads are concatenated and projected back through $W_O$:

$$\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) W_O$$

:::note
**Compute budget:** the total projection size across all heads is kept equal to $d_{\text{model}}$. With 8 heads and $d_{\text{model}} = 512$, each head uses $d_k = 64$. Multi-head attention is not more expensive than single-head &mdash; it is the same compute, split into independent patterns.
:::
