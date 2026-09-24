:::divider id="divider-multi-head" title="Multi-Head Attention" sub="Several lookup patterns running in parallel"
:::

---

<!-- .slide: id="why-multi-head" -->

## Why Multiple Heads?

One head = one weighted average per token. Project first, then split: each head gets its own 64 dimensions of $Q$, $K$, and $V$, so several lookup patterns run in parallel.
<div style="text-align: center; margin: 4px 0;">
<svg viewBox="0 0 880 312" width="100%" style="max-height: 300px;">
  <rect x="308" y="4" width="100" height="30" rx="5" fill="rgba(232,234,240,0.10)" stroke="#e8eaf0" stroke-width="1.6"/>
  <text x="344" y="24" fill="#e8eaf0" font-size="14" text-anchor="middle" font-weight="600">X</text><text x="374" y="24" fill="#8892a4" font-size="10" text-anchor="middle">n × 512</text>
  <g stroke="#8892a4" stroke-width="1.2" marker-end="url(#arrmhfirst)">
    <line x1="358" y1="34" x2="286" y2="48"/>
    <line x1="358" y1="34" x2="358" y2="48"/>
    <line x1="358" y1="34" x2="430" y2="48"/>
    <line x1="286" y1="84" x2="286" y2="96"/>
    <line x1="358" y1="84" x2="358" y2="96"/>
    <line x1="430" y1="84" x2="430" y2="96"/>
  </g>
  <g text-anchor="middle" font-weight="600">
    <rect x="254" y="50" width="64" height="34" rx="4" fill="rgba(74,158,255,0.12)" stroke="#4a9eff" stroke-width="1.5"/><text x="286" y="65" fill="#4a9eff" font-size="12">W<tspan dy="3" font-size="9">Q</tspan></text><text x="286" y="79" fill="#8892a4" font-size="9" font-weight="400">512 × 512</text>
    <text x="286" y="108" fill="#4a9eff" font-size="11">Q <tspan fill="#8892a4" font-weight="400" font-size="10">n × 512</tspan></text>
    <rect x="326" y="50" width="64" height="34" rx="4" fill="rgba(245,166,35,0.12)" stroke="#f5a623" stroke-width="1.5"/><text x="358" y="65" fill="#f5a623" font-size="12">W<tspan dy="3" font-size="9">K</tspan></text><text x="358" y="79" fill="#8892a4" font-size="9" font-weight="400">512 × 512</text>
    <text x="358" y="108" fill="#f5a623" font-size="11">K <tspan fill="#8892a4" font-weight="400" font-size="10">n × 512</tspan></text>
    <rect x="398" y="50" width="64" height="34" rx="4" fill="rgba(80,200,120,0.12)" stroke="#50c878" stroke-width="1.5"/><text x="430" y="65" fill="#50c878" font-size="12">W<tspan dy="3" font-size="9">V</tspan></text><text x="430" y="79" fill="#8892a4" font-size="9" font-weight="400">512 × 512</text>
    <text x="430" y="108" fill="#50c878" font-size="11">V <tspan fill="#8892a4" font-weight="400" font-size="10">n × 512</tspan></text>
  </g>
  <g font-size="11" text-anchor="middle">
    <rect x="250" y="114" width="72" height="24" fill="rgba(74,158,255,0.10)" stroke="#4a9eff" stroke-width="1"/><text x="286" y="130" fill="#4a9eff">Q<tspan dy="3" font-size="8">1</tspan><tspan dy="-3" fill="#8892a4" font-size="9">  n × 64</tspan></text>
    <rect x="250" y="138" width="72" height="24" fill="rgba(74,158,255,0.10)" stroke="#4a9eff" stroke-width="1"/><text x="286" y="154" fill="#4a9eff">Q<tspan dy="3" font-size="8">2</tspan><tspan dy="-3" fill="#8892a4" font-size="9">  n × 64</tspan></text>
    <rect x="250" y="162" width="72" height="24" fill="rgba(74,158,255,0.10)" stroke="#4a9eff" stroke-width="1"/><text x="286" y="178" fill="#4a9eff">Q<tspan dy="3" font-size="8">3</tspan><tspan dy="-3" fill="#8892a4" font-size="9">  n × 64</tspan></text>
    <rect x="250" y="186" width="72" height="24" fill="rgba(74,158,255,0.10)" stroke="#4a9eff" stroke-width="1"/><text x="286" y="202" fill="#4a9eff">Q<tspan dy="3" font-size="8">4</tspan><tspan dy="-3" fill="#8892a4" font-size="9">  n × 64</tspan></text>
    <rect x="250" y="210" width="72" height="24" fill="rgba(74,158,255,0.10)" stroke="#4a9eff" stroke-width="1"/><text x="286" y="226" fill="#4a9eff">Q<tspan dy="3" font-size="8">5</tspan><tspan dy="-3" fill="#8892a4" font-size="9">  n × 64</tspan></text>
    <rect x="250" y="234" width="72" height="24" fill="rgba(74,158,255,0.10)" stroke="#4a9eff" stroke-width="1"/><text x="286" y="250" fill="#4a9eff">Q<tspan dy="3" font-size="8">6</tspan><tspan dy="-3" fill="#8892a4" font-size="9">  n × 64</tspan></text>
    <rect x="250" y="258" width="72" height="24" fill="rgba(74,158,255,0.10)" stroke="#4a9eff" stroke-width="1"/><text x="286" y="274" fill="#4a9eff">Q<tspan dy="3" font-size="8">7</tspan><tspan dy="-3" fill="#8892a4" font-size="9">  n × 64</tspan></text>
    <rect x="250" y="282" width="72" height="24" fill="rgba(74,158,255,0.10)" stroke="#4a9eff" stroke-width="1"/><text x="286" y="298" fill="#4a9eff">Q<tspan dy="3" font-size="8">8</tspan><tspan dy="-3" fill="#8892a4" font-size="9">  n × 64</tspan></text>
    <rect x="322" y="114" width="72" height="24" fill="rgba(245,166,35,0.10)" stroke="#f5a623" stroke-width="1"/><text x="358" y="130" fill="#f5a623">K<tspan dy="3" font-size="8">1</tspan><tspan dy="-3" fill="#8892a4" font-size="9">  n × 64</tspan></text>
    <rect x="322" y="138" width="72" height="24" fill="rgba(245,166,35,0.10)" stroke="#f5a623" stroke-width="1"/><text x="358" y="154" fill="#f5a623">K<tspan dy="3" font-size="8">2</tspan><tspan dy="-3" fill="#8892a4" font-size="9">  n × 64</tspan></text>
    <rect x="322" y="162" width="72" height="24" fill="rgba(245,166,35,0.10)" stroke="#f5a623" stroke-width="1"/><text x="358" y="178" fill="#f5a623">K<tspan dy="3" font-size="8">3</tspan><tspan dy="-3" fill="#8892a4" font-size="9">  n × 64</tspan></text>
    <rect x="322" y="186" width="72" height="24" fill="rgba(245,166,35,0.10)" stroke="#f5a623" stroke-width="1"/><text x="358" y="202" fill="#f5a623">K<tspan dy="3" font-size="8">4</tspan><tspan dy="-3" fill="#8892a4" font-size="9">  n × 64</tspan></text>
    <rect x="322" y="210" width="72" height="24" fill="rgba(245,166,35,0.10)" stroke="#f5a623" stroke-width="1"/><text x="358" y="226" fill="#f5a623">K<tspan dy="3" font-size="8">5</tspan><tspan dy="-3" fill="#8892a4" font-size="9">  n × 64</tspan></text>
    <rect x="322" y="234" width="72" height="24" fill="rgba(245,166,35,0.10)" stroke="#f5a623" stroke-width="1"/><text x="358" y="250" fill="#f5a623">K<tspan dy="3" font-size="8">6</tspan><tspan dy="-3" fill="#8892a4" font-size="9">  n × 64</tspan></text>
    <rect x="322" y="258" width="72" height="24" fill="rgba(245,166,35,0.10)" stroke="#f5a623" stroke-width="1"/><text x="358" y="274" fill="#f5a623">K<tspan dy="3" font-size="8">7</tspan><tspan dy="-3" fill="#8892a4" font-size="9">  n × 64</tspan></text>
    <rect x="322" y="282" width="72" height="24" fill="rgba(245,166,35,0.10)" stroke="#f5a623" stroke-width="1"/><text x="358" y="298" fill="#f5a623">K<tspan dy="3" font-size="8">8</tspan><tspan dy="-3" fill="#8892a4" font-size="9">  n × 64</tspan></text>
    <rect x="394" y="114" width="72" height="24" fill="rgba(80,200,120,0.10)" stroke="#50c878" stroke-width="1"/><text x="430" y="130" fill="#50c878">V<tspan dy="3" font-size="8">1</tspan><tspan dy="-3" fill="#8892a4" font-size="9">  n × 64</tspan></text>
    <rect x="394" y="138" width="72" height="24" fill="rgba(80,200,120,0.10)" stroke="#50c878" stroke-width="1"/><text x="430" y="154" fill="#50c878">V<tspan dy="3" font-size="8">2</tspan><tspan dy="-3" fill="#8892a4" font-size="9">  n × 64</tspan></text>
    <rect x="394" y="162" width="72" height="24" fill="rgba(80,200,120,0.10)" stroke="#50c878" stroke-width="1"/><text x="430" y="178" fill="#50c878">V<tspan dy="3" font-size="8">3</tspan><tspan dy="-3" fill="#8892a4" font-size="9">  n × 64</tspan></text>
    <rect x="394" y="186" width="72" height="24" fill="rgba(80,200,120,0.10)" stroke="#50c878" stroke-width="1"/><text x="430" y="202" fill="#50c878">V<tspan dy="3" font-size="8">4</tspan><tspan dy="-3" fill="#8892a4" font-size="9">  n × 64</tspan></text>
    <rect x="394" y="210" width="72" height="24" fill="rgba(80,200,120,0.10)" stroke="#50c878" stroke-width="1"/><text x="430" y="226" fill="#50c878">V<tspan dy="3" font-size="8">5</tspan><tspan dy="-3" fill="#8892a4" font-size="9">  n × 64</tspan></text>
    <rect x="394" y="234" width="72" height="24" fill="rgba(80,200,120,0.10)" stroke="#50c878" stroke-width="1"/><text x="430" y="250" fill="#50c878">V<tspan dy="3" font-size="8">6</tspan><tspan dy="-3" fill="#8892a4" font-size="9">  n × 64</tspan></text>
    <rect x="394" y="258" width="72" height="24" fill="rgba(80,200,120,0.10)" stroke="#50c878" stroke-width="1"/><text x="430" y="274" fill="#50c878">V<tspan dy="3" font-size="8">7</tspan><tspan dy="-3" fill="#8892a4" font-size="9">  n × 64</tspan></text>
    <rect x="394" y="282" width="72" height="24" fill="rgba(80,200,120,0.10)" stroke="#50c878" stroke-width="1"/><text x="430" y="298" fill="#50c878">V<tspan dy="3" font-size="8">8</tspan><tspan dy="-3" fill="#8892a4" font-size="9">  n × 64</tspan></text>
  </g>
  <g stroke="#8892a4" stroke-width="1.2" marker-end="url(#arrmhfirst)">
    <line x1="466" y1="126" x2="494" y2="126"/>
    <line x1="466" y1="150" x2="494" y2="150"/>
    <line x1="466" y1="174" x2="494" y2="174"/>
    <line x1="466" y1="198" x2="494" y2="198"/>
    <line x1="466" y1="222" x2="494" y2="222"/>
    <line x1="466" y1="246" x2="494" y2="246"/>
    <line x1="466" y1="270" x2="494" y2="270"/>
    <line x1="466" y1="294" x2="494" y2="294"/>
  </g>
  <g font-size="11" text-anchor="middle" font-weight="600">
    <rect x="496" y="116" width="100" height="20" rx="4" fill="#0d1225" stroke="#e8eaf0" stroke-width="1.1"/><text x="546" y="130" fill="#e8eaf0">head 1</text>
    <rect x="496" y="140" width="100" height="20" rx="4" fill="#0d1225" stroke="#e8eaf0" stroke-width="1.1"/><text x="546" y="154" fill="#e8eaf0">head 2</text>
    <rect x="496" y="164" width="100" height="20" rx="4" fill="#0d1225" stroke="#e8eaf0" stroke-width="1.1"/><text x="546" y="178" fill="#e8eaf0">head 3</text>
    <rect x="496" y="188" width="100" height="20" rx="4" fill="#0d1225" stroke="#e8eaf0" stroke-width="1.1"/><text x="546" y="202" fill="#e8eaf0">head 4</text>
    <rect x="496" y="212" width="100" height="20" rx="4" fill="#0d1225" stroke="#e8eaf0" stroke-width="1.1"/><text x="546" y="226" fill="#e8eaf0">head 5</text>
    <rect x="496" y="236" width="100" height="20" rx="4" fill="#0d1225" stroke="#e8eaf0" stroke-width="1.1"/><text x="546" y="250" fill="#e8eaf0">head 6</text>
    <rect x="496" y="260" width="100" height="20" rx="4" fill="#0d1225" stroke="#e8eaf0" stroke-width="1.1"/><text x="546" y="274" fill="#e8eaf0">head 7</text>
    <rect x="496" y="284" width="100" height="20" rx="4" fill="#0d1225" stroke="#e8eaf0" stroke-width="1.1"/><text x="546" y="298" fill="#e8eaf0">head 8</text>
  </g>
  <g font-size="12" text-anchor="end">
    <text x="228" y="24" fill="#8892a4">input tokens</text>
    <text x="228" y="60" fill="#e8eaf0">1. project with three matrices</text>
    <text x="228" y="76" fill="#8892a4">same W's as single-head</text>
    <text x="228" y="202" fill="#e8eaf0">2. cut each into 8 slices</text>
    <text x="228" y="218" fill="#8892a4">64 of the 512 columns each</text>
  </g>
  <g font-size="12" text-anchor="start">
    <text x="626" y="186" fill="#e8eaf0">3. head h attends with</text>
    <text x="626" y="202" fill="#e8eaf0">its own row: Q<tspan dy="3" font-size="9">h</tspan><tspan dy="-3">, K</tspan><tspan dy="3" font-size="9">h</tspan><tspan dy="-3">, V</tspan><tspan dy="3" font-size="9">h</tspan></text>
    <text x="626" y="226" fill="#8892a4">each slice is computed</text>
    <text x="626" y="242" fill="#8892a4">from all 512 features of X</text>
  </g>
  <defs><marker id="arrmhfirst" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#8892a4"/></marker></defs>
</svg>
</div>

:::columns cols="2" gap="30px"
**Different comparisons**

One head tracks subject-verb links, another coreference, another nearby tokens.
+++
**Same compute budget**

The total width stays 512, divided across heads rather than copied per head.
:::

---

<!-- .slide: id="multi-head-mechanics" -->

## Multi-Head Mechanics

<div style="text-align: center; margin: 2px 0;">
<svg viewBox="0 0 880 328" width="100%" style="max-height: 330px;">
  <rect x="148" y="4" width="100" height="30" rx="5" fill="rgba(232,234,240,0.10)" stroke="#e8eaf0" stroke-width="1.6"/>
  <text x="184" y="24" fill="#e8eaf0" font-size="14" text-anchor="middle" font-weight="600">X</text><text x="214" y="24" fill="#8892a4" font-size="10" text-anchor="middle">n × 512</text>
  <g stroke="#8892a4" stroke-width="1.2" marker-end="url(#arrmhfull)">
    <line x1="198" y1="34" x2="126" y2="48"/>
    <line x1="198" y1="34" x2="198" y2="48"/>
    <line x1="198" y1="34" x2="270" y2="48"/>
    <line x1="126" y1="84" x2="126" y2="96"/>
    <line x1="198" y1="84" x2="198" y2="96"/>
    <line x1="270" y1="84" x2="270" y2="96"/>
  </g>
  <g text-anchor="middle" font-weight="600">
    <rect x="94" y="50" width="64" height="34" rx="4" fill="rgba(74,158,255,0.12)" stroke="#4a9eff" stroke-width="1.5"/><text x="126" y="65" fill="#4a9eff" font-size="12">W<tspan dy="3" font-size="9">Q</tspan></text><text x="126" y="79" fill="#8892a4" font-size="9" font-weight="400">512 × 512</text>
    <text x="126" y="108" fill="#4a9eff" font-size="11">Q <tspan fill="#8892a4" font-weight="400" font-size="10">n × 512</tspan></text>
    <rect x="166" y="50" width="64" height="34" rx="4" fill="rgba(245,166,35,0.12)" stroke="#f5a623" stroke-width="1.5"/><text x="198" y="65" fill="#f5a623" font-size="12">W<tspan dy="3" font-size="9">K</tspan></text><text x="198" y="79" fill="#8892a4" font-size="9" font-weight="400">512 × 512</text>
    <text x="198" y="108" fill="#f5a623" font-size="11">K <tspan fill="#8892a4" font-weight="400" font-size="10">n × 512</tspan></text>
    <rect x="238" y="50" width="64" height="34" rx="4" fill="rgba(80,200,120,0.12)" stroke="#50c878" stroke-width="1.5"/><text x="270" y="65" fill="#50c878" font-size="12">W<tspan dy="3" font-size="9">V</tspan></text><text x="270" y="79" fill="#8892a4" font-size="9" font-weight="400">512 × 512</text>
    <text x="270" y="108" fill="#50c878" font-size="11">V <tspan fill="#8892a4" font-weight="400" font-size="10">n × 512</tspan></text>
  </g>
  <g font-size="11" text-anchor="middle">
    <rect x="90" y="114" width="72" height="24" fill="rgba(74,158,255,0.10)" stroke="#4a9eff" stroke-width="1"/><text x="126" y="130" fill="#4a9eff">Q<tspan dy="3" font-size="8">1</tspan><tspan dy="-3" fill="#8892a4" font-size="9">  n × 64</tspan></text>
    <rect x="90" y="138" width="72" height="24" fill="rgba(74,158,255,0.10)" stroke="#4a9eff" stroke-width="1"/><text x="126" y="154" fill="#4a9eff">Q<tspan dy="3" font-size="8">2</tspan><tspan dy="-3" fill="#8892a4" font-size="9">  n × 64</tspan></text>
    <rect x="90" y="162" width="72" height="24" fill="rgba(74,158,255,0.10)" stroke="#4a9eff" stroke-width="1"/><text x="126" y="178" fill="#4a9eff">Q<tspan dy="3" font-size="8">3</tspan><tspan dy="-3" fill="#8892a4" font-size="9">  n × 64</tspan></text>
    <rect x="90" y="186" width="72" height="24" fill="rgba(74,158,255,0.10)" stroke="#4a9eff" stroke-width="1"/><text x="126" y="202" fill="#4a9eff">Q<tspan dy="3" font-size="8">4</tspan><tspan dy="-3" fill="#8892a4" font-size="9">  n × 64</tspan></text>
    <rect x="90" y="210" width="72" height="24" fill="rgba(74,158,255,0.10)" stroke="#4a9eff" stroke-width="1"/><text x="126" y="226" fill="#4a9eff">Q<tspan dy="3" font-size="8">5</tspan><tspan dy="-3" fill="#8892a4" font-size="9">  n × 64</tspan></text>
    <rect x="90" y="234" width="72" height="24" fill="rgba(74,158,255,0.10)" stroke="#4a9eff" stroke-width="1"/><text x="126" y="250" fill="#4a9eff">Q<tspan dy="3" font-size="8">6</tspan><tspan dy="-3" fill="#8892a4" font-size="9">  n × 64</tspan></text>
    <rect x="90" y="258" width="72" height="24" fill="rgba(74,158,255,0.10)" stroke="#4a9eff" stroke-width="1"/><text x="126" y="274" fill="#4a9eff">Q<tspan dy="3" font-size="8">7</tspan><tspan dy="-3" fill="#8892a4" font-size="9">  n × 64</tspan></text>
    <rect x="90" y="282" width="72" height="24" fill="rgba(74,158,255,0.10)" stroke="#4a9eff" stroke-width="1"/><text x="126" y="298" fill="#4a9eff">Q<tspan dy="3" font-size="8">8</tspan><tspan dy="-3" fill="#8892a4" font-size="9">  n × 64</tspan></text>
    <rect x="162" y="114" width="72" height="24" fill="rgba(245,166,35,0.10)" stroke="#f5a623" stroke-width="1"/><text x="198" y="130" fill="#f5a623">K<tspan dy="3" font-size="8">1</tspan><tspan dy="-3" fill="#8892a4" font-size="9">  n × 64</tspan></text>
    <rect x="162" y="138" width="72" height="24" fill="rgba(245,166,35,0.10)" stroke="#f5a623" stroke-width="1"/><text x="198" y="154" fill="#f5a623">K<tspan dy="3" font-size="8">2</tspan><tspan dy="-3" fill="#8892a4" font-size="9">  n × 64</tspan></text>
    <rect x="162" y="162" width="72" height="24" fill="rgba(245,166,35,0.10)" stroke="#f5a623" stroke-width="1"/><text x="198" y="178" fill="#f5a623">K<tspan dy="3" font-size="8">3</tspan><tspan dy="-3" fill="#8892a4" font-size="9">  n × 64</tspan></text>
    <rect x="162" y="186" width="72" height="24" fill="rgba(245,166,35,0.10)" stroke="#f5a623" stroke-width="1"/><text x="198" y="202" fill="#f5a623">K<tspan dy="3" font-size="8">4</tspan><tspan dy="-3" fill="#8892a4" font-size="9">  n × 64</tspan></text>
    <rect x="162" y="210" width="72" height="24" fill="rgba(245,166,35,0.10)" stroke="#f5a623" stroke-width="1"/><text x="198" y="226" fill="#f5a623">K<tspan dy="3" font-size="8">5</tspan><tspan dy="-3" fill="#8892a4" font-size="9">  n × 64</tspan></text>
    <rect x="162" y="234" width="72" height="24" fill="rgba(245,166,35,0.10)" stroke="#f5a623" stroke-width="1"/><text x="198" y="250" fill="#f5a623">K<tspan dy="3" font-size="8">6</tspan><tspan dy="-3" fill="#8892a4" font-size="9">  n × 64</tspan></text>
    <rect x="162" y="258" width="72" height="24" fill="rgba(245,166,35,0.10)" stroke="#f5a623" stroke-width="1"/><text x="198" y="274" fill="#f5a623">K<tspan dy="3" font-size="8">7</tspan><tspan dy="-3" fill="#8892a4" font-size="9">  n × 64</tspan></text>
    <rect x="162" y="282" width="72" height="24" fill="rgba(245,166,35,0.10)" stroke="#f5a623" stroke-width="1"/><text x="198" y="298" fill="#f5a623">K<tspan dy="3" font-size="8">8</tspan><tspan dy="-3" fill="#8892a4" font-size="9">  n × 64</tspan></text>
    <rect x="234" y="114" width="72" height="24" fill="rgba(80,200,120,0.10)" stroke="#50c878" stroke-width="1"/><text x="270" y="130" fill="#50c878">V<tspan dy="3" font-size="8">1</tspan><tspan dy="-3" fill="#8892a4" font-size="9">  n × 64</tspan></text>
    <rect x="234" y="138" width="72" height="24" fill="rgba(80,200,120,0.10)" stroke="#50c878" stroke-width="1"/><text x="270" y="154" fill="#50c878">V<tspan dy="3" font-size="8">2</tspan><tspan dy="-3" fill="#8892a4" font-size="9">  n × 64</tspan></text>
    <rect x="234" y="162" width="72" height="24" fill="rgba(80,200,120,0.10)" stroke="#50c878" stroke-width="1"/><text x="270" y="178" fill="#50c878">V<tspan dy="3" font-size="8">3</tspan><tspan dy="-3" fill="#8892a4" font-size="9">  n × 64</tspan></text>
    <rect x="234" y="186" width="72" height="24" fill="rgba(80,200,120,0.10)" stroke="#50c878" stroke-width="1"/><text x="270" y="202" fill="#50c878">V<tspan dy="3" font-size="8">4</tspan><tspan dy="-3" fill="#8892a4" font-size="9">  n × 64</tspan></text>
    <rect x="234" y="210" width="72" height="24" fill="rgba(80,200,120,0.10)" stroke="#50c878" stroke-width="1"/><text x="270" y="226" fill="#50c878">V<tspan dy="3" font-size="8">5</tspan><tspan dy="-3" fill="#8892a4" font-size="9">  n × 64</tspan></text>
    <rect x="234" y="234" width="72" height="24" fill="rgba(80,200,120,0.10)" stroke="#50c878" stroke-width="1"/><text x="270" y="250" fill="#50c878">V<tspan dy="3" font-size="8">6</tspan><tspan dy="-3" fill="#8892a4" font-size="9">  n × 64</tspan></text>
    <rect x="234" y="258" width="72" height="24" fill="rgba(80,200,120,0.10)" stroke="#50c878" stroke-width="1"/><text x="270" y="274" fill="#50c878">V<tspan dy="3" font-size="8">7</tspan><tspan dy="-3" fill="#8892a4" font-size="9">  n × 64</tspan></text>
    <rect x="234" y="282" width="72" height="24" fill="rgba(80,200,120,0.10)" stroke="#50c878" stroke-width="1"/><text x="270" y="298" fill="#50c878">V<tspan dy="3" font-size="8">8</tspan><tspan dy="-3" fill="#8892a4" font-size="9">  n × 64</tspan></text>
  </g>
  <g fill="#8892a4" font-size="11" text-anchor="middle">
    <text x="381" y="106">softmax(QKᵀ/√d<tspan dy="3" font-size="8">k</tspan><tspan dy="-3">)V</tspan></text>
    <text x="476" y="106">head outputs</text>
    <text x="591" y="106">concat</text>
    <text x="591" y="322">n × 512</text>
  </g>
  <g stroke="#8892a4" stroke-width="1.2" marker-end="url(#arrmhfull)">
    <line x1="306" y1="126" x2="334" y2="126"/>
    <line x1="306" y1="150" x2="334" y2="150"/>
    <line x1="306" y1="174" x2="334" y2="174"/>
    <line x1="306" y1="198" x2="334" y2="198"/>
    <line x1="306" y1="222" x2="334" y2="222"/>
    <line x1="306" y1="246" x2="334" y2="246"/>
    <line x1="306" y1="270" x2="334" y2="270"/>
    <line x1="306" y1="294" x2="334" y2="294"/>
  </g>
  <g font-size="11" text-anchor="middle" font-weight="600">
    <rect x="336" y="116" width="90" height="20" rx="4" fill="#0d1225" stroke="#e8eaf0" stroke-width="1.1"/><text x="381" y="130" fill="#e8eaf0">head 1</text>
    <rect x="336" y="140" width="90" height="20" rx="4" fill="#0d1225" stroke="#e8eaf0" stroke-width="1.1"/><text x="381" y="154" fill="#e8eaf0">head 2</text>
    <rect x="336" y="164" width="90" height="20" rx="4" fill="#0d1225" stroke="#e8eaf0" stroke-width="1.1"/><text x="381" y="178" fill="#e8eaf0">head 3</text>
    <rect x="336" y="188" width="90" height="20" rx="4" fill="#0d1225" stroke="#e8eaf0" stroke-width="1.1"/><text x="381" y="202" fill="#e8eaf0">head 4</text>
    <rect x="336" y="212" width="90" height="20" rx="4" fill="#0d1225" stroke="#e8eaf0" stroke-width="1.1"/><text x="381" y="226" fill="#e8eaf0">head 5</text>
    <rect x="336" y="236" width="90" height="20" rx="4" fill="#0d1225" stroke="#e8eaf0" stroke-width="1.1"/><text x="381" y="250" fill="#e8eaf0">head 6</text>
    <rect x="336" y="260" width="90" height="20" rx="4" fill="#0d1225" stroke="#e8eaf0" stroke-width="1.1"/><text x="381" y="274" fill="#e8eaf0">head 7</text>
    <rect x="336" y="284" width="90" height="20" rx="4" fill="#0d1225" stroke="#e8eaf0" stroke-width="1.1"/><text x="381" y="298" fill="#e8eaf0">head 8</text>
  </g>
  <g stroke="#8892a4" stroke-width="1.2" marker-end="url(#arrmhfull)">
    <line x1="426" y1="126" x2="444" y2="126"/>
    <line x1="426" y1="150" x2="444" y2="150"/>
    <line x1="426" y1="174" x2="444" y2="174"/>
    <line x1="426" y1="198" x2="444" y2="198"/>
    <line x1="426" y1="222" x2="444" y2="222"/>
    <line x1="426" y1="246" x2="444" y2="246"/>
    <line x1="426" y1="270" x2="444" y2="270"/>
    <line x1="426" y1="294" x2="444" y2="294"/>
  </g>
  <g font-size="10" text-anchor="middle" fill="#e8eaf0">
    <rect x="446" y="117" width="60" height="18" rx="3" fill="rgba(232,234,240,0.12)"/><text x="476" y="130">n × 64</text>
    <rect x="446" y="141" width="60" height="18" rx="3" fill="rgba(232,234,240,0.12)"/><text x="476" y="154">n × 64</text>
    <rect x="446" y="165" width="60" height="18" rx="3" fill="rgba(232,234,240,0.12)"/><text x="476" y="178">n × 64</text>
    <rect x="446" y="189" width="60" height="18" rx="3" fill="rgba(232,234,240,0.12)"/><text x="476" y="202">n × 64</text>
    <rect x="446" y="213" width="60" height="18" rx="3" fill="rgba(232,234,240,0.12)"/><text x="476" y="226">n × 64</text>
    <rect x="446" y="237" width="60" height="18" rx="3" fill="rgba(232,234,240,0.12)"/><text x="476" y="250">n × 64</text>
    <rect x="446" y="261" width="60" height="18" rx="3" fill="rgba(232,234,240,0.12)"/><text x="476" y="274">n × 64</text>
    <rect x="446" y="285" width="60" height="18" rx="3" fill="rgba(232,234,240,0.12)"/><text x="476" y="298">n × 64</text>
  </g>
  <g stroke="#8892a4" stroke-width="1.2" marker-end="url(#arrmhfull)">
    <line x1="506" y1="126" x2="569" y2="126"/>
    <line x1="506" y1="150" x2="569" y2="150"/>
    <line x1="506" y1="174" x2="569" y2="174"/>
    <line x1="506" y1="198" x2="569" y2="198"/>
    <line x1="506" y1="222" x2="569" y2="222"/>
    <line x1="506" y1="246" x2="569" y2="246"/>
    <line x1="506" y1="270" x2="569" y2="270"/>
    <line x1="506" y1="294" x2="569" y2="294"/>
  </g>
  <g stroke="#0a0e1a" stroke-width="1">
    <rect x="571" y="114" width="40" height="24" fill="rgba(232,234,240,0.18)"/>
    <rect x="571" y="138" width="40" height="24" fill="rgba(232,234,240,0.1)"/>
    <rect x="571" y="162" width="40" height="24" fill="rgba(232,234,240,0.18)"/>
    <rect x="571" y="186" width="40" height="24" fill="rgba(232,234,240,0.1)"/>
    <rect x="571" y="210" width="40" height="24" fill="rgba(232,234,240,0.18)"/>
    <rect x="571" y="234" width="40" height="24" fill="rgba(232,234,240,0.1)"/>
    <rect x="571" y="258" width="40" height="24" fill="rgba(232,234,240,0.18)"/>
    <rect x="571" y="282" width="40" height="24" fill="rgba(232,234,240,0.1)"/>
  </g>
  <rect x="571" y="114" width="40" height="192" fill="none" stroke="#e8eaf0" stroke-width="1.4"/>
  <g stroke="#8892a4" stroke-width="1.2" marker-end="url(#arrmhfull)">
    <line x1="611" y1="210" x2="644" y2="210"/>
    <line x1="736" y1="210" x2="774" y2="210"/>
  </g>
  <rect x="646" y="182" width="90" height="56" rx="5" fill="rgba(199,146,234,0.12)" stroke="#c792ea" stroke-width="1.8"/>
  <text x="691" y="208" fill="#c792ea" font-size="15" text-anchor="middle" font-weight="600">W<tspan dy="4" font-size="11">O</tspan></text>
  <text x="691" y="226" fill="#8892a4" font-size="11" text-anchor="middle">512 × 512</text>
  <text x="691" y="262" fill="#8892a4" font-size="11" text-anchor="middle">mixes information</text>
  <text x="691" y="276" fill="#8892a4" font-size="11" text-anchor="middle">across heads</text>
  <rect x="776" y="185" width="80" height="50" rx="5" fill="rgba(232,234,240,0.10)" stroke="#e8eaf0" stroke-width="1.6"/>
  <text x="816" y="207" fill="#e8eaf0" font-size="13" text-anchor="middle" font-weight="600">output</text>
  <text x="816" y="225" fill="#8892a4" font-size="11" text-anchor="middle">n × 512</text>
  <defs><marker id="arrmhfull" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#8892a4"/></marker></defs>
</svg>
</div>

Slice $h$ of $Q$, $K$, $V$ is head $h$; $W_Q^h$ is the matching 64-column block of $W_Q$. $W_O$ mixes the concatenated heads:

$$\begin{aligned} \text{head}_h &= \text{Attention}(XW_Q^h, XW_K^h, XW_V^h) \cr \text{MultiHead}(X) &= \text{Concat}(\text{head}_1, \ldots, \text{head}_H) W_O \end{aligned}$$
