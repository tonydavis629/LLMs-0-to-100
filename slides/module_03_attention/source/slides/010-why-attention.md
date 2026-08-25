:::divider id="divider-why-attention" title="Why Attention?" sub="From fixed windows to selective retrieval"
:::

---

<!-- .slide: id="mlp-failure" -->

## Predicting the Next Word with an MLP

Task: predict the next word. The MLP approach:

- read a **fixed window** of recent tokens
- flatten them into one vector
- map that to a distribution over the vocabulary
<div style="text-align: center; margin: 8px 0;">
<svg viewBox="0 0 860 260" width="100%" style="max-height: 250px;">
  <!-- window slots -->
  <text x="30" y="28" fill="#8892a4" font-size="13">Fixed window of 4 tokens</text>
  <g font-size="14" font-weight="600" text-anchor="middle">
    <rect x="30" y="40" width="90" height="38" rx="5" fill="#0d1225" stroke="#4a9eff" stroke-width="2"/>
    <text x="75" y="64" fill="#e8eaf0">cat</text>
    <rect x="135" y="40" width="90" height="38" rx="5" fill="#0d1225" stroke="#50c878" stroke-width="2"/>
    <text x="180" y="64" fill="#e8eaf0">sat</text>
    <rect x="240" y="40" width="90" height="38" rx="5" fill="#0d1225" stroke="#f5a623" stroke-width="2"/>
    <text x="285" y="64" fill="#e8eaf0">on</text>
    <rect x="345" y="40" width="90" height="38" rx="5" fill="#0d1225" stroke="#c792ea" stroke-width="2"/>
    <text x="390" y="64" fill="#e8eaf0">the</text>
  </g>
  <g font-size="11" fill="#8892a4" text-anchor="middle">
    <text x="75" y="95">slot 1 &middot; W&#8321;</text>
    <text x="180" y="95">slot 2 &middot; W&#8322;</text>
    <text x="285" y="95">slot 3 &middot; W&#8323;</text>
    <text x="390" y="95">slot 4 &middot; W&#8324;</text>
  </g>
  <!-- arrows to hidden -->
  <line x1="75" y1="100" x2="232" y2="150" stroke="#4a9eff" stroke-width="1.5"/>
  <line x1="180" y1="100" x2="232" y2="150" stroke="#50c878" stroke-width="1.5"/>
  <line x1="285" y1="100" x2="232" y2="150" stroke="#f5a623" stroke-width="1.5"/>
  <line x1="390" y1="100" x2="232" y2="150" stroke="#c792ea" stroke-width="1.5"/>
  <rect x="150" y="150" width="165" height="34" rx="5" fill="rgba(74,158,255,0.10)" stroke="#4a9eff" stroke-width="1.5"/>
  <text x="232" y="172" fill="#e8eaf0" font-size="13" text-anchor="middle">hidden layer</text>
  <line x1="315" y1="167" x2="360" y2="167" stroke="#8892a4" stroke-width="1.5" marker-end="url(#arrm)"/>
  <rect x="360" y="150" width="150" height="34" rx="5" fill="#0d1225" stroke="#8892a4" stroke-width="1.5"/>
  <text x="435" y="172" fill="#e8eaf0" font-size="13" text-anchor="middle">softmax over vocab</text>
  <line x1="510" y1="167" x2="560" y2="167" stroke="#8892a4" stroke-width="1.5" marker-end="url(#arrm)"/>
  <rect x="560" y="148" width="100" height="38" rx="5" fill="#0d1225" stroke="#3fb950" stroke-width="2.5"/>
  <text x="610" y="172" fill="#3fb950" font-size="15" text-anchor="middle" font-weight="600">mat</text>
  <text x="610" y="135" fill="#8892a4" font-size="12" text-anchor="middle">prediction</text>
  <defs>
    <marker id="arrm" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
      <path d="M0,0 L7,3.5 L0,7 Z" fill="#8892a4"/>
    </marker>
  </defs>
</svg>
</div>

Each slot has its own weight block. Language does not fit a fixed grid of positions.

---

<!-- .slide: id="mlp-rigid-positions" -->

:::interactive id="mlp-rigid" widget="mlpRigid" title="Rigid Positionality: Weights Are Bound to Slots"
:::

---

<!-- .slide: id="mlp-two-failures" -->

## Two Things the MLP Cannot Do

:::columns cols="2" gap="30px"
**Rigid positionality**

- Each slot has its own weights
- "the" in slot 1 and "the" in slot 4 use different parameters
- Anything learned at one position must be relearned at every other
+++
**No selective retrieval**

- To predict "mat", the useful context is "sat" and "cat"
- The MLP treats every slot equally
- No mechanism for one token to pull information from another
:::

:::note
Language needs variable-length context, weight sharing across positions, and selective retrieval. A fixed-window MLP provides none of these.
:::

---

<!-- .slide: id="fixed-context-bottleneck" -->

## The Fixed-Context Bottleneck

Recurrent models (RNNs, LSTMs) read tokens one at a time and compress everything seen so far into a **single hidden state vector**.
<div style="text-align: center; margin: 8px 0;">
<svg viewBox="0 0 820 150" width="100%" style="max-height: 140px;">
  <g font-size="13" text-anchor="middle" font-weight="600">
    <rect x="20" y="50" width="70" height="34" rx="4" fill="#0d1225" stroke="#4a9eff" stroke-width="1.5"/><text x="55" y="72" fill="#e8eaf0">the</text>
    <rect x="110" y="50" width="70" height="34" rx="4" fill="#0d1225" stroke="#4a9eff" stroke-width="1.5"/><text x="145" y="72" fill="#e8eaf0">cat</text>
    <rect x="200" y="50" width="70" height="34" rx="4" fill="#0d1225" stroke="#4a9eff" stroke-width="1.5"/><text x="235" y="72" fill="#e8eaf0">sat</text>
    <rect x="290" y="50" width="70" height="34" rx="4" fill="#0d1225" stroke="#4a9eff" stroke-width="1.5"/><text x="325" y="72" fill="#e8eaf0">on</text>
    <rect x="380" y="50" width="70" height="34" rx="4" fill="#0d1225" stroke="#4a9eff" stroke-width="1.5"/><text x="415" y="72" fill="#e8eaf0">the</text>
    <rect x="470" y="50" width="70" height="34" rx="4" fill="#0d1225" stroke="#4a9eff" stroke-width="1.5"/><text x="505" y="72" fill="#e8eaf0">mat</text>
  </g>
  <g stroke="#8892a4" stroke-width="1.5" marker-end="url(#arrb)">
    <line x1="90" y1="67" x2="108" y2="67"/><line x1="180" y1="67" x2="198" y2="67"/>
    <line x1="270" y1="67" x2="288" y2="67"/><line x1="360" y1="67" x2="378" y2="67"/>
    <line x1="450" y1="67" x2="468" y2="67"/><line x1="540" y1="67" x2="600" y2="67"/>
  </g>
  <rect x="600" y="46" width="170" height="42" rx="6" fill="rgba(231,76,60,0.10)" stroke="#e74c3c" stroke-width="2"/>
  <text x="685" y="64" fill="#e8eaf0" font-size="12" text-anchor="middle">one fixed-size</text>
  <text x="685" y="80" fill="#e8eaf0" font-size="12" text-anchor="middle">hidden state</text>
  <text x="685" y="118" fill="#e74c3c" font-size="12" text-anchor="middle">everything squeezed through here</text>
  <defs><marker id="arrb" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#8892a4"/></marker></defs>
</svg>
</div>

:::note
**Problem:**

- Any input length must fit through one fixed-size bottleneck
- The model decides what to keep before it knows what it will need
- Early tokens fade as the state is overwritten
:::

---

:::figure img="images/bahdanau_cho_bengio.jpg" name="Bahdanau, Cho &amp; Bengio" kicker="Made Attention a Central Mechanism"
- Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio (2014)
- "Neural Machine Translation by Jointly Learning to Align and Translate"
- Their translation model learned to **align** each output word to the most relevant input words
- Instead of one compressed vector, the decoder could look back at every input representation directly
- Attention began as a fix for the fixed-size bottleneck, then became a first-class mechanism
:::

---

<!-- .slide: id="attention-concept-viz" -->

## What Attention Does

Every word builds its output by **pulling information from the other words**. Click a word to see what it attends to. Thicker arrow = more of that word's value flows in.

<div class="interactive-host" data-widget="attentionArrows"></div>

:::note
Attention does not pick one token. Each token takes a weighted average over all the others; training decides which weights grow large.
:::
