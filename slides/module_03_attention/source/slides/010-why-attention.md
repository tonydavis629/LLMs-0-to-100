:::divider id="divider-why-attention" title="Why Attention?" sub="From fixed windows to selective retrieval"
:::

---

<!-- .slide: id="mlp-failure" -->

## Predicting the Next Word with an MLP

The task: given the words so far, predict the next one. An MLP reads a **fixed window** of the most recent tokens, flattens them into one long vector, and maps that to a distribution over the vocabulary.

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

Each slot has its own weight block. The window works for this sentence, but the structure of language does not fit a fixed grid of positions.

---

<!-- .slide: id="mlp-rigid-positions" -->

:::interactive id="mlp-rigid" widget="mlpRigid" title="Rigid Positionality: Weights Are Bound to Slots"
:::

---

<!-- .slide: id="mlp-two-failures" -->

## Two Things the MLP Cannot Do

:::columns cols="2" gap="30px"
**Rigid positionality**

Each slot connects to its own weights. The word "the" in slot 1 and "the" in slot 4 are processed by completely different parameters. Whatever the model learns about "the" at one position must be relearned from scratch at every other position.
+++
**No selective retrieval**

To predict "mat", the useful context is "sat" and "cat". But the MLP flattens the whole window into one vector and treats every slot equally. There is no mechanism for one token to reach out and pull information from another.
:::

:::note
The architecture must match the data. Language needs variable-length context, weight sharing across positions, and a way to selectively retrieve information. An MLP over a fixed window provides none of these.
:::

---

<!-- .slide: id="fixed-context-bottleneck" -->

## The Fixed-Context Bottleneck

Recurrent models (RNNs, LSTMs) handle variable length by reading tokens one at a time and compressing everything seen so far into a **single hidden state vector**.

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
**Problem:** no matter how long the input, all of it must fit through one fixed-size bottleneck. The model has to decide what to keep before it knows what will be needed later. Early tokens fade as the state is overwritten.
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

Before any math, here is the idea: every word looks at every other word and decides which ones matter for understanding itself. The thicker the line, the stronger the attention.

<div style="text-align: center; margin: 10px 0;">
<svg viewBox="0 0 700 220" width="100%" style="max-height: 210px;">
  <g transform="translate(0, 30)">
    <rect x="30" y="0" width="70" height="34" rx="4" fill="#0d1225" stroke="#4a9eff" stroke-width="2"/>
    <text x="65" y="22" fill="#e8eaf0" font-size="13" text-anchor="middle" font-weight="600">the</text>
    <rect x="120" y="0" width="70" height="34" rx="4" fill="#0d1225" stroke="#4a9eff" stroke-width="2"/>
    <text x="155" y="22" fill="#e8eaf0" font-size="13" text-anchor="middle" font-weight="600">cat</text>
    <rect x="210" y="0" width="70" height="34" rx="4" fill="#0d1225" stroke="#f5a623" stroke-width="2.5"/>
    <text x="245" y="22" fill="#e8eaf0" font-size="13" text-anchor="middle" font-weight="600">sat</text>
    <rect x="300" y="0" width="70" height="34" rx="4" fill="#0d1225" stroke="#4a9eff" stroke-width="2"/>
    <text x="335" y="22" fill="#e8eaf0" font-size="13" text-anchor="middle" font-weight="600">on</text>
    <rect x="390" y="0" width="70" height="34" rx="4" fill="#0d1225" stroke="#4a9eff" stroke-width="2"/>
    <text x="425" y="22" fill="#e8eaf0" font-size="13" text-anchor="middle" font-weight="600">the</text>
    <rect x="480" y="0" width="70" height="34" rx="4" fill="#0d1225" stroke="#4a9eff" stroke-width="2"/>
    <text x="515" y="22" fill="#e8eaf0" font-size="13" text-anchor="middle" font-weight="600">mat</text>
  </g>
  <line x1="245" y1="65" x2="65" y2="65" stroke="#4a9eff" stroke-width="2" opacity="0.35"/>
  <line x1="245" y1="65" x2="155" y2="65" stroke="#f5a623" stroke-width="5" opacity="0.8"/>
  <line x1="245" y1="65" x2="335" y2="65" stroke="#f5a623" stroke-width="4" opacity="0.7"/>
  <line x1="245" y1="65" x2="425" y2="65" stroke="#4a9eff" stroke-width="1.5" opacity="0.3"/>
  <line x1="245" y1="65" x2="515" y2="65" stroke="#4a9eff" stroke-width="1.5" opacity="0.3"/>
  <text x="350" y="115" fill="#8892a4" font-size="13" text-anchor="middle">"sat" attends strongly to "cat" and "on"</text>
  <g transform="translate(0, 135)">
    <rect x="30" y="0" width="70" height="34" rx="4" fill="#0d1225" stroke="#4a9eff" stroke-width="2"/>
    <text x="65" y="22" fill="#e8eaf0" font-size="13" text-anchor="middle" font-weight="600">the</text>
    <rect x="120" y="0" width="70" height="34" rx="4" fill="#0d1225" stroke="#4a9eff" stroke-width="2"/>
    <text x="155" y="22" fill="#e8eaf0" font-size="13" text-anchor="middle" font-weight="600">cat</text>
    <rect x="210" y="0" width="70" height="34" rx="4" fill="#0d1225" stroke="#4a9eff" stroke-width="2"/>
    <text x="245" y="22" fill="#e8eaf0" font-size="13" text-anchor="middle" font-weight="600">was</text>
    <rect x="300" y="0" width="70" height="34" rx="4" fill="#0d1225" stroke="#4a9eff" stroke-width="2"/>
    <text x="335" y="22" fill="#e8eaf0" font-size="13" text-anchor="middle" font-weight="600">tired</text>
    <rect x="390" y="0" width="70" height="34" rx="4" fill="#0d1225" stroke="#f5a623" stroke-width="2.5"/>
    <text x="425" y="22" fill="#e8eaf0" font-size="13" text-anchor="middle" font-weight="600">it</text>
    <rect x="480" y="0" width="70" height="34" rx="4" fill="#0d1225" stroke="#4a9eff" stroke-width="2"/>
    <text x="515" y="22" fill="#e8eaf0" font-size="13" text-anchor="middle" font-weight="600">slept</text>
  </g>
  <line x1="425" y1="170" x2="155" y2="170" stroke="#f5a623" stroke-width="5" opacity="0.8"/>
  <line x1="425" y1="170" x2="65" y2="170" stroke="#4a9eff" stroke-width="1.5" opacity="0.3"/>
  <line x1="425" y1="170" x2="245" y2="170" stroke="#4a9eff" stroke-width="1.5" opacity="0.3"/>
  <line x1="425" y1="170" x2="335" y2="170" stroke="#4a9eff" stroke-width="1.5" opacity="0.3"/>
  <line x1="425" y1="170" x2="515" y2="170" stroke="#4a9eff" stroke-width="2" opacity="0.4"/>
  <text x="350" y="210" fill="#8892a4" font-size="13" text-anchor="middle">"it" attends back to "cat" (coreference)</text>
</svg>
</div>

:::note
**The key insight:** each token chooses its own context. A verb looks for its subject; a pronoun looks for its antecedent. The connections are learned, not hand-coded, and the same machinery works at every position.
:::
