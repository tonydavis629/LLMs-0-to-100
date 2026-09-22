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
<svg viewBox="0 0 880 190" width="100%" style="max-height: 190px;">
  <text x="50" y="20" fill="#8892a4" font-size="12">the same RNN cell runs once per token; each step overwrites the hidden state: new h = f(old h, token)</text>
  <rect x="50" y="146" width="80" height="32" rx="4" fill="#0d1225" stroke="#4a9eff" stroke-width="1.5"/><text x="90" y="167" fill="#e8eaf0" font-size="13" text-anchor="middle" font-weight="600">the</text>
  <line x1="90" y1="146" x2="90" y2="98" stroke="#4a9eff" stroke-width="1.5" marker-end="url(#arrbb)"/>
  <rect x="50" y="56" width="80" height="40" rx="6" fill="rgba(136,146,164,0.10)" stroke="#8892a4" stroke-width="1.5"/><text x="90" y="81" fill="#e8eaf0" font-size="13" text-anchor="middle" font-weight="600">RNN</text>
  <line x1="130" y1="76" x2="168" y2="76" stroke="#f5a623" stroke-width="1.8" marker-end="url(#arrbo)"/>
  <text x="148" y="64" fill="#f5a623" font-size="14" text-anchor="middle" font-weight="600">h<tspan dy="4" font-size="10">1</tspan></text>
  <rect x="170" y="146" width="80" height="32" rx="4" fill="#0d1225" stroke="#4a9eff" stroke-width="1.5"/><text x="210" y="167" fill="#e8eaf0" font-size="13" text-anchor="middle" font-weight="600">cat</text>
  <line x1="210" y1="146" x2="210" y2="98" stroke="#4a9eff" stroke-width="1.5" marker-end="url(#arrbb)"/>
  <rect x="170" y="56" width="80" height="40" rx="6" fill="rgba(136,146,164,0.10)" stroke="#8892a4" stroke-width="1.5"/><text x="210" y="81" fill="#e8eaf0" font-size="13" text-anchor="middle" font-weight="600">RNN</text>
  <line x1="250" y1="76" x2="288" y2="76" stroke="#f5a623" stroke-width="1.8" marker-end="url(#arrbo)"/>
  <text x="268" y="64" fill="#f5a623" font-size="14" text-anchor="middle" font-weight="600">h<tspan dy="4" font-size="10">2</tspan></text>
  <rect x="290" y="146" width="80" height="32" rx="4" fill="#0d1225" stroke="#4a9eff" stroke-width="1.5"/><text x="330" y="167" fill="#e8eaf0" font-size="13" text-anchor="middle" font-weight="600">sat</text>
  <line x1="330" y1="146" x2="330" y2="98" stroke="#4a9eff" stroke-width="1.5" marker-end="url(#arrbb)"/>
  <rect x="290" y="56" width="80" height="40" rx="6" fill="rgba(136,146,164,0.10)" stroke="#8892a4" stroke-width="1.5"/><text x="330" y="81" fill="#e8eaf0" font-size="13" text-anchor="middle" font-weight="600">RNN</text>
  <line x1="370" y1="76" x2="408" y2="76" stroke="#f5a623" stroke-width="1.8" marker-end="url(#arrbo)"/>
  <text x="388" y="64" fill="#f5a623" font-size="14" text-anchor="middle" font-weight="600">h<tspan dy="4" font-size="10">3</tspan></text>
  <rect x="410" y="146" width="80" height="32" rx="4" fill="#0d1225" stroke="#4a9eff" stroke-width="1.5"/><text x="450" y="167" fill="#e8eaf0" font-size="13" text-anchor="middle" font-weight="600">on</text>
  <line x1="450" y1="146" x2="450" y2="98" stroke="#4a9eff" stroke-width="1.5" marker-end="url(#arrbb)"/>
  <rect x="410" y="56" width="80" height="40" rx="6" fill="rgba(136,146,164,0.10)" stroke="#8892a4" stroke-width="1.5"/><text x="450" y="81" fill="#e8eaf0" font-size="13" text-anchor="middle" font-weight="600">RNN</text>
  <line x1="490" y1="76" x2="528" y2="76" stroke="#f5a623" stroke-width="1.8" marker-end="url(#arrbo)"/>
  <text x="508" y="64" fill="#f5a623" font-size="14" text-anchor="middle" font-weight="600">h<tspan dy="4" font-size="10">4</tspan></text>
  <rect x="530" y="146" width="80" height="32" rx="4" fill="#0d1225" stroke="#4a9eff" stroke-width="1.5"/><text x="570" y="167" fill="#e8eaf0" font-size="13" text-anchor="middle" font-weight="600">the</text>
  <line x1="570" y1="146" x2="570" y2="98" stroke="#4a9eff" stroke-width="1.5" marker-end="url(#arrbb)"/>
  <rect x="530" y="56" width="80" height="40" rx="6" fill="rgba(136,146,164,0.10)" stroke="#8892a4" stroke-width="1.5"/><text x="570" y="81" fill="#e8eaf0" font-size="13" text-anchor="middle" font-weight="600">RNN</text>
  <line x1="610" y1="76" x2="648" y2="76" stroke="#f5a623" stroke-width="1.8" marker-end="url(#arrbo)"/>
  <text x="628" y="64" fill="#f5a623" font-size="14" text-anchor="middle" font-weight="600">h<tspan dy="4" font-size="10">5</tspan></text>
  <rect x="650" y="146" width="80" height="32" rx="4" fill="#0d1225" stroke="#4a9eff" stroke-width="1.5"/><text x="690" y="167" fill="#e8eaf0" font-size="13" text-anchor="middle" font-weight="600">mat</text>
  <line x1="690" y1="146" x2="690" y2="98" stroke="#4a9eff" stroke-width="1.5" marker-end="url(#arrbb)"/>
  <rect x="650" y="56" width="80" height="40" rx="6" fill="rgba(136,146,164,0.10)" stroke="#8892a4" stroke-width="1.5"/><text x="690" y="81" fill="#e8eaf0" font-size="13" text-anchor="middle" font-weight="600">RNN</text>
  <line x1="10" y1="76" x2="48" y2="76" stroke="#f5a623" stroke-width="1.8" marker-end="url(#arrbo)"/>
  <text x="28" y="64" fill="#f5a623" font-size="14" text-anchor="middle" font-weight="600">h<tspan dy="4" font-size="10">0</tspan></text>
  <line x1="730" y1="76" x2="786" y2="76" stroke="#e74c3c" stroke-width="2" marker-end="url(#arrbr)"/>
  <rect x="788" y="52" width="76" height="48" rx="6" fill="rgba(231,76,60,0.12)" stroke="#e74c3c" stroke-width="2"/>
  <text x="824" y="82" fill="#e8eaf0" font-size="17" text-anchor="middle" font-weight="600">h<tspan dy="4" font-size="13">6</tspan></text>
  <text x="826" y="124" fill="#e74c3c" font-size="12" text-anchor="middle">one fixed-size</text>
  <text x="826" y="140" fill="#e74c3c" font-size="12" text-anchor="middle">vector holds the</text>
  <text x="826" y="156" fill="#e74c3c" font-size="12" text-anchor="middle">whole sentence</text>
  <defs>
    <marker id="arrbb" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#4a9eff"/></marker>
    <marker id="arrbo" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#f5a623"/></marker>
    <marker id="arrbr" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#e74c3c"/></marker>
  </defs>
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
