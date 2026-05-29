:::divider id="divider-why-attention" title="Why Attention?" sub="From compression to selective retrieval"
:::

---

<!-- .slide: id="fixed-context-bottleneck" -->

## The Fixed-Context Bottleneck

Earlier sequence models (RNNs, LSTMs) processed tokens one at a time and compressed the entire sequence into a **single hidden state vector**.

:::note
**Problem:** No matter how long the input, all information must fit through one fixed-size bottleneck. The model must decide what to keep before it knows what will be needed later.
:::

Attention solves this by giving each output token **direct access** to all input representations. No compression step. No bottleneck. Each token builds its own context by looking at whichever other tokens are relevant.

---

<!-- .slide: id="attention-concept-viz" -->

## Attention: Selective Connections Between Words

Before the math, here is the idea: every word looks at every other word and decides which ones matter for understanding itself.

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
**The key insight:** each token chooses its own context. A verb looks for its subject; a pronoun looks for its antecedent. The connections are learned, not hand-coded.
:::

---

<!-- .slide: id="attention-as-lookup" -->

## Attention as Learned Lookup

Think of attention as a differentiable database query:

:::columns cols="3" gap="20px"
**Query**

What I am looking for

"Which tokens are relevant to me?"
+++
**Key**

What each token advertises

"I contain information about X"
+++
**Value**

What I retrieve

The actual content to aggregate
:::

The model learns its own queries, keys, and values. No manual feature engineering. The lookup pattern is discovered during training.

---

<!-- .slide: id="compare-normalize-retrieve" -->

## Attention: Compare, Normalize, Retrieve

<div style="text-align: center; margin: 5px 0;">
<svg viewBox="0 0 780 270" width="100%" style="max-height: 260px;">
  <rect x="20" y="8" width="64" height="30" rx="3" fill="#0d1225" stroke="#4a9eff" stroke-width="1.5"/>
  <text x="52" y="28" fill="#e8eaf0" font-size="11" text-anchor="middle">X</text>
  <rect x="10" y="60" width="74" height="28" rx="3" fill="rgba(74,158,255,0.12)" stroke="#4a9eff" stroke-width="1.5"/>
  <text x="47" y="79" fill="#4a9eff" font-size="11" text-anchor="middle" font-weight="600">W_Q</text>
  <rect x="10" y="100" width="74" height="28" rx="3" fill="rgba(245,166,35,0.12)" stroke="#f5a623" stroke-width="1.5"/>
  <text x="47" y="119" fill="#f5a623" font-size="11" text-anchor="middle" font-weight="600">W_K</text>
  <rect x="10" y="140" width="74" height="28" rx="3" fill="rgba(80,200,120,0.12)" stroke="#50c878" stroke-width="1.5"/>
  <text x="47" y="159" fill="#50c878" font-size="11" text-anchor="middle" font-weight="600">W_V</text>
  <line x1="52" y1="38" x2="47" y2="60" stroke="#8892a4" stroke-width="1"/>
  <line x1="52" y1="38" x2="47" y2="100" stroke="#8892a4" stroke-width="1"/>
  <line x1="52" y1="38" x2="47" y2="140" stroke="#8892a4" stroke-width="1"/>
  <rect x="120" y="60" width="56" height="28" rx="3" fill="#0d1225" stroke="#4a9eff" stroke-width="1.5"/>
  <text x="148" y="79" fill="#4a9eff" font-size="11" text-anchor="middle" font-weight="600">Q</text>
  <rect x="120" y="100" width="56" height="28" rx="3" fill="#0d1225" stroke="#f5a623" stroke-width="1.5"/>
  <text x="148" y="119" fill="#f5a623" font-size="11" text-anchor="middle" font-weight="600">K</text>
  <rect x="120" y="140" width="56" height="28" rx="3" fill="#0d1225" stroke="#50c878" stroke-width="1.5"/>
  <text x="148" y="159" fill="#50c878" font-size="11" text-anchor="middle" font-weight="600">V</text>
  <line x1="84" y1="74" x2="120" y2="74" stroke="#4a9eff" stroke-width="1.5"/>
  <line x1="84" y1="114" x2="120" y2="114" stroke="#f5a623" stroke-width="1.5"/>
  <line x1="84" y1="154" x2="120" y2="154" stroke="#50c878" stroke-width="1.5"/>
  <rect x="210" y="70" width="110" height="50" rx="4" fill="rgba(74,158,255,0.08)" stroke="#4a9eff" stroke-width="1.5"/>
  <text x="265" y="92" fill="#4a9eff" font-size="12" text-anchor="middle" font-weight="600">1. Compare</text>
  <text x="265" y="110" fill="#e8eaf0" font-size="11" text-anchor="middle">Q K^T</text>
  <line x1="176" y1="74" x2="210" y2="86" stroke="#4a9eff" stroke-width="1.5" marker-end="url(#arrbl)"/>
  <line x1="176" y1="114" x2="210" y2="104" stroke="#f5a623" stroke-width="1.5" marker-end="url(#arrbl)"/>
  <rect x="345" y="78" width="54" height="34" rx="3" fill="#0d1225" stroke="#8892a4" stroke-width="1.5"/>
  <text x="372" y="99" fill="#e8eaf0" font-size="11" text-anchor="middle">scores</text>
  <line x1="320" y1="95" x2="345" y2="95" stroke="#8892a4" stroke-width="1.5" marker-end="url(#arrgr)"/>
  <rect x="425" y="70" width="130" height="50" rx="4" fill="rgba(245,166,35,0.08)" stroke="#f5a623" stroke-width="1.5"/>
  <text x="490" y="92" fill="#f5a623" font-size="12" text-anchor="middle" font-weight="600">2. Normalize</text>
  <text x="490" y="110" fill="#e8eaf0" font-size="11" text-anchor="middle">softmax / sqrt(d_k)</text>
  <line x1="399" y1="95" x2="425" y2="95" stroke="#8892a4" stroke-width="1.5" marker-end="url(#arrgr)"/>
  <rect x="575" y="78" width="60" height="34" rx="3" fill="#0d1225" stroke="#50c878" stroke-width="1.5"/>
  <text x="605" y="99" fill="#e8eaf0" font-size="11" text-anchor="middle">weights</text>
  <line x1="555" y1="95" x2="575" y2="95" stroke="#50c878" stroke-width="1.5" marker-end="url(#arrgr)"/>
  <rect x="660" y="70" width="110" height="50" rx="4" fill="rgba(80,200,120,0.08)" stroke="#4a9eff" stroke-width="1.5"/>
  <text x="715" y="92" fill="#4a9eff" font-size="12" text-anchor="middle" font-weight="600">3. Retrieve</text>
  <text x="715" y="110" fill="#e8eaf0" font-size="11" text-anchor="middle">weights * V</text>
  <line x1="635" y1="95" x2="660" y2="95" stroke="#4a9eff" stroke-width="1.5" marker-end="url(#arrbl)"/>
  <path d="M148 168 Q148 210 705 210 Q740 210 715 140" fill="none" stroke="#50c878" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#arrgr)"/>
  <rect x="690" y="140" width="70" height="32" rx="3" fill="#0d1225" stroke="#4a9eff" stroke-width="2"/>
  <text x="725" y="160" fill="#e8eaf0" font-size="12" text-anchor="middle" font-weight="600">Output</text>
  <line x1="715" y1="120" x2="715" y2="140" stroke="#4a9eff" stroke-width="1.5" marker-end="url(#arrbl)"/>
  <text x="10" y="250" fill="#8892a4" font-size="11">Each token vector X is projected into Query, Key, and Value. The three steps produce a new representation for every token.</text>
  <defs>
    <marker id="arrbl" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="#4a9eff"/>
    </marker>
    <marker id="arrgr" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="#8892a4"/>
    </marker>
  </defs>
</svg>
</div>

:::columns cols="3" gap="20px"
**Compare**

Dot product measures alignment between Q and K
+++
**Normalize**

Softmax turns scores into a probability distribution
+++
**Retrieve**

Weighted average of values, where weights come from Q-K alignment
:::

---

<!-- .slide: id="qkv-intro" -->

## Queries, Keys, and Values: The Learned Projections

A traditional database uses an exact key match. Attention uses soft, differentiable matching. The model learns what to ask, what to advertise, and what to retrieve.

:::columns cols="2" gap="40px"
**Traditional Database**

- Exact match on a key field
- Returns one record
- Not differentiable
- Hand-designed index
+++
**Attention**

- Soft match between query and key vectors
- Returns a weighted combination of all values
- Fully differentiable
- Queries, keys, and values are all **learned**
:::

Each token vector is projected into three different spaces:

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

:::note
The same token can "ask" different questions (via Q) than it "advertises" (via K) or "offers" (via V). The projection matrices are learned during training.
:::

The model learns its own queries, keys, and values. No manual feature engineering. The lookup pattern is discovered during training.

---

<!-- .slide: id="compare-normalize-retrieve" -->

## The Core Operation

Attention has three steps:

:::columns cols="3" gap="20px"
**1. Compare**

$$\text{scores} = QK^T$$

How well does each query match each key?
+++
**2. Normalize**

$$\text{weights} = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)$$

Convert scores to a probability distribution
+++
**3. Retrieve**

$$\text{output} = \text{weights} \cdot V$$

Take a weighted average of value vectors
:::

Compare, normalize, retrieve. That is all of attention.

---

:::figure img="images/vaswani_et_al.jpg" name="Vaswani et al." kicker="Attention Is All You Need"
- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan Gomez, Lukasz Kaiser, and Illia Polosukhin (2017)
- "Attention Is All You Need" &mdash; replaced recurrence entirely with attention
- Introduced scaled dot-product multi-head attention
- The Transformer architecture: no RNN, no convolution, just attention and feed-forward layers
- One of the most cited papers in machine learning
:::
