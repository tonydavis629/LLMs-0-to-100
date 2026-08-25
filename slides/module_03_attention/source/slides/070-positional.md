:::divider id="divider-positional" title="Positional Embeddings" sub="Teaching the model that order matters"
:::

---

<!-- .slide: id="permutation-equivariance" -->

## Attention Ignores Order

- Attention compares every token to every other token
- The comparison ignores **where** tokens sit
- Shuffle the inputs, and the outputs come back in the same shuffle: **permutation equivariant**

:::note
**Problem:** "dog bites man" and "man bites dog" are the same bag of tokens. Without position information, attention treats them identically.
:::

---

<!-- .slide: id="permutation-demo" -->

:::interactive id="perm-shuffle" widget="permutationShuffle" title="Same Tokens, Reordered: Same Attention"
:::

---

<!-- .slide: id="adding-position" -->

## The Fix: Add a Position Signal

Give each position its own signature and **add it to the token embedding** before attention runs:

$$X_{\text{pos}} = X + P$$

<div style="text-align: center; margin: 8px 0;">
<svg viewBox="0 0 720 130" width="100%" style="max-height: 120px;">
  <g font-size="12" text-anchor="middle" font-weight="600">
    <rect x="40" y="40" width="120" height="40" rx="5" fill="#0d1225" stroke="#4a9eff" stroke-width="1.5"/><text x="100" y="65" fill="#e8eaf0">token embedding</text>
    <text x="100" y="100" fill="#8892a4" font-size="11" font-weight="400">what the word is</text>
    <text x="195" y="65" fill="#e8eaf0" font-size="22">+</text>
    <rect x="230" y="40" width="120" height="40" rx="5" fill="#0d1225" stroke="#f5a623" stroke-width="1.5"/><text x="290" y="65" fill="#e8eaf0">position signal</text>
    <text x="290" y="100" fill="#8892a4" font-size="11" font-weight="400">where it sits</text>
    <text x="385" y="65" fill="#e8eaf0" font-size="22">=</text>
    <rect x="420" y="40" width="150" height="40" rx="5" fill="#0d1225" stroke="#3fb950" stroke-width="1.5"/><text x="495" y="65" fill="#e8eaf0">position-aware input</text>
  </g>
</svg>
</div>

Identical words at different positions now arrive as **different vectors**. Remaining question: what should $P$ look like?

---

<!-- .slide: id="sinusoidal" -->

## Sinusoidal Positional Encodings

A fixed pattern: each dimension of $P$ is a sine or cosine wave with its own frequency.

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right) \qquad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

:::columns cols="2" gap="30px"
**Low dimensions:** high frequencies. Flip quickly from token to token. Encode fine position.
+++
**High dimensions:** low frequencies. Change slowly. Encode coarse position over long spans.
:::

No learned parameters. The pattern extends to lengths never seen in training.

---

<!-- .slide: id="sinusoidal-interactive" -->

:::interactive id="pe-explorer" widget="positionalEncoding" title="Sinusoidal Encodings and Their Effect on Attention"
:::

---

<!-- .slide: id="learned-and-other" -->

## Learned and Relative Schemes

:::columns cols="2" gap="30px"
**Learned positional embeddings**

A trainable table, one vector per position:

$$X_{\text{pos}} = X + P, \quad P \in \mathbb{R}^{L \times d}$$

- simple and effective
- limited to the maximum training length
- used in BERT and GPT-2
+++
**Relative and bias-based schemes**

Often the **distance** between tokens matters, not the absolute index.

- **Relative position** encodings inject a term that depends on $m - n$
- **ALiBi** adds a distance penalty to the attention scores
- Both extrapolate to longer contexts more gracefully
:::

---

<!-- .slide: id="rope" -->

## RoPE: Position as Rotation

**RoPE** (Rotary Position Embedding, Su et al. 2021):

- **rotate** each query and key by an angle proportional to its position
- the dot product between positions $m$ and $n$ then depends only on the **offset** $m - n$

<div style="text-align: center; margin: 6px 0;">
<svg viewBox="0 0 720 230" width="100%" style="max-height: 220px;">
  <!-- three positions, each rotating the same base vector more -->
  <g font-size="12" text-anchor="middle">
    <!-- pos 0 -->
    <circle cx="120" cy="120" r="70" fill="none" stroke="#2a3450" stroke-width="1"/>
    <line x1="120" y1="120" x2="190" y2="120" stroke="#4a9eff" stroke-width="3" marker-end="url(#arrr)"/>
    <text x="120" y="210" fill="#8892a4">position 0</text>
    <text x="120" y="32" fill="#4a9eff" font-weight="600">angle 0</text>
    <!-- pos 1 -->
    <circle cx="340" cy="120" r="70" fill="none" stroke="#2a3450" stroke-width="1"/>
    <line x1="340" y1="120" x2="389" y2="71" stroke="#4a9eff" stroke-width="3" marker-end="url(#arrr)"/>
    <path d="M410 120 A70 70 0 0 0 389 71" fill="none" stroke="#f5a623" stroke-width="1.5"/>
    <text x="340" y="210" fill="#8892a4">position 1</text>
    <text x="340" y="32" fill="#f5a623" font-weight="600">rotate by &#952;</text>
    <!-- pos 2 -->
    <circle cx="560" cy="120" r="70" fill="none" stroke="#2a3450" stroke-width="1"/>
    <line x1="560" y1="120" x2="560" y2="50" stroke="#4a9eff" stroke-width="3" marker-end="url(#arrr)"/>
    <path d="M630 120 A70 70 0 0 0 560 50" fill="none" stroke="#f5a623" stroke-width="1.5"/>
    <text x="560" y="210" fill="#8892a4">position 2</text>
    <text x="560" y="32" fill="#f5a623" font-weight="600">rotate by 2&#952;</text>
  </g>
  <defs><marker id="arrr" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#4a9eff"/></marker></defs>
</svg>
</div>

Later positions rotate the same vector further. Used in LLaMA, Mistral, and most modern LLMs.
