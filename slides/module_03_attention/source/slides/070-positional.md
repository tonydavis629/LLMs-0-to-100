:::divider id="divider-positional" title="Positional Embeddings" sub="Teaching the model that order matters"
:::

---

<!-- .slide: id="permutation-equivariance" -->

## Attention Ignores Order

Self-attention compares every token to every other token, but the comparison does not depend on **where** the tokens sit. Shuffle the inputs and the outputs come back in exactly the same shuffle &mdash; the operation is **permutation equivariant**.

:::note
**This is a problem.** "dog bites man" and "man bites dog" contain the same bag of tokens. With no position information, attention treats them identically. It learns *what to look at* but not *where things are*.
:::

---

<!-- .slide: id="permutation-demo" -->

:::interactive id="perm-shuffle" widget="permutationShuffle" title="Same Tokens, Reordered: Same Attention"
:::

---

<!-- .slide: id="adding-position" -->

## The Fix: Add a Position Signal

If the tokens carry no order, we give each position its own signature and **add it to the token embedding** before attention ever runs:

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

Now two identical words at different positions arrive at attention as **different vectors**, so "dog bites man" is no longer the same as "man bites dog". The open question: what should $P$ look like?

---

<!-- .slide: id="sinusoidal" -->

## Sinusoidal Positional Encodings

A common fixed pattern makes each dimension of $P$ a sine or cosine wave, with a different frequency in each dimension.

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right) \qquad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

:::columns cols="2" gap="30px"
**Low dimensions** use high frequencies &mdash; they flip quickly from token to token, encoding fine position.
+++
**High dimensions** use low frequencies &mdash; they change slowly, encoding coarse position over long spans.
:::

No parameters to learn, and the pattern extends to sequence lengths never seen in training.

---

<!-- .slide: id="sinusoidal-interactive" -->

:::interactive id="pe-explorer" widget="positionalEncoding" title="Sinusoidal Encodings and Their Effect on Attention"
:::

---

<!-- .slide: id="learned-and-other" -->

## Learned and Relative Schemes

:::columns cols="2" gap="30px"
**Learned positional embeddings**

A trainable table with one vector per position, added like any embedding:

$$X_{\text{pos}} = X + P, \quad P \in \mathbb{R}^{L \times d}$$

Simple and effective, but limited to the maximum length seen in training. Used in BERT and GPT-2.
+++
**Relative and bias-based schemes**

What often matters is the **distance** between tokens, not absolute index.

- **Relative position** encodings inject a term that depends on $m - n$
- **ALiBi** adds a distance penalty straight onto the attention scores
- These extrapolate to longer contexts more gracefully
:::

---

<!-- .slide: id="rope" -->

## RoPE: Position as Rotation

**RoPE** (Rotary Position Embedding, Su et al. 2021) encodes position by **rotating** each query and key by an angle proportional to its position. The dot product between a query at position $m$ and a key at position $n$ then depends only on the **relative offset** $m - n$.

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

The same vector is rotated further at each later position. RoPE is used in LLaMA, Mistral, and most modern LLMs.
