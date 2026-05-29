:::divider id="title" title="LLMs 0 to 100" sub="Module 3: Attention Mechanisms"
From Fixed Windows to Learned Lookups
:::

---

<!-- .slide: id="review-1" -->

## Review: Module 2

:::columns cols="2" gap="30px"
**The MLP**

A multi-layer perceptron applies repeated linear transformations with nonlinear activations between them:

$$\mathbf{h} = \sigma(W\mathbf{x} + \mathbf{b})$$

An MLP with enough hidden units can approximate any continuous function (universal approximation theorem).
+++
**Limitations for Sequences**

An MLP over a fixed window of tokens treats each position identically, but:

- The context window is fixed: token 1 cannot directly inform token 50
- Flattening token vectors destroys the idea that the same pattern can appear in many positions
- Parameter count grows with context length
- There is no direct mechanism for one token to select information from another token
:::

---

<!-- .slide: id="review-2" -->

## Review: Why the Architecture Matters

:::columns cols="2" gap="30px"
**From Module 2**

The single neuron could not solve XOR because a linear boundary cannot separate entangled classes. The fix was a hidden layer with a nonlinearity &mdash; the architecture had to match the structure of the problem.
+++
**Same Principle, Bigger Scale**

Language has structure that an MLP does not match:

- Every token can depend on any earlier token, not just its neighbors
- The same syntactic pattern ("the ___") appears at every position
- The relevant context length varies per token

**The architecture must match the structure of the data.**
:::

---

<!-- .slide: id="mlp-failure" -->

## Why an MLP Fails at Language

Consider predicting the next word in a sentence. An MLP uses a fixed window of tokens, but language is not fixed:

<div style="text-align: center; margin: 15px 0;">
<svg viewBox="0 0 800 220" width="100%" style="max-height: 210px;">
  <rect x="20" y="20" width="80" height="36" rx="4" fill="#0d1225" stroke="#4a9eff" stroke-width="2"/>
  <text x="60" y="43" fill="#e8eaf0" font-size="13" text-anchor="middle" font-weight="600">the</text>
  <rect x="110" y="20" width="80" height="36" rx="4" fill="#0d1225" stroke="#4a9eff" stroke-width="2"/>
  <text x="150" y="43" fill="#e8eaf0" font-size="13" text-anchor="middle" font-weight="600">cat</text>
  <rect x="200" y="20" width="80" height="36" rx="4" fill="#0d1225" stroke="#4a9eff" stroke-width="2"/>
  <text x="240" y="43" fill="#e8eaf0" font-size="13" text-anchor="middle" font-weight="600">sat</text>
  <rect x="290" y="20" width="80" height="36" rx="4" fill="#0d1225" stroke="#4a9eff" stroke-width="2"/>
  <text x="330" y="43" fill="#e8eaf0" font-size="13" text-anchor="middle" font-weight="600">on</text>
  <rect x="380" y="20" width="80" height="36" rx="4" fill="#0d1225" stroke="#f5a623" stroke-width="2"/>
  <text x="420" y="43" fill="#e8eaf0" font-size="13" text-anchor="middle" font-weight="600">the</text>
  <rect x="470" y="20" width="80" height="36" rx="4" fill="#0d1225" stroke="#4a9eff" stroke-width="2"/>
  <text x="510" y="43" fill="#e8eaf0" font-size="13" text-anchor="middle" font-weight="600">mat</text>
  <text x="60" y="72" fill="#8892a4" font-size="10" text-anchor="middle">pos 1</text>
  <text x="150" y="72" fill="#8892a4" font-size="10" text-anchor="middle">pos 2</text>
  <text x="240" y="72" fill="#8892a4" font-size="10" text-anchor="middle">pos 3</text>
  <text x="330" y="72" fill="#8892a4" font-size="10" text-anchor="middle">pos 4</text>
  <text x="420" y="72" fill="#8892a4" font-size="10" text-anchor="middle">pos 5</text>
  <text x="510" y="72" fill="#8892a4" font-size="10" text-anchor="middle">pos 6</text>
  <rect x="110" y="95" width="260" height="40" rx="4" fill="none" stroke="#e74c3c" stroke-width="2" stroke-dasharray="6,4"/>
  <text x="240" y="120" fill="#e74c3c" font-size="12" text-anchor="middle">MLP fixed window (3 tokens)</text>
  <rect x="20" y="155" width="530" height="40" rx="4" fill="none" stroke="#3fb950" stroke-width="2"/>
  <text x="285" y="180" fill="#3fb950" font-size="12" text-anchor="middle">Attention: every token can see every other token</text>
  <line x1="240" y1="56" x2="240" y2="90" stroke="#8892a4" stroke-width="1.5" stroke-dasharray="4,3"/>
  <line x1="240" y1="135" x2="240" y2="152" stroke="#8892a4" stroke-width="1.5" stroke-dasharray="4,3"/>
</svg>
</div>

:::columns cols="2" gap="30px"
**Rigid positionality**

Each input position connects to its own weights. "the" at position 1 and "the" at position 5 use entirely different parameters. The model cannot reuse what it learned about "the" from one position at another.
+++
**No selective retrieval**

To predict "mat", the model needs "sat" and "cat". But the MLP sees everything in the window at once with no way to emphasize what matters. All tokens are flattened into one vector and treated equally.
:::

:::note
The architecture must match the data. Language needs variable context, weight sharing across positions, and selective information retrieval. An MLP provides none of these.
:::

---

:::figure img="images/bahdanau_cho_bengio.jpg" name="Bahdanau, Cho &amp; Bengio" kicker="Made Attention a Central Mechanism"
- Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio (2014)
- "Neural Machine Translation by Jointly Learning to Align and Translate"
- Earlier sequence models compressed the entire source sentence into one fixed-size vector
- Their model learned to **align** each output word to the most relevant input words
- Attention was originally a fix for the bottleneck of fixed-size context vectors
- This paper made attention a first-class mechanism, not just a patch
:::
