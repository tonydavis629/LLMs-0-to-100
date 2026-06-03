:::divider id="divider-sampling" title="Generating Text" sub="From logits to tokens: decoding strategies"
:::

---

<!-- .slide: id="logits-to-text" -->

## From Logits to Text

A forward pass gives a probability distribution over the vocabulary for the next token. Decoding chooses one token, appends it to the context, and repeats.

$$p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

Here $i$ and $j$ both index the **vocabulary**. $z_i$ is the logit (raw score) for the one candidate token $i$ whose probability we are computing; the denominator sums $e^{z_j}$ over **every** token $j$ in the vocabulary, normalizing the result so the probabilities sum to one. The choice of how to sample from this distribution is the **decoding strategy**, and it shapes what you actually read.

---

<!-- .slide: id="greedy-decoding" -->

## Greedy Decoding

Always pick the token with the highest probability:

$$\text{token}_t = \arg\max_i p_i$$

Simple, deterministic, and fast. But it produces repetitive, flat text because it never explores lower-probability but more interesting continuations. Once the model enters a loop ("the the the"), greediness keeps it there.

---

<!-- .slide: id="temperature" -->

## Temperature

Scale the logits before softmax to control randomness:

$$p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

- $T < 1$: sharpens the distribution. The model becomes more conservative, sticking to high-probability tokens.
- $T = 1$: the original distribution.
- $T > 1$: flattens the distribution. Unlikely tokens gain probability; the output becomes more random and creative.

Temperature is the simplest way to tune the exploration-exploitation trade-off at inference time.

---

<!-- .slide: id="topk-topp" -->

## Top-k and Top-p (Nucleus) Sampling

Temperature alone can still sample implausible tokens. Two truncation methods fix this:

:::columns cols="2" gap="30px"
**Top-k**

Keep only the $k$ most likely tokens, zero out the rest, renormalize, and sample. Fixed $k = 50$ is common. Simple, but can be too restrictive for peaked distributions or too permissive for flat ones.
+++
**Top-p (nucleus)**

Keep the smallest set of tokens whose cumulative probability exceeds $p$. For a peaked distribution, this may be only 5 tokens. For a flat distribution, it may be 200. Adapts dynamically to the distribution shape.
:::

In practice, temperature and top-p/top-k are used together: scale by temperature first, then truncate, then sample.

---

:::manim id="sampling-anim" scene="sampling-demo"
:::

---

<!-- .slide: id="beam-search" -->

## Beam Search

<div class="beam-tree">
<svg viewBox="0 0 880 320" role="img" aria-label="Beam search drawn as a left-to-right tree with beam width 2">
<path d="M130 165 C215 165 215 85 300 85" fill="none" stroke="#f5a623" stroke-width="2.5"/>
<path d="M130 165 C215 165 215 175 300 175" fill="none" stroke="#f5a623" stroke-width="2.5"/>
<path d="M130 165 C215 165 215 265 300 265" fill="none" stroke="#5a6478" stroke-width="1.5" stroke-dasharray="5 5"/>
<path d="M430 85 C510 85 510 55 590 55" fill="none" stroke="#f5a623" stroke-width="2.5"/>
<path d="M430 85 C510 85 510 120 590 120" fill="none" stroke="#5a6478" stroke-width="1.5" stroke-dasharray="5 5"/>
<path d="M430 175 C510 175 510 210 590 210" fill="none" stroke="#f5a623" stroke-width="2.5"/>
<path d="M430 175 C510 175 510 275 590 275" fill="none" stroke="#5a6478" stroke-width="1.5" stroke-dasharray="5 5"/>
<rect x="20" y="145" width="110" height="40" rx="8" fill="rgba(74,158,255,0.12)" stroke="rgba(74,158,255,0.6)" stroke-width="1.5"/>
<text x="75" y="170" text-anchor="middle" font-size="16" fill="#e8eaf0">The</text>
<rect x="300" y="65" width="130" height="40" rx="8" fill="rgba(245,166,35,0.14)" stroke="rgba(245,166,35,0.65)" stroke-width="1.5"/>
<text x="365" y="90" text-anchor="middle" font-size="15" fill="#e8eaf0">cat&#160;&#160;0.42</text>
<rect x="300" y="155" width="130" height="40" rx="8" fill="rgba(245,166,35,0.14)" stroke="rgba(245,166,35,0.65)" stroke-width="1.5"/>
<text x="365" y="180" text-anchor="middle" font-size="15" fill="#e8eaf0">dog&#160;&#160;0.31</text>
<rect x="300" y="245" width="130" height="40" rx="8" fill="rgba(136,146,164,0.06)" stroke="rgba(136,146,164,0.30)" stroke-width="1.5"/>
<text x="365" y="270" text-anchor="middle" font-size="15" fill="#8892a4">car&#160;&#160;0.08</text>
<rect x="590" y="35" width="150" height="40" rx="8" fill="rgba(245,166,35,0.14)" stroke="rgba(245,166,35,0.65)" stroke-width="1.5"/>
<text x="665" y="60" text-anchor="middle" font-size="15" fill="#e8eaf0">cat sat&#160;&#160;0.18</text>
<rect x="590" y="100" width="150" height="40" rx="8" fill="rgba(136,146,164,0.06)" stroke="rgba(136,146,164,0.30)" stroke-width="1.5"/>
<text x="665" y="125" text-anchor="middle" font-size="15" fill="#8892a4">cat ran&#160;&#160;0.05</text>
<rect x="590" y="190" width="150" height="40" rx="8" fill="rgba(245,166,35,0.14)" stroke="rgba(245,166,35,0.65)" stroke-width="1.5"/>
<text x="665" y="215" text-anchor="middle" font-size="15" fill="#e8eaf0">dog ran&#160;&#160;0.15</text>
<rect x="590" y="255" width="150" height="40" rx="8" fill="rgba(136,146,164,0.06)" stroke="rgba(136,146,164,0.30)" stroke-width="1.5"/>
<text x="665" y="280" text-anchor="middle" font-size="15" fill="#8892a4">dog the&#160;&#160;0.04</text>
<text x="752" y="60" text-anchor="start" font-size="13" fill="#f5a623">beam 1</text>
<text x="752" y="215" text-anchor="start" font-size="13" fill="#f5a623">beam 2</text>
<text x="75" y="225" text-anchor="middle" font-size="12" fill="#8892a4">start</text>
<text x="365" y="35" text-anchor="middle" font-size="12" fill="#8892a4">step 1</text>
<text x="665" y="22" text-anchor="middle" font-size="12" fill="#8892a4">step 2</text>
</svg>
</div>

The two solid orange paths are the surviving **beams**; the dashed branches were scored and pruned. Instead of committing to one token, keep the $b$ best partial sequences at every step ($b = 2$ here). Expand every beam, score the candidates, and keep the top $b$.

Best for constrained sequence-to-sequence tasks like translation, where there is a single correct answer and exploration helps. Poor for open-ended generation: the most likely sequence under a language model is usually vacuous, repetitive, or otherwise undesirable.

A trained model is only half the story: the decoding strategy shapes what you actually read. Now that we know how the next token is chosen, let us follow a full prompt through the stack that produces these distributions.
