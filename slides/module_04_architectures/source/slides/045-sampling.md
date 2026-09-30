:::divider id="divider-sampling" title="Generating Text" sub="From logits to tokens: decoding strategies"
:::

---

<!-- .slide: id="logits-to-text" -->

## From Logits to Text

A forward pass produces a raw score (a **logit** $z$) for every token in the vocabulary. **Softmax** turns those scores into probabilities that sum to one:

$$p_{\textcolor{#f5a623}{i}} = \frac{e^{z_{\textcolor{#f5a623}{i}}}}{\sum_{\textcolor{#4a9eff}{j}} e^{z_{\textcolor{#4a9eff}{j}}}}$$

<div class="prompt-line"><span class="pl-text">The capital of France is</span><span class="pl-next">next token?</span></div>

<div class="logit-index-visual">
  <div class="lv-num"><b>i</b> &mdash; the one token whose probability we are computing (numerator)</div>
  <div class="vocab-stack">
    <div class="vocab-row">
      <span class="vtok i">Paris<small>z = 8.2</small></span>
      <span class="vtok">London<small>5.1</small></span>
      <span class="vtok">Lyon<small>4.7</small></span>
      <span class="vtok">city<small>4.0</small></span>
      <span class="vtok">the<small>3.2</small></span>
      <span class="vtok">of<small>2.5</small></span>
      <span class="vtok more">&hellip;</span>
    </div>
    <div class="vocab-bracket"></div>
  </div>
  <div class="lv-den"><b>j</b> &mdash; runs over <strong>every</strong> token in the vocabulary; the sum normalizes (denominator)</div>
</div>

---

<!-- .slide: id="greedy-decoding" -->

## Greedy Decoding

Always pick the token with the highest probability:

$$\text{token}_t = \arg\max_i p_i$$

- Simple, deterministic, fast
- Produces repetitive, flat text; never explores lower-probability continuations
- Once in a loop ("the the the"), it stays there

---

<!-- .slide: id="temperature" -->

## Temperature

Scale the logits by $\textcolor{#f5a623}{T}$ before softmax to control randomness:

$$p_i = \frac{e^{z_i / \textcolor{#f5a623}{T}}}{\sum_j e^{z_j / \textcolor{#f5a623}{T}}}$$

<div class="temp-regimes">
  <div class="temp-card"><b>T &lt; 1</b><span>sharper</span><p>conservative; sticks to the highest-probability tokens</p></div>
  <div class="temp-card"><b>T = 1</b><span>original</span><p>the model's raw distribution, unchanged</p></div>
  <div class="temp-card"><b>T &gt; 1</b><span>flatter</span><p>more random and creative; unlikely tokens gain probability</p></div>
</div>

The simplest inference-time control over randomness.

---

<!-- .slide: id="topk-topp" -->

## Top-k and Top-p (Nucleus) Sampling

Temperature alone can still sample implausible tokens. Two truncation fixes:

:::columns cols="2" gap="30px"
**Top-k**

- Keep the $k$ most likely tokens, renormalize, sample
- $k = 50$ is common
- Fixed cutoff: too strict for peaked distributions, too loose for flat ones
+++
**Top-p (nucleus)**

- Keep the smallest set with cumulative probability &gt; $p$
- Peaked distribution: ~5 tokens. Flat: ~200
- Adapts to distribution shape
:::

In practice: scale by temperature, then truncate, then sample.

---

:::interactive id="sampling-explorer" widget="samplingExplorer" title="Temperature, Top-k, Top-p on One Distribution"
:::

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

Keep the $b$ best partial sequences at every step ($b = 2$ here). Solid orange paths survive; dashed branches were scored and pruned.

- **Good for:** translation and other tasks with one correct answer
- **Bad for:** open-ended generation; the most likely sequence is usually vacuous or repetitive

Next: follow a full prompt through the stack that produces these distributions.
