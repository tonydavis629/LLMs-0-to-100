:::divider id="divider-sampling" title="Generating Text" sub="From logits to tokens: decoding strategies"
:::

---

<!-- .slide: id="logits-to-text" -->

## From Logits to Text

A forward pass gives a probability distribution over the vocabulary for the next token. Decoding chooses one token, appends it to the context, and repeats.

$$p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

where $z_i$ is the logit for vocabulary item $i$. The choice of how to sample from this distribution is the **decoding strategy**, and it shapes what you actually read.

---

<!-- .slide: id="greedy-decoding" -->

## Greedy Decoding

Always pick the token with the highest probability:

$$\text{token}_t = \arg\max_i \, p_i$$

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

<!-- .slide: id="beam-search" -->

## Beam Search

<div class="beam-diagram">
  <div class="beam-node root">The</div>
  <div class="beam-level">
    <div class="beam-node kept">cat<br><span>0.42</span></div>
    <div class="beam-node kept">dog<br><span>0.31</span></div>
    <div class="beam-node dropped">car<br><span>0.08</span></div>
  </div>
  <div class="beam-level">
    <div class="beam-node kept">cat sat<br><span>0.18</span></div>
    <div class="beam-node kept">dog ran<br><span>0.15</span></div>
    <div class="beam-node dropped">cat barked<br><span>0.03</span></div>
  </div>
</div>

Instead of committing to one token, keep the $b$ best partial sequences at every step. Expand every beam, score the candidates, and keep the top $b$.

Best for constrained sequence-to-sequence tasks like translation, where there is a single correct answer and exploration helps. Poor for open-ended generation: the most likely sequence under a language model is usually vacuous, repetitive, or otherwise undesirable.

This sets up the exercise: a trained model is only half the story. The decoding strategy shapes what you actually read.
