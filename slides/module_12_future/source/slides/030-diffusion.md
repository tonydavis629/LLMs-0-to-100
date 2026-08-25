:::divider id="divider-diffusion" title="Assumption One: Text Is Generated Left to Right" sub="Diffusion language models"
:::

---

<!-- .slide: id="diffusion-assumption" -->

## Where the Assumption Is Encoded

Two mechanisms from earlier modules enforce left-to-right generation.

$$p(x_1, \dots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_{<t})$$

:::columns cols="2" gap="34px"
**The chain-rule factorization**

- Exactly true for any ordering
- Left to right makes the likelihood **exactly computable** in one pass
+++
**The causal mask**

- Position $t$ cannot see position $t+1$
- One forward pass gives a valid prediction at every position
:::

Left-to-right is not a property of language. It is a factorization chosen for computational convenience. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="diffusion-costs" -->

## Costs of Left-to-Right Generation

<div class="card-grid cols-3">
<div class="card"><h4>Strictly sequential</h4><p>One forward pass per token. Generation is <strong>memory-bound</strong>: the time goes to moving weights, not arithmetic.</p></div>
<div class="card"><h4>No revision</h4><p>The model cannot change a token once it is generated, even after an early mistake becomes apparent.</p></div>
<div class="card"><h4>No infilling</h4><p>The model structurally cannot condition on text <strong>after</strong> a blank. Every editing workflow is a workaround for this.</p></div>
</div>

Only the first of these is about speed. The other two are about what the model can express at all. <!-- .element: class="text-lg" style="margin-top: 14px;" -->

---

:::figure img="images/jascha_sohl_dickstein.jpg" name="Jascha Sohl-Dickstein" kicker="Deep Unsupervised Learning using Nonequilibrium Thermodynamics (2015)" alt="Jascha Sohl-Dickstein"
Introduced diffusion models with an idea from statistical physics: destroy a structured distribution with noise until it becomes simple, then train a network to run the destruction backwards. The 2015 paper preceded the image-generation wave by years and text diffusion by a decade.

Diffusion appeared earlier in the course for images. Here it competes for text generation.
:::

---

<!-- .slide: id="diffusion-mechanics" -->

## Diffusion: Generation Without an Ordering

Start fully masked. Unmask iteratively, refining the whole sequence over a few dozen steps.

:::columns cols="2" gap="34px"
**Training**

- Mask a fraction of positions
- Predict **every masked position at once**
- Same denoising objective as span corruption in pretraining
+++
**Generation**

- Start with everything masked
- Predict all positions, keep the **confident** ones
- Re-mask the rest, repeat
:::

Step count is a dial: few steps means fast but lower quality, many steps approaches autoregressive quality with less speedup. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="diffusion-order" -->

## Autoregressive vs Diffusion Generation

<div class="svg-figure">
<svg viewBox="0 0 1000 360" role="img" aria-label="Autoregressive generation commits one token per forward pass, left to right, taking eight passes for eight tokens. Diffusion starts from all masked positions and unmasks the confident ones across four passes, filling positions in no particular order.">
  <text x="20" y="34" fill="#4a9eff" font-size="22" font-family="Inter, sans-serif">Autoregressive: one token per pass, always left to right</text>
  <g fill="none" stroke="#2a3450" stroke-width="2">
    <rect x="20" y="56" width="216" height="34" /><rect x="274" y="56" width="216" height="34" /><rect x="528" y="56" width="216" height="34" /><rect x="782" y="56" width="216" height="34" />
  </g>
  <g fill="#4a9eff">
    <rect x="22" y="58" width="25" height="30" />
    <rect x="276" y="58" width="25" height="30" /><rect x="303" y="58" width="25" height="30" />
    <rect x="530" y="58" width="25" height="30" /><rect x="557" y="58" width="25" height="30" /><rect x="584" y="58" width="25" height="30" />
    <rect x="784" y="58" width="25" height="30" /><rect x="811" y="58" width="25" height="30" /><rect x="838" y="58" width="25" height="30" /><rect x="865" y="58" width="25" height="30" /><rect x="892" y="58" width="25" height="30" /><rect x="919" y="58" width="25" height="30" /><rect x="946" y="58" width="25" height="30" /><rect x="973" y="58" width="23" height="30" />
  </g>
  <text x="20" y="116" fill="#8892a4" font-size="19" font-family="Inter, sans-serif">pass 1</text>
  <text x="274" y="116" fill="#8892a4" font-size="19" font-family="Inter, sans-serif">pass 2</text>
  <text x="528" y="116" fill="#8892a4" font-size="19" font-family="Inter, sans-serif">pass 3</text>
  <text x="782" y="116" fill="#8892a4" font-size="19" font-family="Inter, sans-serif">pass 8</text>
  <line x1="20" y1="160" x2="980" y2="160" stroke="#2a3450" stroke-width="2" />
  <text x="20" y="206" fill="#50c878" font-size="22" font-family="Inter, sans-serif">Diffusion: every position at once, unmasking the confident ones</text>
  <g fill="none" stroke="#2a3450" stroke-width="2">
    <rect x="20" y="228" width="216" height="34" /><rect x="274" y="228" width="216" height="34" /><rect x="528" y="228" width="216" height="34" /><rect x="782" y="228" width="216" height="34" />
  </g>
  <g fill="#50c878">
    <rect x="303" y="230" width="25" height="30" /><rect x="384" y="230" width="25" height="30" /><rect x="465" y="230" width="25" height="30" />
    <rect x="530" y="230" width="25" height="30" /><rect x="557" y="230" width="25" height="30" /><rect x="611" y="230" width="25" height="30" /><rect x="638" y="230" width="25" height="30" /><rect x="692" y="230" width="25" height="30" /><rect x="719" y="230" width="25" height="30" />
    <rect x="784" y="230" width="25" height="30" /><rect x="811" y="230" width="25" height="30" /><rect x="838" y="230" width="25" height="30" /><rect x="865" y="230" width="25" height="30" /><rect x="892" y="230" width="25" height="30" /><rect x="919" y="230" width="25" height="30" /><rect x="946" y="230" width="25" height="30" /><rect x="973" y="230" width="23" height="30" />
  </g>
  <text x="20" y="288" fill="#8892a4" font-size="19" font-family="Inter, sans-serif">pass 1: all masked</text>
  <text x="274" y="288" fill="#8892a4" font-size="19" font-family="Inter, sans-serif">pass 2</text>
  <text x="528" y="288" fill="#8892a4" font-size="19" font-family="Inter, sans-serif">pass 3</text>
  <text x="782" y="288" fill="#8892a4" font-size="19" font-family="Inter, sans-serif">pass 4: done</text>
</svg>
</div>

Eight tokens cost eight passes on top and four on the bottom. Lengthen the sequence and the top number grows with it while the bottom number stays where you set it. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="diffusion-status" -->

## Current Status

Note what each piece of evidence actually shows.

<div class="bench-table">
<table>
<thead><tr><th>Evidence</th><th>What it shows</th></tr></thead>
<tbody>
<tr><td><strong>LLaDA</strong> (2025), 8B parameters</td><td>The recipe holds against comparable autoregressive baselines at 8B. This is a real result</td></tr>
<tr><td><strong>Mercury</strong>, <strong>Gemini Diffusion</strong></td><td>Commercial systems advertising large throughput gains. Real products, vendor-reported numbers</td></tr>
<tr><td>Frontier scale</td><td><strong>Not demonstrated.</strong> No diffusion model has been shown to match the best autoregressive models</td></tr>
</tbody>
</table>
</div>

Summary: competitive at mid scale, unproven above it. <!-- .element: class="text-lg" style="margin-top: 14px;" -->

---

<!-- .slide: id="diffusion-stack" -->

## The Serving Stack Is the Obstacle

:::columns cols="2" gap="34px"
**Why the KV cache exists**

- Past tokens **never change**
- Cache the past, compute only the new token
- The whole serving apparatus rests on this guarantee
+++
**What diffusion does to it**

- Past tokens change on every step
- Caching, batching, and streaming must be **rebuilt from scratch**
:::

A successor must beat the incumbent's loss curve **plus nine years of infrastructure** built on assumptions it violates. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="diffusion-hybrid" -->

## Block Diffusion

**Block diffusion**: autoregressive across blocks, diffusive within them.

:::columns cols="2" gap="34px"
**What you keep**

- KV caching between blocks
- Earlier blocks are finished and will not change
+++
**What you gain**

- Parallel generation inside each block
- Revision within a block before committing it
:::

The second hybrid in two sections. Autoregression won for computational convenience, not because language is inherently left to right. <!-- .element: class="text-lg" style="margin-top: 14px;" -->

---

<!-- .slide: id="sidequest-lecun" -->

## Side Quest: Is Autoregression a Dead End?

Yann LeCun's critique:

- Generating one token at a time **accumulates error**
- Each token conditions on earlier mistakes, with no world model to correct against
- On this view the paradigm cannot reach robust intelligence at any scale
- His alternative: predict in representation space, not token space

Debate questions:

- What **observation** would confirm or refute this? If none exists, is it a scientific claim?
- Reasoning models spend tokens **checking their own work** and revising. Does that count as the missing error correction, or is it the same error accumulation with extra steps?
- Diffusion models revise committed text by construction. Does that answer LeCun's objection, or a different one?
