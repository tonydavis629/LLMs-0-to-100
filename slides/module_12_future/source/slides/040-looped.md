:::divider id="divider-looped" title="Assumption Two: Every Token Gets the Same Computation" sub="Looped and recurrent depth"
:::

---


<!-- .slide: id="looped-assumption" -->

## Fixed Depth

The transformer has $L$ layers. Every token passes through all $L$ of them.

:::columns cols="2" gap="34px"
**The word "the"**

Trivially predictable from context. Gets $L$ layers of computation.
+++
**The last step of a hard proof**

Requires several dependent inferences. Gets $L$ layers of computation.
:::

Depth is fixed at training time. The architecture cannot spend more computation on a harder token. <!-- .element: class="text-lg" style="margin-top: 14px;" -->

---

<!-- .slide: id="looped-cot" -->

## Chain of Thought: The Deployed Solution

Reasoning models buy more computation per answer by **emitting more tokens.** It works and it shipped. It accepts a constraint:

<div class="card-grid cols-3">
<div class="card"><h4>To compute more</h4><p>The model must <strong>serialize its thinking into words</strong>.</p></div>
<div class="card"><h4>Then read it back</h4><p>Through its own context window, one token at a time.</p></div>
<div class="card"><h4>Paying full price</h4><p>Every thought costs a forward pass, a KV cache entry, and output tokens you are billed for.</p></div>
</div>

More computation does not have to take the form of words. That gap is the rest of the section. <!-- .element: class="text-lg" style="margin-top: 14px;" -->

---

:::figure img="images/alex_graves.jpg" name="Alex Graves" kicker="Adaptive Computation Time (2016), Neural Turing Machines (2014)" alt="Alex Graves"
Asked both of this lecture's remaining questions a decade early. Adaptive Computation Time: a network decides **for itself** how many steps to spend on each input, with a learned halting rule. That is this section. Neural Turing Machines: give a network an external memory it can read and write during the forward pass. That is the next section.

His work on connectionist temporal classification and LSTM sequence generation is a large part of why recurrent networks worked well enough in the 2010s to be worth replacing.
:::

---

<!-- .slide: id="looped-loop" -->

## Looped Transformers

An alternative way to spend more computation: **apply the same block repeatedly.**

:::columns cols="2" gap="34px"
**Universal Transformers (2018)**

- Shared weights applied recurrently in depth
- Learned halting rule per position
- Predates GPT-3
+++
**What changes**

- Depth becomes a **runtime dial**, not a training-time constant
- Same weights: four iterations on an easy token, forty on a hard one
:::

Recurrence in *depth*, not in sequence. Linear attention looped over tokens; this loops over layers. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="looped-diagram" -->

## Fixed vs Looped Depth

<div class="svg-figure">
<svg viewBox="0 0 1000 380" role="img" aria-label="A fixed-depth transformer sends both an easy token and a hard token through the same six layers. A looped transformer sends the easy token through a two-layer block twice and the hard token through the same block eight times.">
  <text x="20" y="32" fill="#8892a4" font-size="21" font-family="Inter, sans-serif">Fixed depth: six layers, whatever the token</text>
  <g fill="none" stroke="#4a9eff" stroke-width="2">
    <rect x="40" y="66" width="150" height="26" /><rect x="40" y="98" width="150" height="26" /><rect x="40" y="130" width="150" height="26" /><rect x="40" y="162" width="150" height="26" /><rect x="40" y="194" width="150" height="26" /><rect x="40" y="226" width="150" height="26" />
    <rect x="250" y="66" width="150" height="26" /><rect x="250" y="98" width="150" height="26" /><rect x="250" y="130" width="150" height="26" /><rect x="250" y="162" width="150" height="26" /><rect x="250" y="194" width="150" height="26" /><rect x="250" y="226" width="150" height="26" />
  </g>
  <text x="115" y="292" fill="#e8eaf0" font-size="21" font-family="Inter, sans-serif" text-anchor="middle">"the"</text>
  <text x="325" y="292" fill="#e8eaf0" font-size="21" font-family="Inter, sans-serif" text-anchor="middle">last step of a proof</text>
  <text x="220" y="330" fill="#f5a623" font-size="20" font-family="Inter, sans-serif" text-anchor="middle">6 layers each, decided at training time</text>
  <line x1="470" y1="20" x2="470" y2="350" stroke="#2a3450" stroke-width="2" />
  <text x="520" y="32" fill="#8892a4" font-size="21" font-family="Inter, sans-serif">Looped depth: one block, run as needed</text>
  <g fill="none" stroke="#50c878" stroke-width="2">
    <rect x="560" y="130" width="150" height="26" /><rect x="560" y="162" width="150" height="26" />
    <rect x="790" y="130" width="150" height="26" /><rect x="790" y="162" width="150" height="26" />
  </g>
  <path d="M 545 143 a 26 34 0 1 0 0 62" stroke="#50c878" stroke-width="2" fill="none" />
  <path d="M 545 205 l 6 -9 l -11 -2 z" fill="#50c878" />
  <path d="M 775 143 a 26 34 0 1 0 0 62" stroke="#50c878" stroke-width="2" fill="none" />
  <path d="M 775 205 l 6 -9 l -11 -2 z" fill="#50c878" />
  <text x="635" y="112" fill="#50c878" font-size="21" font-family="Inter, sans-serif" text-anchor="middle">&#215; 2</text>
  <text x="865" y="112" fill="#50c878" font-size="21" font-family="Inter, sans-serif" text-anchor="middle">&#215; 8</text>
  <text x="635" y="292" fill="#e8eaf0" font-size="21" font-family="Inter, sans-serif" text-anchor="middle">"the"</text>
  <text x="865" y="292" fill="#e8eaf0" font-size="21" font-family="Inter, sans-serif" text-anchor="middle">last step of a proof</text>
  <text x="750" y="330" fill="#50c878" font-size="20" font-family="Inter, sans-serif" text-anchor="middle">same weights, count decided at inference</text>
</svg>
</div>

---

<!-- .slide: id="looped-theory" -->

## Fixed Depth Limits Expressiveness

A fixed-depth network computes a **fixed-length circuit.**

:::columns cols="2" gap="34px"
**The limit**

- A problem needing more sequential steps than layers is out of reach
- Width does not help: it buys parallel work, not more steps
+++
**What a loop does**

- Runs the same parameters a **variable** number of steps
- Expresses iterative algorithms a fixed stack cannot represent
:::

This is expressiveness, not efficiency. Some problems are not expensive for a fixed-depth transformer; they are unreachable. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="looped-latent" -->

## Latent Reasoning

More iterations at inference improves quality. **No extra tokens emitted.**

Geiping et al. (2025): a 3.5B recurrent-depth model, unrolled at test time, reaches the compute load of a **50B model** and improves reasoning benchmarks.

:::note variant="hint"
Compute-equivalent, not performance-equivalent. Matching a 50B model's operations per answer is not matching a well-trained 50B model's benchmark scores. "A 3.5B model matches a 50B model" is the wrong summary.
:::

---

<!-- .slide: id="looped-coconut" -->

## Coconut: Latent Chain of Thought

**Coconut**: feed the final hidden state back as the next input embedding, instead of decoding it into a token.

:::columns cols="2" gap="34px"
**Chain of thought**

A chain of **tokens**. Legible and supervisable, but forced through the vocabulary bottleneck.
+++
**Chain of continuous thought**

A chain of **vectors**. No vocabulary bottleneck and no tokens emitted, but also nothing to read or audit.
:::

Both approaches separate **compute spent per answer** from **tokens emitted per answer**, two quantities this course has treated as one. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="looped-mod" -->

## Adaptive Compute: Width and Depth

<div class="bench-table">
<table>
<thead><tr><th>Axis</th><th>Mechanism</th><th>Effect</th></tr></thead>
<tbody>
<tr><td><strong>Width</strong></td><td>Mixture of experts</td><td>Each token activates a few experts out of many. Compute per token is sparse across width</td></tr>
<tr><td><strong>Depth</strong></td><td>Mixture-of-depths</td><td>Each token skips some layers entirely. Compute per token is sparse across depth</td></tr>
<tr><td><strong>Depth, unbounded</strong></td><td>Looped transformers</td><td>Each token can take <em>more</em> steps than the network has layers</td></tr>
</tbody>
</table>
</div>

All three mechanisms avoid spending the same compute on every token. <!-- .element: class="text-lg" style="margin-top: 14px;" -->

---

<!-- .slide: id="looped-verdict" -->

## Why Verbalized Reasoning Won

Looped transformers are parameter-efficient and date to 2018, yet verbalized chain of thought shipped. Two reasons:

:::columns cols="2" gap="34px"
**Tooling**

- Serving, billing, and batching assume fixed work per token
- Variable depth breaks latency prediction and throughput planning
+++
**Supervision**

- RL needs something to score: you can **reward a transcript you can read**
- Latent reasoning is hard to supervise for the same reason it is efficient: nobody can see it
:::

You cannot audit reasoning that was never written down. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="sidequest-hardware" -->

## Side Quest: The Hardware Lottery

Sara Hooker (2020): ideas win or lose partly on **how well they fit the hardware of their era**. Transformers saturate GPUs; recurrent models do not, and lost the 2010s partly for it.

:::columns cols="2" gap="34px"
**Applied to this section**

- Looped depth is parameter-efficient
- The incumbent holds the **hardware**, the **serving stack**, and the **RL tooling**
- A challenger must beat all three at once
+++
**Applied to diffusion**

- Block diffusion exists specifically so diffusion can keep the KV cache
:::

The idea that wins is not the best idea. It is the best idea that runs well on existing hardware. <!-- .element: class="text-lg" style="margin-top: 12px;" -->
