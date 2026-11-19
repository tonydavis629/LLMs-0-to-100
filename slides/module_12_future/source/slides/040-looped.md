:::divider id="divider-looped" title="Spending More Compute on Hard Tokens" sub="Assumption two: every token gets the same computation"
:::

---


<!-- .slide: id="looped-assumption" -->

## Every Token Gets the Same Depth

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

## Chain of Thought Buys Compute With Tokens

Reasoning models buy more computation per answer by **emitting more tokens.** It works and it shipped. It accepts a constraint:

<div class="card-grid cols-3">
<div class="card"><h4>To compute more</h4><p>The model must <strong>serialize its thinking into words</strong>.</p></div>
<div class="card"><h4>Then read it back</h4><p>Through its own context window, one token at a time.</p></div>
<div class="card"><h4>Paying full price</h4><p>Every thought costs a forward pass, a KV cache entry, and output tokens you are billed for.</p></div>
</div>

More computation does not have to take the form of words. That gap is the rest of the section. <!-- .element: class="text-lg" style="margin-top: 14px;" -->

---

:::figure img="images/alex_graves.jpg" name="Alex Graves" kicker="Adaptive Computation Time (2016), Neural Turing Machines (2014)" alt="Alex Graves"
Two of his ideas came a decade early. Adaptive Computation Time lets a network decide **for itself** how many steps to spend on each input, with a learned halting rule. Neural Turing Machines give a network an external memory it can read and write during the forward pass.

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

## Recurrent Depth Unrolls One Block at Test Time

<div class="svg-figure">
<svg viewBox="0 0 1000 300" role="img" aria-label="Tokens enter a prelude that embeds them once. The embedding feeds a recurrent block that starts from a random state and runs r times with the same weights, with the embedding re-entering on every pass. A coda decodes the final state into the next token.">
  <rect x="20" y="130" width="100" height="56" fill="none" stroke="#8892a4" stroke-width="2" />
  <text x="70" y="164" fill="#e8eaf0" font-size="19" font-family="Inter, sans-serif" text-anchor="middle">tokens</text>
  <path d="M 120 158 L 162 158" stroke="#8892a4" stroke-width="2" fill="none" />
  <path d="M 162 158 l -10 -5 l 0 10 z" fill="#8892a4" />
  <rect x="170" y="130" width="140" height="56" fill="none" stroke="#4a9eff" stroke-width="2" />
  <text x="240" y="156" fill="#4a9eff" font-size="20" font-family="Inter, sans-serif" text-anchor="middle">Prelude</text>
  <text x="240" y="178" fill="#8892a4" font-size="16" font-family="Inter, sans-serif" text-anchor="middle">embed once</text>
  <path d="M 310 158 L 412 158" stroke="#4a9eff" stroke-width="2" fill="none" />
  <path d="M 412 158 l -10 -5 l 0 10 z" fill="#4a9eff" />
  <text x="360" y="148" fill="#4a9eff" font-size="18" font-family="Inter, sans-serif" text-anchor="middle">e</text>
  <rect x="420" y="118" width="200" height="80" fill="none" stroke="#50c878" stroke-width="3" />
  <text x="520" y="152" fill="#50c878" font-size="21" font-family="Inter, sans-serif" text-anchor="middle">Recurrent block</text>
  <text x="520" y="178" fill="#8892a4" font-size="16" font-family="Inter, sans-serif" text-anchor="middle">same weights every pass</text>
  <path d="M 600 118 C 600 50, 440 50, 440 112" stroke="#50c878" stroke-width="2" fill="none" />
  <path d="M 440 114 l -6 -11 l 12 0 z" fill="#50c878" />
  <text x="520" y="46" fill="#50c878" font-size="20" font-family="Inter, sans-serif" text-anchor="middle">repeat r times, e re-enters each pass</text>
  <path d="M 520 262 L 520 206" stroke="#8892a4" stroke-width="2" stroke-dasharray="6 5" fill="none" />
  <path d="M 520 200 l -6 11 l 12 0 z" fill="#8892a4" />
  <text x="520" y="284" fill="#8892a4" font-size="17" font-family="Inter, sans-serif" text-anchor="middle">starts from a random state s&#8320;</text>
  <path d="M 620 158 L 692 158" stroke="#50c878" stroke-width="2" fill="none" />
  <path d="M 692 158 l -10 -5 l 0 10 z" fill="#50c878" />
  <rect x="700" y="130" width="140" height="56" fill="none" stroke="#4a9eff" stroke-width="2" />
  <text x="770" y="156" fill="#4a9eff" font-size="20" font-family="Inter, sans-serif" text-anchor="middle">Coda</text>
  <text x="770" y="178" fill="#8892a4" font-size="16" font-family="Inter, sans-serif" text-anchor="middle">decode once</text>
  <path d="M 840 158 L 872 158" stroke="#4a9eff" stroke-width="2" fill="none" />
  <path d="M 872 158 l -10 -5 l 0 10 z" fill="#4a9eff" />
  <text x="925" y="164" fill="#e8eaf0" font-size="19" font-family="Inter, sans-serif" text-anchor="middle">next token</text>
</svg>
</div>

- Geiping et al. (2025) trained a 3.5B model this way, sampling $r$ at random so one model works at any depth
- More passes at test time improves reasoning scores. **No extra tokens are emitted**
- Unrolled far enough, it spends the compute of a **50B model** per answer

:::note variant="hint"
That is compute-equivalent, not performance-equivalent. Doing a 50B model's arithmetic is not matching a well-trained 50B model's scores.
:::

---

<!-- .slide: id="looped-coconut" -->

## Coconut Thinks in Hidden States

<div class="svg-figure">
<svg viewBox="0 0 1000 290" role="img" aria-label="Chain of thought: each forward pass decodes its last hidden state into a token, which is embedded and fed to the next pass. Coconut: the last hidden state is fed straight back as the next input, skipping the vocabulary, until the model switches back to language for the answer.">
  <text x="20" y="32" fill="#4a9eff" font-size="21" font-family="Inter, sans-serif">Chain of thought: decode to a word, then read it back</text>
  <rect x="20" y="56" width="120" height="48" fill="none" stroke="#8892a4" stroke-width="2" />
  <text x="80" y="86" fill="#e8eaf0" font-size="18" font-family="Inter, sans-serif" text-anchor="middle">question</text>
  <g fill="none" stroke="#e8eaf0" stroke-width="2">
    <rect x="180" y="56" width="80" height="48" /><rect x="420" y="56" width="80" height="48" /><rect x="660" y="56" width="80" height="48" />
  </g>
  <g fill="#e8eaf0" font-size="18" font-family="Inter, sans-serif" text-anchor="middle">
    <text x="220" y="86">LM</text><text x="460" y="86">LM</text><text x="700" y="86">LM</text>
  </g>
  <g fill="#4a9eff" fill-opacity="0.3" stroke="#4a9eff" stroke-width="2">
    <rect x="290" y="62" width="100" height="36" /><rect x="530" y="62" width="100" height="36" />
  </g>
  <g fill="#e8eaf0" font-size="17" font-family="Inter, sans-serif" text-anchor="middle">
    <text x="340" y="86">"so 12"</text><text x="580" y="86">"then 7"</text>
  </g>
  <g stroke="#8892a4" stroke-width="2" fill="none">
    <path d="M 140 80 L 178 80" /><path d="M 260 80 L 288 80" /><path d="M 390 80 L 418 80" /><path d="M 500 80 L 528 80" /><path d="M 630 80 L 658 80" /><path d="M 740 80 L 778 80" />
  </g>
  <rect x="780" y="56" width="120" height="48" fill="none" stroke="#8892a4" stroke-width="2" />
  <text x="840" y="86" fill="#e8eaf0" font-size="18" font-family="Inter, sans-serif" text-anchor="middle">answer</text>
  <text x="460" y="128" fill="#8892a4" font-size="16" font-family="Inter, sans-serif" text-anchor="middle">each thought squeezed into one token from the vocabulary</text>
  <line x1="20" y1="148" x2="980" y2="148" stroke="#2a3450" stroke-width="2" />
  <text x="20" y="184" fill="#50c878" font-size="21" font-family="Inter, sans-serif">Coconut: feed the hidden state straight back</text>
  <rect x="20" y="206" width="120" height="48" fill="none" stroke="#8892a4" stroke-width="2" />
  <text x="80" y="236" fill="#e8eaf0" font-size="18" font-family="Inter, sans-serif" text-anchor="middle">question</text>
  <g fill="none" stroke="#e8eaf0" stroke-width="2">
    <rect x="180" y="206" width="80" height="48" /><rect x="420" y="206" width="80" height="48" /><rect x="660" y="206" width="80" height="48" />
  </g>
  <g fill="#e8eaf0" font-size="18" font-family="Inter, sans-serif" text-anchor="middle">
    <text x="220" y="236">LM</text><text x="460" y="236">LM</text><text x="700" y="236">LM</text>
  </g>
  <g fill="#50c878" fill-opacity="0.45" stroke="#50c878" stroke-width="1">
    <rect x="310" y="212" width="12" height="36" /><rect x="326" y="212" width="12" height="36" /><rect x="342" y="212" width="12" height="36" /><rect x="358" y="212" width="12" height="36" />
    <rect x="550" y="212" width="12" height="36" /><rect x="566" y="212" width="12" height="36" /><rect x="582" y="212" width="12" height="36" /><rect x="598" y="212" width="12" height="36" />
  </g>
  <g stroke="#50c878" stroke-width="2" fill="none">
    <path d="M 140 230 L 178 230" /><path d="M 260 230 L 306 230" /><path d="M 374 230 L 418 230" /><path d="M 500 230 L 546 230" /><path d="M 614 230 L 658 230" /><path d="M 740 230 L 778 230" />
  </g>
  <rect x="780" y="206" width="120" height="48" fill="none" stroke="#8892a4" stroke-width="2" />
  <text x="840" y="236" fill="#e8eaf0" font-size="18" font-family="Inter, sans-serif" text-anchor="middle">answer</text>
  <text x="460" y="280" fill="#8892a4" font-size="16" font-family="Inter, sans-serif" text-anchor="middle">each thought is a full hidden vector, never turned into a word</text>
</svg>
</div>

- Trained by a curriculum: replace the first written reasoning step with a hidden state, then the first two, and so on
- One hidden thought can hold **several candidate next steps** at once, so the model explores like a breadth-first search
- Mixed results: better than chain of thought on a logic-planning task, worse on grade-school math

---

<!-- .slide: id="looped-trm" -->

## A Tiny Recursive Model Beats Large LLMs on ARC

<div class="svg-figure">
<svg viewBox="0 0 1000 290" role="img" aria-label="The tiny recursive model keeps a question x, an answer y, and a latent scratchpad z. An inner loop updates z from x, y, and z several times. Then the model updates y from y and z. The outer loop repeats up to 16 times.">
  <rect x="30" y="60" width="150" height="56" fill="none" stroke="#8892a4" stroke-width="2" />
  <text x="105" y="94" fill="#e8eaf0" font-size="19" font-family="Inter, sans-serif" text-anchor="middle">question x</text>
  <path d="M 180 88 L 322 88" stroke="#8892a4" stroke-width="2" fill="none" />
  <path d="M 322 88 l -10 -5 l 0 10 z" fill="#8892a4" />
  <rect x="330" y="60" width="180" height="56" fill="none" stroke="#50c878" stroke-width="3" />
  <text x="420" y="94" fill="#50c878" font-size="19" font-family="Inter, sans-serif" text-anchor="middle">latent z</text>
  <path d="M 490 60 C 490 30, 350 30, 350 54" stroke="#50c878" stroke-width="2" fill="none" />
  <path d="M 350 56 l -6 -11 l 12 0 z" fill="#50c878" />
  <text x="420" y="24" fill="#50c878" font-size="17" font-family="Inter, sans-serif" text-anchor="middle">think n times: z &#8592; f(x, y, z)</text>
  <path d="M 420 116 L 420 182" stroke="#f5a623" stroke-width="2" fill="none" />
  <path d="M 420 188 l -6 -11 l 12 0 z" fill="#f5a623" />
  <text x="434" y="156" fill="#f5a623" font-size="17" font-family="Inter, sans-serif">act: y &#8592; g(y, z)</text>
  <rect x="330" y="190" width="180" height="56" fill="none" stroke="#f5a623" stroke-width="3" />
  <text x="420" y="224" fill="#f5a623" font-size="19" font-family="Inter, sans-serif" text-anchor="middle">answer y</text>
  <path d="M 510 218 C 590 218, 590 88, 518 88" stroke="#8892a4" stroke-width="2" fill="none" />
  <path d="M 516 88 l 11 -6 l 0 12 z" fill="#8892a4" />
  <text x="420" y="278" fill="#8892a4" font-size="17" font-family="Inter, sans-serif" text-anchor="middle">outer loop: repeat up to 16 times</text>
  <line x1="660" y1="20" x2="660" y2="270" stroke="#2a3450" stroke-width="2" />
  <text x="690" y="70" fill="#e8eaf0" font-size="20" font-family="Inter, sans-serif">7M parameters, 2 layers</text>
  <text x="690" y="120" fill="#50c878" font-size="20" font-family="Inter, sans-serif">ARC-AGI-1: 45%</text>
  <text x="690" y="160" fill="#50c878" font-size="20" font-family="Inter, sans-serif">ARC-AGI-2: 8%</text>
  <text x="690" y="210" fill="#8892a4" font-size="17" font-family="Inter, sans-serif">above DeepSeek R1, o3-mini,</text>
  <text x="690" y="234" fill="#8892a4" font-size="17" font-family="Inter, sans-serif">and Gemini 2.5 Pro</text>
</svg>
</div>

- **TRM** (Jolicoeur-Martineau, 2025) simplified **HRM** (2025), which ran a fast low-level module and a slow high-level one
- Two tiny layers, applied over and over, give an effective depth of dozens of layers
- Caveat: TRM is trained for each puzzle family and does not read or write language

---

<!-- .slide: id="looped-where" -->

## Three Places to Put the Thinking

<div class="svg-figure">
<svg viewBox="0 0 1000 250" role="img" aria-label="A spectrum of three designs. Chain of thought thinks in written tokens, which anyone can read. Coconut thinks in hidden vectors between tokens, which can be partly decoded. Recurrent depth and tiny recursive models think inside the forward pass and leave no trace. Moving right spends fewer tokens and is harder to read.">
  <g fill="none" stroke-width="2">
    <rect x="20" y="20" width="300" height="140" rx="8" stroke="#4a9eff" /><rect x="350" y="20" width="300" height="140" rx="8" stroke="#a06bd4" /><rect x="680" y="20" width="300" height="140" rx="8" stroke="#50c878" />
  </g>
  <text x="170" y="56" fill="#4a9eff" font-size="21" font-family="Inter, sans-serif" text-anchor="middle">Chain of thought</text>
  <text x="170" y="94" fill="#e8eaf0" font-size="17" font-family="Inter, sans-serif" text-anchor="middle">thinks in written tokens</text>
  <text x="170" y="126" fill="#8892a4" font-size="17" font-family="Inter, sans-serif" text-anchor="middle">anyone can read the trace</text>
  <text x="500" y="56" fill="#a06bd4" font-size="21" font-family="Inter, sans-serif" text-anchor="middle">Coconut</text>
  <text x="500" y="94" fill="#e8eaf0" font-size="17" font-family="Inter, sans-serif" text-anchor="middle">thinks in vectors between tokens</text>
  <text x="500" y="126" fill="#8892a4" font-size="17" font-family="Inter, sans-serif" text-anchor="middle">probes can partly decode it</text>
  <text x="830" y="56" fill="#50c878" font-size="21" font-family="Inter, sans-serif" text-anchor="middle">Recurrent depth, TRM</text>
  <text x="830" y="94" fill="#e8eaf0" font-size="17" font-family="Inter, sans-serif" text-anchor="middle">thinks inside the forward pass</text>
  <text x="830" y="126" fill="#8892a4" font-size="17" font-family="Inter, sans-serif" text-anchor="middle">leaves no trace at all</text>
  <path d="M 40 200 L 950 200" stroke="#8892a4" stroke-width="2" fill="none" />
  <path d="M 960 200 l -12 -6 l 0 12 z" fill="#8892a4" />
  <text x="500" y="232" fill="#8892a4" font-size="18" font-family="Inter, sans-serif" text-anchor="middle">fewer tokens spent, and less to read or supervise</text>
</svg>
</div>

Korbak et al. (2025), with authors from several labs, call chain-of-thought monitoring a fragile safety opportunity: reading the trace is one of the few ways to check what a model is doing. Latent reasoning removes that trace. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="looped-mod" -->

## Adaptive Compute in Width and Depth

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

## <span class="side-quest-badge">Side Quest</span> The Hardware Lottery

Sara Hooker (2020): ideas win or lose partly on **how well they fit the hardware of their era**. Transformers saturate GPUs; recurrent models do not, and lost the 2010s partly for it.

:::columns cols="2" gap="34px"
**Applied to looped depth**

- Looped depth is parameter-efficient
- The incumbent holds the **hardware**, the **serving stack**, and the **RL tooling**
- A challenger must beat all three at once
+++
**Applied to diffusion**

- Block diffusion exists specifically so diffusion can keep the KV cache
:::

The idea that wins is not the best idea. It is the best idea that runs well on existing hardware. <!-- .element: class="text-lg" style="margin-top: 12px;" -->
