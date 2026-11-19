<!-- .slide: id="review-purpose" -->

## What This Module Covers

:::columns cols="2" gap="34px"
**First: the review**

- Each module in one line
- The five most durable concepts
- How fast each layer goes stale
+++
**Then: the lecture**

- Is attention the right primitive?
- Three assumptions the course treated as fixed
- Whether networks can run programs
- Where agents go next
:::

This module covers active research. Parts of it will be out of date within a few years. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="review-stack" -->

## Course Overview

<div class="bench-table">
<table>
<thead><tr><th>Module</th><th>What it added</th></tr></thead>
<tbody>
<tr><td>1 &middot; Introduction</td><td>Tokens, probability, and entropy make text a <strong>prediction problem</strong></td></tr>
<tr><td>2 &middot; Perceptrons</td><td>Gradient descent makes any differentiable model <strong>learnable</strong></td></tr>
<tr><td>3 &middot; Attention</td><td>A <strong>soft lookup</strong> over the whole sequence, with no recurrence</td></tr>
<tr><td>4 &middot; Architectures</td><td>The <strong>transformer block</strong>: attention, an MLP, residuals, normalization</td></tr>
<tr><td>5 &middot; Pretraining</td><td><strong>Capability</strong>, and loss that falls as a power law in compute</td></tr>
<tr><td>6 &middot; Finetuning</td><td>An <strong>interface</strong>: the chat template, instruction following</td></tr>
<tr><td>7 &middot; RL</td><td><strong>Judgment</strong>: reasoning trained in with verifiable rewards</td></tr>
<tr><td>8 &middot; Multimodal</td><td>Other modalities become <strong>tokens in the same sequence</strong></td></tr>
<tr><td>9 &middot; Evaluation</td><td>Whether any of it <strong>actually worked</strong></td></tr>
<tr><td>10 &middot; Deployment</td><td>Weights become <strong>tokens per second</strong> at a price</td></tr>
<tr><td>11 &middot; Applications</td><td>Prompts, retrieval, tools, and loops turn an API into a <strong>product</strong></td></tr>
</tbody>
</table>
</div>

---

<!-- .slide: id="review-throughline" -->

## Everything Is Next-Token Prediction

Every module worked on one problem: **predicting the next token.**

:::columns cols="3" gap="26px"
**Modules 1 to 4: represent**

- Count
- Learn
- Attend
- Stack
+++
**Modules 5 to 8: train**

- Scale
- Instruct
- Judge
- Add modalities
+++
**Modules 9 to 11: check and ship**

- Measure
- Serve
- Build a product
:::

RL, retrieval, and agents are all built on top of next-token prediction. <!-- .element: class="text-lg" style="margin-top: 14px;" -->

---

<!-- .slide: id="review-keepers" -->

## The Five Most Durable Concepts

None of these depend on any particular architecture.

<div class="card-grid cols-3">
<div class="card"><h4>Information theory</h4><p>Cross-entropy is <strong>average surprise</strong>. Every loss you minimized was a bit count.</p></div>
<div class="card"><h4>Gradient descent</h4><p>The only learning algorithm this course ever used. Everything else was a choice of what to differentiate.</p></div>
<div class="card"><h4>Attention</h4><p>Attention is $\mathrm{softmax}(QK^\top/\sqrt{d_k})V$ and nothing more.</p></div>
<div class="card"><h4>Scaling laws</h4><p>Loss falls <strong>predictably</strong> with compute. That predictability is why anyone spends a billion dollars on a training run.</p></div>
<div class="card"><h4>Evaluation</h4><p>You cannot improve what you <strong>cannot measure</strong>, and most measurements are worse than they look.</p></div>
</div>

---

<!-- .slide: id="review-halflife" -->

## Half-Life of the Material

Which parts of the course are worth relearning in five years, and which are not.

<div class="bench-table">
<table>
<thead><tr><th>Layer</th><th>Half-life</th><th>Status</th></tr></thead>
<tbody>
<tr><td>The math of Modules 1 and 2</td><td><strong>Permanent</strong></td><td>Entropy and gradient descent are not going to be revised</td></tr>
<tr><td>The transformer (Modules 3, 4)</td><td><strong>Dominant, contested</strong></td><td>Nine years old, still winning, with serious challengers</td></tr>
<tr><td>The recipe (Modules 5 to 7)</td><td><strong>Converging, moving</strong></td><td>Pretrain, finetune, RL is now standard, but the mix shifts yearly</td></tr>
<tr><td>The application layer (Module 11)</td><td><strong>Turns over yearly</strong></td><td>Frameworks and patterns you learned this month may not survive the year</td></tr>
</tbody>
</table>
</div>

The foundations at the bottom of the table stay useful the longest. <!-- .element: class="text-lg" style="margin-top: 14px;" -->

---

<!-- .slide: id="sidequest-gnn" -->

## <span class="side-quest-badge">Side Quest</span> Transformers Are Graph Neural Networks

Attention is **message passing on a fully connected graph:**

- Every token is a node
- Attention weights are soft edges
- Value aggregation is the message step

:::columns cols="2" gap="34px"
**Graph neural networks**

The graph is given as input, and a node only exchanges messages with its neighbors.
+++
**Attention**

No graph is given: every token can attend to every other, and the weights on those edges are learned per input. That costs $O(n^2)$.
:::

Any nodes work: image patches, atoms, functions, users. The GNN literature covers the same machinery. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="review-roadmap" -->

## Assumptions Built Into Today's LLMs

Every model in this course made the same seven choices. Each one is being tested.

<div class="bench-table dense">
<table>
<thead><tr><th>The assumption</th><th>What replaces it</th></tr></thead>
<tbody>
<tr><td>The model predicts <strong>tokens</strong> from a fixed vocabulary</td><td>Byte-level models and concept models</td></tr>
<tr><td>Every token <strong>attends to every earlier token</strong></td><td>Linear attention, state-space models, and hybrids</td></tr>
<tr><td>Text is generated <strong>left to right</strong></td><td>Diffusion language models</td></tr>
<tr><td>Every token gets the <strong>same computation</strong></td><td>Looped depth and latent reasoning</td></tr>
<tr><td>Networks <strong>approximate</strong> rather than run exact programs</td><td>Compiled transformers and neurosymbolic systems</td></tr>
<tr><td>Learning <strong>stops at deployment</strong></td><td>Test-time training and nested learning</td></tr>
<tr><td>One model serves <strong>one user</strong> and spends a person's money</td><td>Agent swarms, shared sessions, and self-funding agents</td></tr>
</tbody>
</table>
</div>
