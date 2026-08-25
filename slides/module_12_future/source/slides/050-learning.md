:::divider id="divider-learning" title="Assumption Three: Learning Stops at Deployment" sub="Test-time training"
:::

---

<!-- .slide: id="learning-assumption" -->

## The Weights Are Frozen

- Everything a deployed model appears to learn lives in the **context window**
- The session ends and it is lost
- Debug a problem Monday, the model retains nothing Tuesday

:::columns cols="2" gap="34px"
**Workaround: retrieval**

- Put the past in the prompt
- Costs tokens on every call
- The model still has not learned anything
+++
**Workaround: finetuning**

- Bake the past into the weights
- Slow, offline cycle
- Nothing like learning from experience as you go
:::

Neither gives the model accumulated experience. This section covers approaches that would. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="learning-forgetting" -->

## Catastrophic Forgetting

**Catastrophic forgetting** (McCloskey and Cohen, 1989):

- Train on new data and old capabilities degrade
- Gradient descent on the new task walks the weights away from the old one
- Nothing in the objective says not to

<div class="card-grid cols-3">
<div class="card"><h4>Replay</h4><p>Mix old data in with the new. Effective, and it means you never actually stop paying for the old data.</p></div>
<div class="card"><h4>Constrain</h4><p>Penalize movement in weights that mattered before (elastic weight consolidation). Needs to know which ones mattered.</p></div>
<div class="card"><h4>Isolate</h4><p>Put new learning in added parameters. This is LoRA, reused as a <strong>memory mechanism</strong> rather than an efficiency trick.</p></div>
</div>

None of the three is a solved recipe at this scale. <!-- .element: class="text-lg" style="margin-top: 14px;" -->

---

<!-- .slide: id="learning-ttt" -->

## Test-Time Training: A Training Loop Inside the Forward Pass

Make the hidden state a **small neural network**. Update it with real gradient steps as the context streams past.

:::columns cols="2" gap="34px"
**What the state was**

A summary: a fixed-size buffer written by a hand-designed update rule.
+++
**What the state becomes**

A **model**, trained on this document by gradient descent while the document is read.
:::

- Sanity check: with a linear inner model, this reduces to linear attention. You write that update by hand in the exercise.
- The ancestor is Schmidhuber's fast weights (**1992**): a network that generates weight updates for another network during the forward pass.

---

<!-- .slide: id="learning-loops" -->

## The Two Nested Loops

:::columns cols="2" gap="34px"
**Outer loop: pretraining**

- Slow, expensive, done once
- Learns weights that make the inner loop **useful**
- Learns *how to learn*
+++
**Inner loop: inference**

- Runs per token, on one sequence
- Learns the **document in front of it**
- Learns *the current input*
:::

Every model in this course had the outer loop. The proposal: make the inner loop a real optimizer too. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="learning-diagram" -->

## Nested Optimization at Two Timescales

<div class="svg-figure">
<svg viewBox="0 0 1000 360" role="img" aria-label="An outer loop of pretraining runs once over a corpus and produces weights. Inside the deployed model, an inner loop runs at inference, taking a gradient step per token on the sequence in front of it.">
  <rect x="20" y="24" width="960" height="310" rx="10" fill="none" stroke="#4a9eff" stroke-width="2" />
  <text x="44" y="58" fill="#4a9eff" font-size="22" font-family="Inter, sans-serif">Outer loop: pretraining</text>
  <text x="44" y="88" fill="#8892a4" font-size="19" font-family="Inter, sans-serif">runs once, over a corpus, for months</text>
  <text x="44" y="118" fill="#8892a4" font-size="19" font-family="Inter, sans-serif">learns weights that make the inner loop useful</text>
  <path d="M 300 152 L 300 190" stroke="#4a9eff" stroke-width="2" fill="none" />
  <path d="M 300 190 l -5 -10 l 10 0 z" fill="#4a9eff" />
  <text x="316" y="178" fill="#8892a4" font-size="18" font-family="Inter, sans-serif">produces the deployed model</text>
  <rect x="60" y="200" width="880" height="112" rx="8" fill="none" stroke="#50c878" stroke-width="2" />
  <text x="84" y="236" fill="#50c878" font-size="22" font-family="Inter, sans-serif">Inner loop: at inference</text>
  <text x="84" y="266" fill="#8892a4" font-size="19" font-family="Inter, sans-serif">runs per token, on this one sequence, for microseconds</text>
  <text x="84" y="296" fill="#8892a4" font-size="19" font-family="Inter, sans-serif">learns the document in front of it, then throws it away</text>
  <path d="M 890 226 a 30 30 0 1 1 -22 -29" stroke="#50c878" stroke-width="2" fill="none" />
  <path d="M 868 197 l 2 11 l 9 -6 z" fill="#50c878" />
  <text x="862" y="292" fill="#50c878" font-size="18" font-family="Inter, sans-serif" text-anchor="middle">per token</text>
</svg>
</div>

The outer loop learns how to learn. The inner loop learns the current document. Nested learning asks why there should be only two levels. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="learning-titans" -->

## Titans: Persistent Memory

**Titans** pushes the inner loop from a per-sequence scratchpad toward persistent memory, with a rule for what gets written:

<div class="metric-box">

Store what the model predicted **badly**.

</div>

- A correctly predicted token carries no new information; a surprising one does
- Cross-entropy becomes a **memory-writing policy**, not a loss

**Nested learning** generalizes the frame:

- A model and its optimizer are **levels of one optimization** at different update frequencies
- Pretraining updates slowly, the inner loop updates per token
- No reason in principle to stop at two levels

---

<!-- .slide: id="learning-stakes" -->

## Consequences for Deployment

If this assumption falls, "deploying a model" stops meaning "shipping a frozen artifact."

<div class="bench-table">
<table>
<thead><tr><th>What breaks</th><th>Why</th></tr></thead>
<tbody>
<tr><td><strong>Evaluation</strong></td><td>Evaluate then ship assumes the thing you tested is the thing you serve. A learning model's eval results have an <strong>expiry date</strong></td></tr>
<tr><td><strong>Serving</strong></td><td>Weights are a read-only asset replicated across GPUs. Replicas that keep learning <strong>drift apart</strong>, and now you have a consistency problem</td></tr>
<tr><td><strong>Applications</strong></td><td>Retrieval exists because the model cannot remember. Some of that machinery becomes unnecessary, and some becomes load-bearing in a new way</td></tr>
</tbody>
</table>
</div>

---

<!-- .slide: id="learning-verdict" -->

## Current Status

The least settled of the three sections, by a wide margin.

<div class="card-grid cols-3">
<div class="card"><h4>Published</h4><p>Test-time training layers exist and work, demonstrated at <strong>small scale</strong>.</p></div>
<div class="card"><h4>Research program</h4><p>Nested learning is a framing and a set of results, <strong>not a product</strong>.</p></div>
<div class="card"><h4>Still open</h4><p>The distance between a frozen model you prompt and a system that <strong>accumulates experience over months</strong> is wide.</p></div>
</div>

The assumption whose removal would change the most, with the least agreement about how. <!-- .element: class="text-lg" style="margin-top: 14px;" -->
