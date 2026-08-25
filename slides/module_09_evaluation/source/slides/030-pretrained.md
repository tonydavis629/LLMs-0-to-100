:::divider id="divider-base" title="Evaluating a Pretrained Model" sub="What can you measure when the model will not answer you?"
:::

---

<!-- .slide: id="base-problem" -->

## A Base Model Does Not Follow Instructions

Ask a base model "what is the capital of France?" and it may continue with more questions (why Module 6 exists). Most abilities **cannot be asked for directly**. Two things can be measured anyway:

<div class="card-grid cols-2">
<div class="card"><h4>How well it predicts text</h4><p>Loss and perplexity on held-out text. <strong>No labels, no instructions, no generation.</strong></p></div>
<div class="card"><h4>What it knows</h4><p>Multiple-choice benchmarks scored by <strong>likelihood</strong>, which never require the model to write anything.</p></div>
</div>

---

<!-- .slide: id="base-perplexity" -->

## Perplexity: Module 5's Metric, Restated

$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t \mid x_{<t}), \qquad \mathrm{PPL} = \exp(\mathcal{L})$$

<div class="metric-box">
<p>Perplexity is the model's average <strong>branching factor</strong>: how many equally likely tokens it chooses among at each position. 1 = never surprised. 50,000 = guessing uniformly over the vocabulary.</p>
</div>

Lower is better. It needs no labels, only held-out text, so it works on any checkpoint at any point in training. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="base-bpb" -->

## Perplexity Does Not Compare Across Tokenizers

Perplexity is **per token**. A bigger vocabulary chops the same text into fewer, larger pieces (Module 4), so two **equally good** models can report very different perplexities.

Both models below spend 300 nats on the same 1,000-byte passage, so they model it equally well:

<div class="compare-table">
<table>
<thead><tr><th>Model</th><th>Vocabulary</th><th>Tokens used</th><th>Total loss</th><th>Perplexity</th><th>BPB</th></tr></thead>
<tbody>
<tr><td><strong>X</strong></td><td class="num">32,000</td><td class="num">250</td><td class="num">300 nats</td><td class="num">3.32</td><td class="num">0.433</td></tr>
<tr><td><strong>Y</strong></td><td class="num">128,000</td><td class="num">200</td><td class="num">300 nats</td><td class="num">4.48</td><td class="num">0.433</td></tr>
</tbody>
</table>
</div>

Model Y looks **35% worse** and is not worse at all. It answered fewer, harder questions. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="base-bpb-units" -->

## Bits per Byte: What the Two Units Are Doing

$$\mathrm{BPB} = \frac{\mathcal{L}_{\text{total}}}{\ln(2) \cdot \text{number of bytes}}$$

:::columns cols="2" gap="34px"
**Bytes: how much text there is**

- UTF-8 size of the held-out passage
- A property of the *file*
- Does not move when you swap tokenizers
+++
**Bits: what the model spent**

- Total surprisal, converted from nats ($\div \ln 2$)
- Moves with the weights
- Barely moves with the tokenizer
:::

Like **dollars per mile**: a cost, divided by a fixed distance to cover. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="base-bpb-compression" -->

## Which Makes It a Compression Rate

A byte holds **8 bits**. Bits per byte = how many bits the model needs to store one byte of English. 8 means no prediction at all.

<div class="card-grid cols-4">
<div class="card"><h4>8.0</h4><p>No model. Every byte equally likely; store the file as it came.</p></div>
<div class="card"><h4>~4.1</h4><p>A character histogram. Knows "e" is common and most byte values never appear.</p></div>
<div class="card"><h4>~1.0</h4><p>Shannon's 1951 estimate for English, measured by having people guess the next letter.</p></div>
<div class="card"><h4>~0.7</h4><p>A competent modern LLM on general text. Better than Shannon's humans.</p></div>
</div>

Multiply by the byte count: the compressed file size. Module 1 claimed **prediction and compression are the same operation**. This is that claim as a number. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="base-bpb-rule" -->

## When Each One Is Allowed

<div class="card-grid cols-2">
<div class="card"><h4>Perplexity is fine</h4><p>Same model family, same tokenizer, different checkpoints. Watching your own loss curve. This covers most of the day-to-day cases.</p></div>
<div class="card warn"><h4>Perplexity is meaningless</h4><p>Two labs' models with different tokenizers. Report bits per byte, or report nothing.</p></div>
</div>

---

<!-- .slide: id="base-perplexity-limits" -->

## Low Perplexity Is Not Usefulness

Perplexity rewards **fluent continuation**, not correct answers. Two models with nearly identical perplexity can differ by tens of points on tasks.

The exercise measures this directly: the RL model has **lower** perplexity than the instruct model, and is **far worse** at three of the four tasks. This is why nobody ships a model on a perplexity number. <!-- .element: class="text-lg" style="margin-top: 10px;" -->

---

<!-- .slide: id="base-benchmarks" -->

## Knowledge Benchmarks for Base Models

All scored by **likelihood**, so they work without instruction-following.

<div class="bench-table dense">
<table>
<thead><tr><th>Benchmark</th><th>Year</th><th>What it asks</th></tr></thead>
<tbody>
<tr><td><strong>MMLU</strong></td><td>2020</td><td>57 subjects of multiple choice, elementary to professional. For years, <em>the</em> headline number for a base model.</td></tr>
<tr><td><strong>HellaSwag</strong></td><td>2019</td><td>Which sentence plausibly continues this situation? Commonsense, adversarially filtered.</td></tr>
<tr><td><strong>ARC</strong></td><td>2018</td><td>Grade-school science questions, split into Easy and Challenge sets.</td></tr>
<tr><td><strong>WinoGrande, PIQA</strong></td><td>2019</td><td>Pronoun resolution and physical commonsense. The standard set in open-model reports.</td></tr>
<tr><td><strong>TriviaQA, Natural Questions</strong></td><td>2017, 2019</td><td>Factual recall with short free-form answers, scored by exact match and F1.</td></tr>
<tr><td><strong>GPQA</strong></td><td>2023</td><td>Graduate-level science, written to be hard <em>even with web search</em>. Built after MMLU began saturating.</td></tr>
</tbody>
</table>
</div>

---

<!-- .slide: id="base-fewshot" -->

## Few-Shot Prompting Is Part of the Protocol

Base models get `k` solved examples in the context, so the format comes from the prompt itself. This is GPT-3's in-context learning (Module 5).

<div class="card-grid cols-3">
<div class="card"><h4>0-shot</h4><p>Question only. Hardest for a base model: it may not produce an answer-shaped output at all.</p></div>
<div class="card"><h4>5-shot</h4><p>The MMLU convention. Five worked examples, then the question.</p></div>
<div class="card"><h4>25-shot</h4><p>Used for ARC on some leaderboards. Scores move with `k`.</p></div>
</div>

**A number without its shot count is not reproducible.** "MMLU 68" is incomplete. "MMLU 68, 5-shot, likelihood-scored" is checkable. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="base-daily-use" -->

## What Pretraining Evaluation Is Actually For

Nobody ships a base model. Its evaluations drive **training decisions**:

<div class="card-grid cols-3">
<div class="card"><h4>Checkpoint selection</h4><p>Compare checkpoints of the same run as training proceeds. Is it still improving?</p></div>
<div class="card"><h4>Data mixtures</h4><p>Does more code, or more filtered web text, move the numbers? A/B the corpus.</p></div>
<div class="card"><h4>Scaling checks</h4><p>Confirm a run is on the loss curve the scaling law predicted (Module 5).</p></div>
</div>

---

:::figure img="images/hendrycks.jpg" name="Dan Hendrycks" kicker="MMLU (2020) and MATH (2021)" alt="Dan Hendrycks"
Created the benchmarks that defined frontier comparison for years: **MMLU** for broad knowledge, **MATH** for competition mathematics. Both were built deliberately harder than models of the time could handle, which kept them useful longer.

Both are now largely saturated, which happens to every benchmark, faster each time.
:::
