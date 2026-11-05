:::divider id="divider-base" title="Evaluating a Pretrained Model" sub="What can you measure when the model will not answer you?"
:::

---

<!-- .slide: id="base-problem" -->

## A Base Model Does Not Follow Instructions

Ask a base model a question and it may just continue with more questions. Two things can be measured anyway:

<div class="card-grid cols-2">
<div class="card"><h4>How well it predicts text</h4><p>Loss and perplexity on held-out text. <strong>No labels needed.</strong></p></div>
<div class="card"><h4>What it knows</h4><p>Multiple choice scored by <strong>likelihood</strong>. The model never writes anything.</p></div>
</div>

---

<!-- .slide: id="base-perplexity" -->

## Perplexity From Module 5

$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t \mid x_{<t}), \qquad \mathrm{PPL} = \exp(\mathcal{L})$$

<div class="metric-box">
<p>The average number of tokens the model is choosing between. 1 = never surprised. 50,000 = guessing uniformly.</p>
</div>

Lower is better. It works on any checkpoint, at any point in training. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="base-bpb" -->

## Perplexity Does Not Compare Across Tokenizers

Perplexity is **per token**, and a bigger vocabulary cuts the same text into fewer tokens. Both models below spend the same 300 nats on the same 1,000 bytes:

<div class="compare-table">
<table>
<thead><tr><th>Model</th><th>Vocabulary</th><th>Tokens used</th><th>Total loss</th><th>Perplexity</th><th>BPB</th></tr></thead>
<tbody>
<tr><td><strong>X</strong></td><td class="num">32,000</td><td class="num">250</td><td class="num">300 nats</td><td class="num">3.32</td><td class="num">0.433</td></tr>
<tr><td><strong>Y</strong></td><td class="num">128,000</td><td class="num">200</td><td class="num">300 nats</td><td class="num">4.48</td><td class="num">0.433</td></tr>
</tbody>
</table>
</div>

Y looks **35% worse** by perplexity. Bits per byte shows they are equal. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="base-bpb-units" -->

## What Bits per Byte Measures

$$\mathrm{BPB} = \frac{\mathcal{L}_{\text{total}}}{\ln(2) \cdot \text{number of bytes}}$$

:::columns cols="2" gap="34px"
**Bytes: how much text there is**

- A property of the file
- Fixed, whatever the tokenizer
+++
**Bits: what the model spent**

- Total loss, in bits
- Moves with the weights
:::

Like **dollars per mile**: a cost divided by a fixed distance. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="base-bpb-compression" -->

## Which Makes It a Compression Rate

A byte is **8 bits**. BPB is how many bits the model needs to store one byte of text.

<div class="card-grid cols-4">
<div class="card"><h4>8.0</h4><p>No model at all.</p></div>
<div class="card"><h4>~4.1</h4><p>Letter frequencies only.</p></div>
<div class="card"><h4>~1.0</h4><p>Shannon's 1951 estimate for humans.</p></div>
<div class="card"><h4>~0.7</h4><p>A good modern LLM.</p></div>
</div>

Module 1 claimed **prediction and compression are the same operation**. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="base-bpb-rule" -->

## When Each One Is Allowed

<div class="card-grid cols-2">
<div class="card"><h4>Perplexity is fine</h4><p>Same tokenizer: comparing checkpoints, watching your own loss curve.</p></div>
<div class="card warn"><h4>Perplexity is meaningless</h4><p>Different tokenizers. Report bits per byte.</p></div>
</div>

---

<!-- .slide: id="base-benchmarks" -->

## Knowledge Benchmarks for Base Models

All scored by **likelihood**. **Saturated** means frontier models reached the human or expert ceiling; small models still spread out on them.

<div class="bench-table dense">
<table>
<thead><tr><th>Benchmark</th><th>Year</th><th>What it asks</th><th>Saturated</th></tr></thead>
<tbody>
<tr><td><strong>MMLU</strong></td><td>2020</td><td>Multiple choice across 57 subjects.</td><td>2024</td></tr>
<tr><td><strong>HellaSwag</strong></td><td>2019</td><td>Pick the sensible next sentence.</td><td>2023</td></tr>
<tr><td><strong>ARC</strong></td><td>2018</td><td>Grade-school science.</td><td>2023</td></tr>
<tr><td><strong>WinoGrande, PIQA</strong></td><td>2019</td><td>Pronoun resolution, physical commonsense.</td><td>~2023</td></tr>
<tr><td><strong>TriviaQA, Natural Questions</strong></td><td>2017, 2019</td><td>Short factual answers.</td><td>Legacy</td></tr>
<tr><td><strong>GPQA</strong></td><td>2023</td><td>Graduate science, hard <em>even with web search</em>.</td><td>2025</td></tr>
</tbody>
</table>
</div>

---

<!-- .slide: id="base-examples" -->

## What These Questions Look Like

<div class="card-grid cols-2">
<div class="card"><h4>MMLU &middot; high school physics</h4><p>The plates of a capacitor are charged to a potential difference of 5 V. If the capacitance is 2 mF, what is the charge on the positive plate?</p><p style="margin-top: 10px;">A. 0.005 C &nbsp; <strong>B. 0.01 C</strong> &nbsp; C. 0.02 C &nbsp; D. 0.5 C</p></div>
<div class="card"><h4>HellaSwag</h4><p>A lady walks to a barbell. She bends down and grabs the pole. the lady...</p><p style="margin-top: 10px;">A. swings and lands in her arms.<br>B. pulls the barbell forward.<br>C. pulls a rope attached to the barbell.<br><strong>D. stands and lifts the weight over her head.</strong></p></div>
</div>

The model writes nothing. Each option is scored by its probability, and the most probable one is the model's answer. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="base-fewshot" -->

## Few-Shot Prompting Is Part of the Protocol

Base models get `k` solved examples before the question, so the prompt teaches the format (Module 5).

<div class="card-grid cols-3">
<div class="card"><h4>0-shot</h4><p>Question only. A base model may not answer at all.</p></div>
<div class="card"><h4>5-shot</h4><p>The MMLU convention.</p></div>
<div class="card"><h4>25-shot</h4><p>Used for ARC on some leaderboards.</p></div>
</div>

"MMLU 68" is incomplete. "MMLU 68, 5-shot, likelihood-scored" can be checked. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="base-daily-use" -->

## What Pretraining Evaluation Is Actually For

Nobody ships a base model. Its evaluations drive **training decisions**:

<div class="card-grid cols-3">
<div class="card"><h4>Checkpoint selection</h4><p>Is the run still improving?</p></div>
<div class="card"><h4>Data mixtures</h4><p>Does more code or cleaner web text help?</p></div>
<div class="card"><h4>Scaling checks</h4><p>Is the loss where the scaling law predicted?</p></div>
</div>

---

:::figure img="images/hendrycks.jpg" name="Dan Hendrycks" kicker="MMLU (2020) and MATH (2021)" alt="Dan Hendrycks"
Built **MMLU** and **MATH**, deliberately harder than the models of the time.

Both defined frontier comparison for years. Both are now saturated.
:::
