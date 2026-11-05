:::divider id="divider-instruct" title="Evaluating an Instruction-Tuned Model" sub="Now we can ask it questions and grade what it writes"
:::

---

<!-- .slide: id="instruct-two-families" -->

## After SFT, Two Kinds of Question

:::columns cols="2" gap="34px"
**Questions with a right answer**

"What is 17 times 24?"

Scored **automatically**. Cheap and reproducible.
+++
**Questions without one**

"Explain recursion to a beginner."

Scored by **comparison**: a human or model picks the better answer.
:::

Much of modern evaluation work is making answers **checkable**. <!-- .element: class="text-lg" style="margin-top: 10px;" -->

---

<!-- .slide: id="instruct-accuracy" -->

## Why Exact Match Breaks

$$\mathrm{accuracy} = \frac{\text{number correct}}{\text{number evaluated}}$$

Same answer, four strings:

<div class="card-grid cols-4">
<div class="card"><h4><code style="white-space: pre;">"4"</code></h4><p>The key</p></div>
<div class="card"><h4><code style="white-space: pre;">"4."</code></h4><p>Trailing punctuation</p></div>
<div class="card"><h4><code style="white-space: pre;">" 4 "</code></h4><p>Whitespace</p></div>
<div class="card warn"><h4><code style="white-space: pre;">"The answer is 4."</code></h4><p>Needs answer extraction</p></div>
</div>

So every benchmark defines **normalization** (case, punctuation, whitespace) and **answer extraction** (the text after `####`, the last number, `\boxed{}`). <!-- .element: class="text-lg" -->

---

<!-- .slide: id="instruct-f1" -->

## Token-Level F1 Gives Partial Credit

:::columns cols="2" gap="34px"
$$\mathrm{precision} = \frac{TP}{TP+FP}$$

$$\mathrm{recall} = \frac{TP}{TP+FN}$$

$$F_1 = 2 \cdot \frac{\mathrm{precision}\cdot\mathrm{recall}}{\mathrm{precision}+\mathrm{recall}}$$

<div class="metric-box">
<p>Predicted <strong>"it is bluu"</strong>, reference <strong>"it is blue"</strong>: 2 of 3 tokens match each way. Exact match: <strong>0.0</strong>. F1: <strong>0.67</strong>.</p>
</div>

This is SQuAD's scoring rule, and you implement it in the exercise.
+++
<img src="images/precision_recall.png" alt="Diagram of relevant and retrieved elements: true positives, false positives, false negatives, true negatives, with precision and recall as ratios" style="display: block; margin: 0 auto; max-height: 520px; background: #ffffff; border-radius: 8px; padding: 8px;">
:::

---

:::figure img="images/sparck_jones.jpg" name="Karen Spärck Jones" kicker="Information retrieval; inverse document frequency (1972)" alt="Karen Spärck Jones"
A founder of information retrieval, the field that gave NLP **precision and recall**.

She insisted that systems be measured **on real tasks**.
:::

---

<!-- .slide: id="instruct-checkable" -->

## Grade the Answer With a Program

<div class="card-grid cols-4">
<div class="card"><h4>Math</h4><p>Compare the final number numerically: <code>4.0</code> equals <code>4</code>.</p></div>
<div class="card"><h4>JSON</h4><p>Does it parse, with the right fields and types?</p></div>
<div class="card"><h4>Code</h4><p>Run it against hidden unit tests.</p></div>
<div class="card"><h4>Format</h4><p>"Exactly three bullet points" is a few lines of Python.</p></div>
</div>

Module 7's **verifiable reward**, used here to grade instead of train. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="instruct-benchmarks" -->

## The Standard Instruct Benchmarks

<div class="bench-table">
<table>
<thead><tr><th>Benchmark</th><th>Year</th><th>What it tests</th><th>Saturated</th></tr></thead>
<tbody>
<tr><td><strong>GSM8K</strong></td><td>2021</td><td>Grade-school word problems; one number after <code>####</code>.</td><td>2023</td></tr>
<tr><td><strong>MATH</strong></td><td>2021</td><td>Competition math, checked by <strong>symbolic equivalence</strong>.</td><td>2024</td></tr>
<tr><td><strong>HumanEval, MBPP</strong></td><td>2021</td><td>Write a Python function; <strong>run hidden tests</strong>.</td><td>2024</td></tr>
<tr><td><strong>IFEval</strong></td><td>2023</td><td>Instructions with checkable constraints.</td><td>2025</td></tr>
<tr><td><strong>BBH, MMLU-Pro</strong></td><td>2022, 2024</td><td>Harder successors to the originals.</td><td>2025, ~2026</td></tr>
</tbody>
</table>
</div>

Saturated for frontier models, still useful for small ones like the exercise's. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="instruct-examples" -->

## What These Questions Look Like

<div class="card-grid cols-3">
<div class="card"><h4>GSM8K</h4><p>Janet&rsquo;s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?</p><p style="margin-top: 10px;"><strong>Key: #### 18</strong></p></div>
<div class="card"><h4>HumanEval/2</h4><p>Given a positive floating point number, return its decimal part.</p><p style="margin-top: 10px;"><code>truncate_number(3.5)</code> &rarr; <code>0.5</code></p><p style="margin-top: 10px;"><strong>Graded by running hidden asserts</strong> such as <code>candidate(123.456)</code> &asymp; <code>0.456</code>.</p></div>
<div class="card"><h4>IFEval</h4><p>I am planning a trip to Japan, and I would like thee to write an itinerary for my journey in a Shakespearean style. You are not allowed to use any commas in your response.</p><p style="margin-top: 10px;"><strong>Checked: does the reply contain a comma?</strong></p></div>
</div>

---

<!-- .slide: id="instruct-passk" -->

## Sampling Changes What the Number Means

`pass@k`: is **any of k samples** correct?

$$\mathrm{pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}} \quad \text{for } c \text{ correct out of } n \text{ samples}$$

<div class="card-grid cols-3">
<div class="card"><h4>pass@1</h4><p>One try. What a user sees.</p></div>
<div class="card"><h4>pass@10</h4><p>Is the answer <strong>anywhere</strong> in ten tries?</p></div>
<div class="card"><h4>The gap</h4><p>How <strong>sharp</strong> the distribution is.</p></div>
</div>

Introduced with HumanEval, where you really can run ten candidates against the tests. <!-- .element: class="text-lg" -->
