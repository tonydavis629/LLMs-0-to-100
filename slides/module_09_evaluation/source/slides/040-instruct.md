:::divider id="divider-instruct" title="Evaluating an Instruction-Tuned Model" sub="Now we can ask it questions and grade what it writes"
:::

---

<!-- .slide: id="instruct-two-families" -->

## After SFT, Two Kinds of Question

:::columns cols="2" gap="34px"
**Questions with a right answer**

"What is 17 times 24?" "Write a function that returns the median."

Scored **automatically**: compare to a key, parse a number, run the tests. Objective, cheap, reproducible.
+++
**Questions without one**

"Summarize this report." "Explain recursion to a beginner."

Scored by **comparison**: humans or a model judge decide which of two answers is better. Expensive, subjective, unavoidable.
:::

Most modern evaluation effort goes into moving questions from the right column to the left: making answers **checkable**. <!-- .element: class="text-lg" style="margin-top: 10px;" -->

---

<!-- .slide: id="instruct-accuracy" -->

## The Simplest Metric, and Why It Breaks

$$\mathrm{accuracy} = \frac{\text{number correct}}{\text{number evaluated}}$$

Exact match is the most transparent metric and the most brittle. Same answer, four strings:

<div class="card-grid cols-4">
<div class="card"><h4>"4"</h4><p>The key</p></div>
<div class="card"><h4>"4."</h4><p>Trailing punctuation</p></div>
<div class="card"><h4>" 4 "</h4><p>Whitespace</p></div>
<div class="card warn"><h4>"The answer is 4."</h4><p>Needs answer extraction</p></div>
</div>

Every benchmark ships two rules before it compares anything: <!-- .element: class="text-lg" -->

- **Normalization**: lowercase, strip punctuation and articles, collapse whitespace
- **Answer extraction**: text after `####`, the last number, the content of `\boxed{}`

---

<!-- .slide: id="instruct-f1" -->

## Partial Credit: Token-Level F1

When the answer is a phrase, exact match throws away information. Precision, recall, and F1 give partial credit:

$$\mathrm{precision} = \frac{TP}{TP+FP}, \qquad \mathrm{recall} = \frac{TP}{TP+FN}$$

$$F_1 = 2 \cdot \frac{\mathrm{precision}\cdot\mathrm{recall}}{\mathrm{precision}+\mathrm{recall}}$$

<div class="metric-box">
<p>Predicted <strong>"it is bluu"</strong> against reference <strong>"it is blue"</strong>: two of three predicted tokens are shared, two of three reference tokens are recovered. Exact match scores <strong>0.0</strong>. F1 scores <strong>0.67</strong>.</p>
</div>

These come from information retrieval, via SQuAD's scoring script. The exercise implements this exact computation. <!-- .element: class="text-lg" -->

---

:::figure img="images/sparck_jones.jpg" name="Karen Spärck Jones" kicker="Information retrieval; inverse document frequency (1972)" alt="Karen Spärck Jones"
A founder of information retrieval, whose field gave NLP the vocabulary of **precision and recall**. Every F1 score in every model card traces back to this work.

Her often-quoted view that computing is too important to be left to men came with a matching insistence: measure systems **on real tasks**, not on their designers' intuitions.
:::

---

<!-- .slide: id="instruct-checkable" -->

## Better Than String Comparison: Run the Answer

If the answer can be **executed**, do not compare strings:

<div class="card-grid cols-4">
<div class="card"><h4>Parse the number</h4><p>Extract the final numeric answer and compare it as a number, not text.</p></div>
<div class="card"><h4>Validate the schema</h4><p>Does the JSON parse, and does it match the required fields and types?</p></div>
<div class="card"><h4>Run the tests</h4><p>Execute the generated function against hidden unit tests.</p></div>
<div class="card"><h4>Check the constraint</h4><p>Exactly three bullet points? No use of the letter "e"? A parser can tell.</p></div>
</div>

Module 7's **verifiable reward** (a deterministic function that decides correctness), reused here for **scoring** instead of training. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="instruct-benchmarks" -->

## The Standard Instruct Benchmarks

<div class="bench-table">
<table>
<thead><tr><th>Benchmark</th><th>Year</th><th>What it tests, and how it is checked</th></tr></thead>
<tbody>
<tr><td><strong>GSM8K</strong></td><td>2021</td><td>Grade-school math word problems. Single numeric answer after a <code>####</code> marker. The canonical checkable-answer benchmark.</td></tr>
<tr><td><strong>MATH</strong></td><td>2021</td><td>Competition mathematics, much harder. Answers checked by <strong>symbolic equivalence</strong>, not string match.</td></tr>
<tr><td><strong>HumanEval, MBPP</strong></td><td>2021</td><td>Write a Python function from a docstring. Scored by <strong>running hidden tests</strong>.</td></tr>
<tr><td><strong>IFEval</strong></td><td>2023</td><td>Instructions with programmatically checkable constraints. Measures <strong>instruction-following itself</strong>, separately from knowledge.</td></tr>
<tr><td><strong>BBH, MMLU-Pro</strong></td><td>2022, 2024</td><td>Harder successors, built once the originals stopped separating models.</td></tr>
</tbody>
</table>
</div>

---

<!-- .slide: id="instruct-passk" -->

## Sampling Changes What the Number Means

**Greedy decoding** scores the single best answer. `pass@k` asks whether **any of k samples** is correct, the right metric when you can test candidates and keep the winner:

$$\mathrm{pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}} \quad \text{for } c \text{ correct out of } n \text{ samples}$$

<div class="card-grid cols-3">
<div class="card"><h4>pass@1</h4><p>Ordinary accuracy of one sample. What a user experiences.</p></div>
<div class="card"><h4>pass@10</h4><p>Is the answer <strong>anywhere</strong> in the distribution across ten tries?</p></div>
<div class="card"><h4>The gap</h4><p>Measures how <strong>sharp</strong> the distribution is, which is what reasoning-model evaluation turns on.</p></div>
</div>

Introduced with HumanEval: for code, you really can generate ten candidates and run the tests. <!-- .element: class="text-lg" -->
