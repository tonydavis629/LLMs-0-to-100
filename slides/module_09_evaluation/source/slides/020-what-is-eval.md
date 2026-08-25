:::divider id="divider-basics" title="Evaluations, Benchmarks, Leaderboards" sub="Three words, three different things"
:::

---

<!-- .slide: id="three-words" -->

## Three Words, Three Different Things

<div class="card-grid cols-3">
<div class="card"><h4>Evaluation</h4><p>A <strong>dataset plus a scoring rule</strong>. Run the model on the cases, score the outputs, report a number.</p></div>
<div class="card"><h4>Benchmark</h4><p>An evaluation that is <strong>published and reused</strong>, so different labs can compare models on the same cases under the same rules.</p></div>
<div class="card"><h4>Leaderboard</h4><p>A <strong>public table</strong> of benchmark results. A summary of evaluations, not an evaluation itself.</p></div>
</div>

Every announced number is a benchmark result: somebody's dataset plus somebody's scoring rule. Both are choices. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="two-requirements" -->

## Two Things Every Evaluation Needs

:::columns cols="2" gap="34px"
**Held-out data**

- Test cases in training data: the score measures **memorization**, not ability
- Same idea as the Module 2 train/test split
- The web-scale version is **contamination**, later in this module
+++
**An agreed scoring rule**

- "Accuracy" means nothing until you define **answer extraction** and **what counts as a match**
- Is "The answer is 4." the same as "4"? Somebody decides and publishes the code
:::

Same benchmark, same weights, different numbers: usually the scoring rule, not fraud. <!-- .element: class="text-lg" style="margin-top: 10px;" -->

---

<!-- .slide: id="three-ways-to-score" -->

## Three Ways to Score an Output

<div class="card-grid cols-3">
<div class="card"><h4>Automatic and exact</h4><p>Compare against a key, run code against tests, check a number.</p><p style="margin-top: 8px;"><strong>Cheap and objective.</strong> Only works when there is a right answer.</p></div>
<div class="card"><h4>Human judgment</h4><p>People read the outputs and rate or rank them.</p><p style="margin-top: 8px;"><strong>Expensive and slow.</strong> Still the ground truth for open-ended work.</p></div>
<div class="card"><h4>Model judgment</h4><p>A strong model grades the outputs against a rubric.</p><p style="margin-top: 8px;"><strong>Cheap and scalable.</strong> A proxy that has to be checked against humans.</p></div>
</div>

**Which one you can use depends on the model you are measuring.** That is the rest of the module. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="two-shapes" -->

## Two Scoring Shapes, and How to Tell Them Apart

<div class="compare-table">
<table>
<thead><tr><th>Shape</th><th>What happens</th><th>Works on</th><th>Examples</th></tr></thead>
<tbody>
<tr><td><strong>Multiple choice, scored by likelihood</strong></td><td>Paste each choice onto the question; read the probability the model assigns to it. <strong>Most probable choice wins.</strong> Nothing is generated: one forward pass per choice.</td><td>Any model, including a base model that cannot follow instructions</td><td>MMLU, HellaSwag, ARC, WinoGrande</td></tr>
<tr><td><strong>Free generation, scored by a checker</strong></td><td>The model writes an answer; <strong>extract and check it</strong> against a key, a parser, or a test suite.</td><td>Instruction-tuned models that answer prompts</td><td>GSM8K, HumanEval, IFEval, most instruct benchmarks</td></tr>
</tbody>
</table>
</div>

The exercise implements both. They can disagree completely about the same two models. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="benchmark-history" -->

## Where This Whole Practice Came From

**ImageNet (2009)** showed that a shared test set with a public leaderboard can organize a field. AlexNet (Module 1) was an ImageNet result. NLP copied the pattern:

<div class="card-grid cols-3">
<div class="card"><h4>SQuAD (2016)</h4><p>100,000 reading-comprehension questions with a shared scoring script. Brought <strong>exact match and token F1</strong> into NLP.</p></div>
<div class="card"><h4>GLUE (2018)</h4><p>Nine tasks bundled into <strong>one score</strong>. Human performance was passed in about a year.</p></div>
<div class="card"><h4>SuperGLUE (2019)</h4><p>Built because GLUE saturated. Passed in about <strong>eighteen months</strong>.</p></div>
</div>

Every benchmark in this module descends from this: **a shared test set, a shared scoring rule, a public table.** <!-- .element: class="text-lg" -->

---

:::figure img="images/feifei_li.jpg" name="Fei-Fei Li" kicker="ImageNet and the ILSVRC competition (2009-2017)" alt="Fei-Fei Li"
Built the dataset and annual competition that established the shared-benchmark-plus-leaderboard model. ImageNet was not an algorithm; it was a **measurement instrument**.

Without the benchmark there is no scoreboard, and the 2012 AlexNet jump is just one lab's claim.
:::

---

:::figure img="images/bowman_rajpurkar.jpg" name="Samuel Bowman &amp; Pranav Rajpurkar" kicker="GLUE and SuperGLUE (2018, 2019); SQuAD (2016)" alt="Samuel Bowman and Pranav Rajpurkar"
Bowman, with **Alex Wang** and collaborators, bundled nine tasks into **one headline number** behind a public leaderboard with a held-out test server, so nobody could tune on the answers. Later suites copy that template.

Rajpurkar and collaborators published SQuAD. Its **official scoring script** (normalize, then exact match and token F1) became the default grader for short free-form answers. You write that script in the exercise.
:::
