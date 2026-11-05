:::divider id="divider-basics" title="Evaluations, Benchmarks, Leaderboards" sub="What each term means"
:::

---

<!-- .slide: id="three-words" -->

## Three Terms People Use Interchangeably

<div class="card-grid cols-3">
<div class="card"><h4>Evaluation</h4><p>A <strong>dataset and a scoring rule</strong>. Run the model, score the outputs, report a number.</p></div>
<div class="card"><h4>Benchmark</h4><p>A <strong>published</strong> evaluation, so every lab runs the same cases under the same rules.</p></div>
<div class="card"><h4>Leaderboard</h4><p>A <strong>public table</strong> of benchmark results.</p></div>
</div>

An announced score depends on which test cases someone picked and how they wrote the grader. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="two-requirements" -->

## Two Things Every Evaluation Needs

:::columns cols="2" gap="34px"
**Held-out data**

- Test cases seen in training measure **memorization**
- Same idea as Module 2's train/test split
+++
**An agreed scoring rule**

- Is "The answer is 4." the same as "4"?
- Someone decides, and publishes the code
:::

When two labs report different scores for the same model, the scoring rule is usually why. <!-- .element: class="text-lg" style="margin-top: 10px;" -->

---

<!-- .slide: id="three-ways-to-score" -->

## Three Ways to Score an Output

<div class="card-grid cols-3">
<div class="card"><h4>Automatic</h4><p>Compare to a key, run tests, check a number.</p><p style="margin-top: 8px;"><strong>Cheap.</strong> Needs a right answer.</p></div>
<div class="card"><h4>Human judgment</h4><p>People rate or rank the outputs.</p><p style="margin-top: 8px;"><strong>Slow and expensive.</strong> The ground truth for open-ended work.</p></div>
<div class="card"><h4>Model judgment</h4><p>A strong model grades against a rubric.</p><p style="margin-top: 8px;"><strong>Cheap.</strong> Must be checked against humans.</p></div>
</div>

Which one you can use depends on the model you are measuring. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="two-shapes" -->

## Multiple Choice vs. Free Generation

<div class="compare-table">
<table>
<thead><tr><th>Shape</th><th>What happens</th><th>Works on</th><th>Examples</th></tr></thead>
<tbody>
<tr><td><strong>Multiple choice, by likelihood</strong></td><td>Score each option by the probability the model gives it. <strong>Most probable wins.</strong> Nothing is generated.</td><td>Any model, even a base model</td><td>MMLU, HellaSwag, ARC</td></tr>
<tr><td><strong>Free generation, by a checker</strong></td><td>The model writes an answer; <strong>extract and check it</strong>.</td><td>Instruction-tuned models</td><td>GSM8K, HumanEval, IFEval</td></tr>
</tbody>
</table>
</div>

The exercise implements both, and they disagree about the same two models. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="benchmark-history" -->

## Where This Whole Practice Came From

**ImageNet (2009)** showed that a shared test set and a public leaderboard can organize a field. NLP copied it:

<div class="card-grid cols-3">
<div class="card"><h4>SQuAD (2016)</h4><p>Reading comprehension with a shared scoring script: <strong>exact match and token F1</strong>.</p></div>
<div class="card"><h4>GLUE (2018)</h4><p>Nine tasks, <strong>one score</strong>. Humans passed in about a year.</p></div>
<div class="card"><h4>SuperGLUE (2019)</h4><p>Built because GLUE saturated. Passed in about <strong>eighteen months</strong>.</p></div>
</div>

---

:::figure img="images/feifei_li.jpg" name="Fei-Fei Li" kicker="ImageNet and the ILSVRC competition (2009-2017)" alt="Fei-Fei Li"
Built ImageNet and its yearly competition: the first **shared benchmark with a leaderboard** at scale.

Without it, the 2012 AlexNet jump would have been one lab's claim.
:::

---

:::figure img="images/bowman_rajpurkar.jpg" name="Samuel Bowman &amp; Pranav Rajpurkar" kicker="GLUE and SuperGLUE (2018, 2019); SQuAD (2016)" alt="Samuel Bowman and Pranav Rajpurkar"
Bowman and collaborators built GLUE: nine tasks, **one number**, and a hidden test set.

Rajpurkar and collaborators built SQuAD. Its **scoring script** is the one you write in the exercise.
:::
