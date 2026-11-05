:::divider id="divider-reasoning" title="Evaluating a Reasoning or RL-Trained Model" sub="The reward went up. So what?"
:::

---

<!-- .slide: id="reasoning-reward-curve" -->

## A Reward Curve Is Not an Evaluation

A rising reward curve (Module 7) shows **the optimizer works**. It cannot show the model got better.

<div class="card-grid cols-2">
<div class="card"><h4>What it measures</h4><p>Progress on the training prompts, under the training reward.</p></div>
<div class="card warn"><h4>What it misses</h4><p>Generalization, other abilities, and reward hacking.</p></div>
</div>

Evaluate an RL model on the **same held-out benchmarks as before**, plus checks for RL's failure modes. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="reasoning-benchmarks" -->

## Hard Problems With Checkable Answers

Module 7's verifiable rewards, used as a **test set**.

<div class="bench-table dense">
<table>
<thead><tr><th>Benchmark</th><th>What it asks</th><th>Status</th></tr></thead>
<tbody>
<tr><td><strong>AIME</strong></td><td>Hard high-school contest math; integer answers.</td><td>Saturated 2025</td></tr>
<tr><td><strong>GPQA Diamond</strong></td><td>Graduate science that resists web search.</td><td>Passed experts 2024, saturated 2025</td></tr>
<tr><td><strong>Competitive programming</strong></td><td>Codeforces-style problems, run against the judge's tests.</td><td>ICPC gold 2025; top human rating still ahead</td></tr>
<tr><td><strong>SWE-bench Verified</strong></td><td>Fix a real GitHub issue so the tests pass.</td><td>Saturated 2025, dropped by OpenAI 2026</td></tr>
<tr><td><strong>ARC-AGI</strong></td><td>Abstract visual puzzles that resist memorization.</td><td>v1 saturated 2024, v2 2026</td></tr>
<tr><td><strong>FrontierMath</strong></td><td>Unpublished research-level math, run by Epoch AI.</td><td>Hardest tier hit 100% in 2026</td></tr>
</tbody>
</table>
</div>

---

<!-- .slide: id="reasoning-example" -->

## What an AIME Problem Looks Like

<div class="metric-box">
<p>Every morning Aya goes for a 9-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of <em>s</em> kilometers per hour, the walk takes her 4 hours, including <em>t</em> minutes spent in the coffee shop. When she walks <em>s</em>+2 kilometers per hour, the walk takes her 2 hours and 24 minutes, including <em>t</em> minutes spent in the coffee shop. Suppose Aya walks at <em>s</em>+&frac12; kilometers per hour. Find the number of minutes the walk takes her, including the <em>t</em> minutes spent in the coffee shop.</p>
</div>

**Answer: 204.** Every AIME answer is an integer from 0 to 999, so grading is exact match. (AIME 2024 I, Problem 1) <!-- .element: class="text-lg" -->

---

<!-- .slide: id="reasoning-open" -->

## Reasoning Benchmarks That Are Still Open

<div class="bench-table">
<table>
<thead><tr><th>Benchmark</th><th>Year</th><th>What it measures</th><th>Best score, late 2026</th></tr></thead>
<tbody>
<tr><td><strong>Humanity's Last Exam</strong></td><td>2025</td><td>2,500 expert-written questions meant to stump frontier models.</td><td>~61%</td></tr>
<tr><td><strong>FrontierMath Erd&odblac;s</strong></td><td>2026</td><td>68 <strong>unsolved</strong> Erd&odblac;s problems. Answers must be proofs in Lean, checked by machine.</td><td>2 of 68 (3%)</td></tr>
<tr><td><strong>ARC-AGI-3</strong></td><td>2026</td><td>Games where the model must discover the rules.</td><td>Under 1% at launch, 63% by October</td></tr>
<tr><td><strong>SimpleQA Verified</strong></td><td>2025</td><td>Short factual questions, <strong>no tools</strong>.</td><td>~78%</td></tr>
</tbody>
</table>
</div>

These scores change every few months. Check the official leaderboard before quoting one. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="reasoning-reporting" -->

## Two Reporting Rules That Matter More Here

<div class="card-grid cols-2">
<div class="card"><h4>1. How samples were combined</h4><p><strong>pass@1</strong>, <strong>majority vote</strong>, or <strong>best-of-n</strong>? These can differ by tens of points.</p></div>
<div class="card"><h4>2. The reasoning budget</h4><p>The same model at 2,000 and 32,000 thinking tokens is a different system.</p></div>
</div>

Without both, the score cannot be reproduced. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="reasoning-passk" -->

## What pass@k Means for Evaluation

<div class="card-grid cols-3">
<div class="card"><h4>pass@1</h4><p>How often the <strong>first try</strong> is right.</p></div>
<div class="card"><h4>pass@k, large k</h4><p>Can the model solve it <strong>at all</strong>, given many tries?</p></div>
<div class="card warn"><h4>What RL does</h4><p>Raises pass@1 but barely moves pass@k. It makes the model pick answers it could already find; it does not solve new problems.</p></div>
</div>

Report both. pass@1 alone makes RL look like it added ability (Module 7). <!-- .element: class="text-lg" -->

---

<!-- .slide: id="reasoning-breakage" -->

## What RL Breaks That the Target Benchmark Will Not Show

<div class="card-grid cols-2">
<div class="card warn"><h4>Other tasks regress</h4><p>The <strong>alignment tax</strong> (Module 7). Keep running the old evaluations.</p></div>
<div class="card warn"><h4>Reward hacks persist</h4><p>Tricks that scored during training are still there at test time.</p></div>
</div>

In the exercise, RL on the reverse task gains **+17 points** there and loses **50 to 75 points** on the other three tasks. <!-- .element: class="text-lg" style="margin-top: 8px;" -->

---

<!-- .slide: id="reasoning-refusal" -->

## Behavior Evaluations Always Come in Pairs

<div class="card-grid cols-2">
<div class="card"><h4>Refusal rate</h4><p>Does it decline harmful requests?</p></div>
<div class="card"><h4>False-refusal rate</h4><p>Does it also decline <strong>harmless</strong> ones that sound alarming? (e.g. <strong>XSTest</strong>)</p></div>
</div>

Refuse everything and you ace the first. Refuse nothing and you ace the second. **Only the pair means anything.** <!-- .element: class="text-lg" -->

---

<!-- .slide: id="reasoning-goodhart" -->

## Goodhart's Law Is the Theme of This Section

> When a measure becomes a target, it ceases to be a good measure.
>
> Marilyn Strathern (1997), restating Charles Goodhart

<div class="card-grid cols-2">
<div class="card"><h4>In training (Module 7)</h4><p><strong>Reward hacking</strong>: the policy scores in ways nobody intended.</p></div>
<div class="card"><h4>In the industry</h4><p><strong>Benchmark chasing</strong>: the published number stops predicting usefulness.</p></div>
</div>
