:::divider id="divider-reasoning" title="Evaluating a Reasoning or RL-Trained Model" sub="The reward went up. So what?"
:::

---

<!-- .slide: id="reasoning-reward-curve" -->

## A Reward Curve Is Not an Evaluation

Module 7's climbing reward curve says **the optimizer works**. The reward is exactly what was optimized, so the curve says nothing about whether the model got better.

<div class="card-grid cols-2">
<div class="card"><h4>What the reward curve measures</h4><p>Training progress on the training prompts, under the training reward. A <strong>diagnostic</strong>.</p></div>
<div class="card warn"><h4>What it cannot measure</h4><p>Whether the ability generalizes, whether other abilities survived, or whether the model found a way to score without solving anything.</p></div>
</div>

Evaluate an RL model with the **same held-out benchmarks as before**, plus new ones for RL's failure modes. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="reasoning-benchmarks" -->

## Hard Problems With Checkable Answers

This is the RLVR setup from Module 7 used as a **test set** rather than a training signal.

<div class="bench-table">
<table>
<thead><tr><th>Benchmark</th><th>What it asks</th></tr></thead>
<tbody>
<tr><td><strong>AIME</strong> and competition math</td><td>Short integer answers, hard enough to be unsaturated. The standard headline number in reasoning-model announcements.</td></tr>
<tr><td><strong>GPQA Diamond</strong></td><td>The hardest slice of GPQA: graduate science questions that resist web search.</td></tr>
<tr><td><strong>Competitive programming</strong></td><td>Codeforces-style problems, scored by running the judge's tests. Often reported as an Elo rating.</td></tr>
<tr><td><strong>SWE-bench</strong></td><td>Resolve a real GitHub issue so the project's test suite passes. Covered under agent evaluation.</td></tr>
<tr><td><strong>ARC-AGI</strong></td><td>Abstract visual puzzles designed to resist memorization. The most-cited "is it really reasoning" benchmark.</td></tr>
</tbody>
</table>
</div>

---

<!-- .slide: id="reasoning-reporting" -->

## Two Reporting Rules That Matter More Here

Reasoning models are **stochastic and expensive**. Without these two numbers, the score is meaningless:

<div class="card-grid cols-2">
<div class="card"><h4>1. How the samples were combined</h4><p>Is this <strong>pass@1</strong>, <strong>majority vote</strong> over n samples (self-consistency), or <strong>best-of-n with a verifier</strong>?</p><p style="margin-top: 8px;">These can differ by tens of points on the same model and the same benchmark.</p></div>
<div class="card"><h4>2. The reasoning budget</h4><p>Test-time compute is a dial (Module 7). The same model at 2,000 thinking tokens and at 32,000 is not the same system.</p><p style="margin-top: 8px;">A score without a token budget is <strong>not reproducible</strong>.</p></div>
</div>

---

<!-- .slide: id="reasoning-passk" -->

## The pass@k Lesson, Restated as an Evaluation Principle

Module 7: RL **sharpens** the sampling distribution rather than expanding ability. As measurement, that is a claim about the pass@1 versus pass@k gap:

<div class="card-grid cols-3">
<div class="card"><h4>pass@1 rises sharply</h4><p>The model now puts its probability mass on the answer it could already sometimes find.</p></div>
<div class="card"><h4>pass@k at large k stays flat</h4><p>The set of problems it can solve <em>at all</em> did not grow.</p></div>
<div class="card warn"><h4>The consequence</h4><p>Measuring only pass@1 <strong>systematically overstates</strong> what RL added.</p></div>
</div>

In the exercise: the RL model's QA pass@1 falls to 70% while pass@5 stays at 100%. The right answer is still in there, just no longer on top. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="reasoning-breakage" -->

## What RL Breaks That the Target Benchmark Will Not Show

<div class="card-grid cols-2">
<div class="card warn"><h4>Regressions on other tasks</h4><p>Training hard on one objective degrades unrelated behavior: the <strong>alignment tax</strong> from Module 7. The fix is boring. Keep running the old evaluations.</p></div>
<div class="card warn"><h4>Reward hacking that survives to test time</h4><p>Padding, hedging, and formatting tricks that scored well during training do not disappear when training stops.</p></div>
</div>

> The exercise makes this concrete: GRPO on the reverse task alone gains **+17 points** on reverse and loses **50 to 75 points** on the three tasks nobody was watching.

The KL penalty does not save you: it constrains the policy on training prompts and does nothing on prompts it never sees. <!-- .element: class="text-lg" style="margin-top: 8px;" -->

---

<!-- .slide: id="reasoning-refusal" -->

## Behavior Evaluations Always Come in Pairs

<div class="card-grid cols-2">
<div class="card"><h4>Refusal rate</h4><p>Does the model decline genuinely harmful requests? Measured on harmful-instruction suites and jailbreak-robustness tests.</p></div>
<div class="card"><h4>False-refusal rate</h4><p>Does it also decline <strong>benign</strong> requests that merely sound alarming? Measured on over-refusal suites such as <strong>XSTest</strong>.</p></div>
</div>

Either number alone is trivially gamed: refuse everything, or refuse nothing. **Only the pair means anything.** <!-- .element: class="text-lg" -->

---

<!-- .slide: id="reasoning-goodhart" -->

## Goodhart's Law Is the Theme of This Section

> When a measure becomes a target, it ceases to be a good measure.

<div class="card-grid cols-2">
<div class="card"><h4>Training-side (Module 7)</h4><p><strong>Reward hacking.</strong> The policy finds a way to score that the reward designer did not intend.</p></div>
<div class="card"><h4>Industry-side (this module)</h4><p><strong>Benchmark chasing.</strong> The field optimizes what is published, and the published number stops predicting real usefulness.</p></div>
</div>

Same mechanism, different scale. It is why the practical advice ends with "build your own held-out set." <!-- .element: class="text-lg" -->
