:::divider id="divider-practice" title="Running Evaluations in Practice" sub="Nobody writes this from scratch"
:::

---

<!-- .slide: id="practice-tooling" -->

## Some Practical Tools

<div class="card-grid cols-2">
<div class="card"><h4>lm-evaluation-harness (EleutherAI)</h4><p>The standard for likelihood-scored benchmarks like MMLU.</p></div>
<div class="card"><h4>HELM (Stanford, 2022)</h4><p>Many scenarios, many metrics, and a record of <strong>what was not covered</strong>.</p></div>
<div class="card"><h4>Inspect, lighteval, OpenAI Evals</h4><p>Frameworks for writing your own evaluations, agents included.</p></div>
<div class="card"><h4>VLMEvalKit, lmms-eval</h4><p>The same, for multimodal models.</p></div>
</div>

---

:::figure img="images/percy_liang.jpg" name="Percy Liang" kicker="Holistic Evaluation of Language Models (Stanford CRFM, 2022)" alt="Percy Liang"
Built **HELM**: evaluate many **scenarios** against many **metrics**, from accuracy to calibration to toxicity.

Model reports went from a row of accuracies to a grid, with the empty cells showing.
:::

---

<!-- .slide: id="practice-three-tiers" -->

## A Workable Setup for a Real Project

<div class="card-grid cols-3">
<div class="card"><h4>1. Fast eval</h4><p>Runs on <strong>every change</strong>, in seconds.</p></div>
<div class="card"><h4>2. Regression suite</h4><p>Cases that <strong>used to fail</strong>. Grows with every fix.</p></div>
<div class="card"><h4>3. Release eval</h4><p>Slow and broad, before you swap models. Public benchmarks plus your private set.</p></div>
</div>

**Keep the per-case outputs, not just the averages.** An average can hide a whole task collapsing. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="practice-handoff" -->

## Quality Is One Axis of a Deployment Decision

The other is **cost**: latency, memory, throughput.

<div class="card-grid cols-2">
<div class="card"><h4>This module</h4><p>Model A is two points better: <strong>at what</strong>, <strong>under what protocol</strong>?</p></div>
<div class="card"><h4>Next module</h4><p>Are two points worth four times the latency?</p></div>
</div>

**Next class: Deployment and Inference.** <!-- .element: class="text-lg" -->
