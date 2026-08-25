:::divider id="divider-practice" title="Running Evaluations in Practice" sub="Nobody writes this from scratch"
:::

---

<!-- .slide: id="practice-tooling" -->

## The Tooling to Know by Name

<div class="card-grid cols-2">
<div class="card"><h4>lm-evaluation-harness (EleutherAI)</h4><p>The de facto standard for likelihood-scored academic benchmarks, and the engine behind the Open LLM Leaderboard. If two labs report the same MMLU number, this is usually why.</p></div>
<div class="card"><h4>HELM (Stanford, 2022)</h4><p>Evaluate many scenarios with many metrics at once, and <strong>report what is not covered</strong>. Shifted the field from one headline number to multi-metric reporting.</p></div>
<div class="card"><h4>Inspect, lighteval, OpenAI Evals</h4><p>General frameworks for writing your own evaluations, including agent and tool-use tasks.</p></div>
<div class="card"><h4>VLMEvalKit, lmms-eval</h4><p>The multimodal equivalents, covering the multimodal benchmarks.</p></div>
</div>

---

:::figure img="images/percy_liang.jpg" name="Percy Liang" kicker="Holistic Evaluation of Language Models (Stanford CRFM, 2022)" alt="Percy Liang"
Argued that one headline accuracy hides most of what matters, then built infrastructure to evaluate many **scenarios** against many **metrics**: accuracy, calibration, robustness, fairness, bias, toxicity, efficiency. HELM also reports **what was not measured**.

Before HELM, a model report was a row of accuracies. After it, a grid, with the empty cells visible.
:::

---

<!-- .slide: id="practice-three-tiers" -->

## A Workable Setup for a Real Project

<div class="card-grid cols-3">
<div class="card"><h4>1. Fast eval</h4><p>A small set you run on <strong>every change</strong> during development. Seconds, not hours. Catches obvious breakage immediately.</p></div>
<div class="card"><h4>2. Regression suite</h4><p>Cases that <strong>used to fail and must keep passing</strong>. Grows every time you fix something. This is the file that saves you.</p></div>
<div class="card"><h4>3. Release eval</h4><p>Slower and broader, run before you swap models in production. Includes the public benchmarks and your own private set.</p></div>
</div>

**Keep the per-case outputs, not just the averages.** Reading failures turns evaluation into a fix. An aggregate can hide a total regression on one task inside a small overall gain. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="practice-handoff" -->

## Quality Is One Axis of a Deployment Decision

The other axis is **cost**: latency, memory, and throughput.

<div class="card-grid cols-2">
<div class="card"><h4>This module</h4><p>Model A is two points better than model B: <strong>at what</strong>, <strong>under what protocol</strong>, <strong>with what confidence</strong>.</p></div>
<div class="card"><h4>Not this module</h4><p>Whether two points are worth four times the latency and eight times the memory. A model you cannot afford to serve is not a better product.</p></div>
</div>

**Next class: Deployment and Inference.** Where the two points meet the memory bandwidth. <!-- .element: class="text-lg" -->
