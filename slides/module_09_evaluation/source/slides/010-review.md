<!-- .slide: id="review-numbers" -->

## Review: Every Module So Far Ended With a Number

Each stage introduced its own metric. This module collects them and adds the standard benchmarks.

<div class="card-grid cols-4">
<div class="card"><h4>Module 5 &middot; Pretraining</h4><p>Held-out <strong>loss</strong> and <strong>perplexity</strong>. No labels needed, just text the model never saw.</p></div>
<div class="card"><h4>Module 6 &middot; SFT</h4><p><strong>Task accuracy</strong> after finetuning: does the model answer the instruction at all?</p></div>
<div class="card"><h4>Module 7 &middot; RL</h4><p>A <strong>reward curve</strong> and before/after accuracy on the reverse task.</p></div>
<div class="card"><h4>Module 8 &middot; Multimodal</h4><p>Exact-match scoring gets <strong>brittle</strong> once the answer is about an image.</p></div>
</div>

**Each training stage produces a different kind of model, so each stage gets a different measurement.** <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="review-question" -->

## Review: The Question This Module Answers

You are handed a checkpoint. **Is it any good? How does the industry report that?**

:::columns cols="2" gap="34px"
**What carries over**

- Train/test splits from Module 2: a score on training data measures memorization
- Cross-entropy and perplexity from Module 5
- The chat template and answer format from Module 6
- Verifiable rewards from Module 7, reused here as **scoring** instead of training signal
+++
**What is new**

- The standard **benchmarks**: MMLU, GSM8K, HumanEval, MMMU, SWE-bench
- Scoring answers that have **no single right answer**
- Why the same model gets **different scores** at different labs
- What a benchmark number does **not** tell you
:::
