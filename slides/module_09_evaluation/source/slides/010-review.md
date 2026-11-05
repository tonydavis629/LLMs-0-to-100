<!-- .slide: id="review-numbers" -->

## Review: Every Module So Far Ended With a Number

<div class="card-grid cols-4">
<div class="card"><h4>Module 5 &middot; Pretraining</h4><p>Held-out <strong>loss</strong> and <strong>perplexity</strong>.</p></div>
<div class="card"><h4>Module 6 &middot; SFT</h4><p><strong>Task accuracy</strong>: does it answer the instruction?</p></div>
<div class="card"><h4>Module 7 &middot; RL</h4><p>A <strong>reward curve</strong>, plus before/after accuracy.</p></div>
<div class="card"><h4>Module 8 &middot; Multimodal</h4><p>Exact match gets <strong>brittle</strong> once the answer is about an image.</p></div>
</div>

**Each training stage makes a different kind of model, so each needs a different measurement.** <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="review-question" -->

## Review: The Question This Module Answers

You are handed a checkpoint. **Is it any good, and how would you report that?**

:::columns cols="2" gap="34px"
**What carries over**

- Train/test splits (Module 2)
- Perplexity (Module 5)
- Chat templates (Module 6)
- Verifiable rewards (Module 7), now used for **scoring**
+++
**What is new**

- The standard **benchmarks**
- Scoring answers with **no single right answer**
- Why labs report **different scores** for one model
- What a benchmark number does **not** tell you
:::
