<!-- .slide: id="review-frozen" -->

## Review: The Training Story Is Over

- Modules 5 through 7 changed the weights
- Module 10 put them behind an API: tokens in, tokens out
- **Nothing in this module touches a weight**

<div class="card-grid cols-4">
<div class="card"><h4>Module 5 &middot; Pretraining</h4><p>Capability: next-token prediction over the web, and <strong>in-context learning</strong> as a surprise side effect.</p></div>
<div class="card"><h4>Module 6 &middot; SFT</h4><p>Behavior: the <strong>chat template</strong>, instruction following, format compliance.</p></div>
<div class="card"><h4>Module 7 &middot; RL</h4><p>Judgment: reasoning trained in with <strong>verifiable rewards</strong>.</p></div>
<div class="card"><h4>Module 10 &middot; Serving</h4><p>Delivery: the KV cache, batching, and a <strong>price per token</strong>.</p></div>
</div>

Every technique in this module **chooses which tokens sit in front of a frozen model.** The context window is the programmable surface. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="review-callbacks" -->

## Review: Three Things We Promised to Finish

:::columns cols="2" gap="34px"
**What carries over**

- GPT-3's few-shot **in-context learning** (Module 5): becomes the primary engineering tool
- The **chat template** (Module 6): roles are special tokens the model was finetuned to respect
- **Sampling parameters** (Module 5): become product knobs
+++
**What gets paid off**

- Module 8's contrastive objective trains **retrieval embeddings**
- Module 9f benchmarked agents; this module **builds one**
- Module 10's KV cache returns as a priced product feature
:::

Plan: prompting, retrieval, tools, agents, then a case study using all four. <!-- .element: class="text-lg" style="margin-top: 10px;" -->
