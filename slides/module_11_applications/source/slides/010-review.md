<!-- .slide: id="review-frozen" -->

## Review: The Model So Far

The earlier modules trained a model, measured it, and put it behind an API.

<div class="card-grid cols-4">
<div class="card"><h4>Module 5 &middot; Pretraining</h4><p>Next-token prediction over web text. <strong>In-context learning</strong> appeared without being trained for.</p></div>
<div class="card"><h4>Modules 6-7 &middot; Finetuning and RL</h4><p>SFT taught the <strong>chat template</strong> and instruction following. RL with <strong>verifiable rewards</strong> trained reasoning.</p></div>
<div class="card"><h4>Module 9 &middot; Evaluation</h4><p><strong>Benchmarks</strong> score a checkpoint. Agent benchmarks score whether the task got done. The average can hide what the per-task scores show.</p></div>
<div class="card"><h4>Module 10 &middot; Deployment</h4><p>The <strong>KV cache</strong>, batching, and a <strong>price per token</strong>.</p></div>
</div>

This module does not change any weights. Everything in it works by choosing which tokens go into the context window. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="review-callbacks" -->

## Review: The Question This Module Answers

The model is trained, evaluated, and deployed. **How do you build something useful on top of it?**

:::columns cols="2" gap="34px"
**What carries over**

- In-context learning (Module 5)
- The chat template and its roles (Module 6)
- Sampling parameters (Module 4)
- Contrastive embeddings (Module 8)
- Agent benchmarks (Module 9)
- The KV cache and token pricing (Module 10)
+++
**What is new**

- **Applications that use LLMs**: prompting, system prompts, structured output
- **Augmenting LLMs**: retrieval for knowledge the model lacks, tools for work it does badly
- **Agents**: tool calls in a loop, applied to coding, computer use, and deep research
- Cost, caching, and prompt injection in production
- Cybersecurity and LLMs for science
:::
