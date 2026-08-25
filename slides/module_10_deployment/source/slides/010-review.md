<!-- .slide: id="review-pieces" -->

## Review: The Pieces This Module Serves

Weights are frozen from here. The problem: run them fast, for many users, at low cost.

<div class="card-grid cols-3">
<div class="card"><h4>Module 4 &middot; Architecture</h4><p>Generation is the forward pass <strong>in a loop</strong>: predict a token, append it, run again. Attention needs every earlier key and value.</p></div>
<div class="card"><h4>Module 5 &middot; Training at scale</h4><p>Training: mixed precision, batches, fleets of GPUs. Inference is a <strong>different workload</strong> with a different bottleneck.</p></div>
<div class="card"><h4>Module 9 &middot; Evaluation</h4><p>Quality is one axis. <strong>Cost</strong> is the other: latency, memory, throughput. Two points better and four times slower can still lose.</p></div>
</div>

---

<!-- .slide: id="review-question" -->

## Review: The Question This Module Answers

**What does it take to put a good checkpoint behind an API, and what does each token cost?**

:::columns cols="2" gap="34px"
**What carries over**

- The transformer forward pass and where the parameters live (Module 4)
- The KV idea from attention: position t needs every earlier key and value (Module 3)
- Sampling parameters: temperature, top-p (Module 4)
- The chat template that wraps every request (Module 6)
+++
**What is new**

- Why the bottleneck is **memory bandwidth**, not compute
- The **KV cache** and the memory a conversation occupies
- **Batching**: the economics of the entire API business
- **Quantization**, speculative decoding, and MoE at serving time
- The serving stack: vLLM and the OpenAI-compatible API
:::
