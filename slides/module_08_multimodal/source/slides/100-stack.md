:::divider id="divider-stack" title="The Modern Multimodal Stack" sub="And what evaluation and deployment inherit"
:::

---

<!-- .slide: id="stack-name-it" -->

## Name the Full Stack

<div class="card-grid cols-3">
<div class="card"><h4>1. Encoders</h4><p>Modality-specific front ends: vision, audio, and more</p></div>
<div class="card"><h4>2. Connector</h4><p>A projector, cross-attention bridge, or shared tokenizer</p></div>
<div class="card"><h4>3. Language model</h4><p>The transformer that ties it all together</p></div>
<div class="card"><h4>4. Multimodal SFT</h4><p>Instruction tuning on image-conditioned dialogue</p></div>
<div class="card"><h4>5. Preference / RL</h4><p>Post-training with feedback (Module 7)</p></div>
<div class="card"><h4>6. Retrieval and tools</h4><p>Plus the deployment infrastructure around it</p></div>
</div>

**Same transformer, larger interface.** The model still moves vectors through attention layers; the new work is deciding **which vectors represent the non-text input.** <!-- .element: class="text-lg" -->

---

<!-- .slide: id="stack-throughline" -->

## The Through-Line Is Intact

:::columns cols="2" gap="34px"
**What each module added**

- Module 5: next-token **pretraining**
- Module 6: **response format**
- Module 7: **behavior** from feedback
- Module 8: **perceptual conditions** in the context
+++
**What never changed**

- Module 4's transformer
- Module 5's cross-entropy
- Module 6's masked loss
- Module 7's feedback loops
:::

Multimodal LLMs are not a new architecture. They are the **same model with components trained on other modalities**. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="sq-one-or-many" -->

## Side Quest: One Model, or a System of Models?

<div class="compare-table">
<table>
<thead><tr><th>Unified mixed-token model</th><th>Routed product stack</th></tr></thead>
<tbody>
<tr><td>One transformer over text, image, audio, video tokens</td><td>A language model, an image encoder, an OCR model, a retriever, and an image generator &mdash; wired together</td></tr>
<tr><td>The cleanest <strong>research</strong> objective</td><td>Often the cheaper, more <strong>reliable</strong> product</td></tr>
</tbody>
</table>
</div>

The cleanest research architecture is often **not** the best product architecture. A shipping system optimizes reliability, not elegance. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="stack-handoff" -->

## Next Classes: Measuring It, Then Serving It

Two things get harder once the model reads more than text:

- **Scoring** it (Module 9): visual answers break exact match; benchmarks can often be passed without looking at the image
- **Running** it efficiently:

<div class="card-grid cols-2">
<div class="card"><h4>Vision encoders add latency</h4><p>Extra forward passes before the LLM even starts</p></div>
<div class="card"><h4>Images add preprocessing</h4><p>Resize, patch, encode &mdash; per request</p></div>
<div class="card"><h4>Visual tokens inflate context</h4><p>Bigger KV cache, more attention cost</p></div>
<div class="card"><h4>Video and audio strain memory</h4><p>Bandwidth becomes the bottleneck</p></div>
</div>

Batching, KV-cache size, preprocessing, memory bandwidth, quantization, and model routing become the practical constraints. **That is Module 10.** <!-- .element: class="text-lg" -->
