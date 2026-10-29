:::divider id="divider-stack" title="The Modern Multimodal Stack" sub="And what evaluation and deployment inherit"
:::

---

<!-- .slide: id="stack-name-it" -->

## Name the Full Stack

<div class="img-figure"><img src="images/arch_full_stack.svg" alt="The carried diagram with every slot filled: image, audio, and video encoders (1) feed connectors (2) into one language model (3) beside the text lane; the model calls tools (5) and reads results back; a training loop (4) runs pretraining, alignment, multimodal SFT, and preference tuning, then repeats with new data" style="max-height: 420px;"></div>

(1) encoders, (2) connectors, (3) the same transformer, (4) the training loop from Modules 5 to 7 plus an alignment stage, (5) tools the model calls at run time. The new work is deciding **which vectors represent the non-text input**. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="stack-throughline" -->

## What Changed and What Didn't

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

A multimodal LLM is the **same model** with extra components that turn other modalities into vectors. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="sq-one-or-many" -->

## <span class="side-quest-badge">Side Quest</span> One Model, or a System of Models?

<div class="compare-table">
<table>
<thead><tr><th>Unified mixed-token model</th><th>Routed product stack</th></tr></thead>
<tbody>
<tr><td>One transformer over text, image, audio, video tokens</td><td>A language model wired to an image encoder, an OCR model, a retriever, and an image generator</td></tr>
<tr><td>The cleanest <strong>research</strong> objective</td><td>Often the cheaper, more <strong>reliable</strong> product</td></tr>
</tbody>
</table>
</div>

Products often choose the routed stack because each part can be tested, swapped, and fixed on its own. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="stack-handoff" -->

## Next Classes: Measuring It, Then Serving It

Two things get harder once the model reads more than text:

- **Scoring** it (Module 9): visual answers break exact match; benchmarks can often be passed without looking at the image
- **Running** it efficiently:

<div class="card-grid cols-2">
<div class="card"><h4>Vision encoders add latency</h4><p>Extra forward passes before the LLM even starts</p></div>
<div class="card"><h4>Images add preprocessing</h4><p>Resize, patch, and encode on every request</p></div>
<div class="card"><h4>Visual tokens inflate context</h4><p>Bigger KV cache, more attention cost</p></div>
<div class="card"><h4>Video and audio strain memory</h4><p>Bandwidth becomes the bottleneck</p></div>
</div>

Batching, KV-cache size, preprocessing, memory bandwidth, quantization, and model routing become the practical constraints. **That is Module 10.** <!-- .element: class="text-lg" -->
