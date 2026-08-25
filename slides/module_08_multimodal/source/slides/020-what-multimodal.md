:::divider id="divider-what" title="What Multimodal Means" sub="Connecting language to perception"
:::

---

<!-- .slide: id="what-modality" -->

## A Modality Is a Type of Signal

- A **modality** is a signal with its own structure: text, image, audio, video, depth, sensor readings, code-execution traces
- A **multimodal model** connects two or more in one system

<div class="card-grid cols-2">
<div class="card"><h4>Understanding</h4><p>Condition on an image, clip, video, or document and answer in <strong>text</strong>.</p></div>
<div class="card"><h4>Retrieval</h4><p>Match images to captions, speech to transcripts, screenshots to descriptions.</p></div>
<div class="card"><h4>Action</h4><p>Perceive a state, decide, and call a tool, control a robot, or drive a UI.</p></div>
</div>

Main object: the **multimodal LLM**, a language model whose context includes non-text representations interleaved with text tokens. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="what-why" -->

## Why It Matters

Many real tasks connect **language to perception**:

- Read a chart, debug a screenshot, describe a medical image
- Transcribe speech, understand a lecture video, use a computer

The unifying abstraction: **every modality gets compressed into a sequence**, either

- **discrete tokens** (text tokens, audio codec tokens), or
- continuous **embedding vectors** (image patch features projected into the LLM's hidden size)

The recipe: **choose a representation, align it with language, train the model to use it.** <!-- .element: class="text-lg" style="margin-top: 10px;" -->

---

<!-- .slide: id="what-tensions" -->

## Two Tensions That Shape Everything

:::columns cols="2" gap="34px"
**Bandwidth**

- Text: symbolic, low-bandwidth
- Pixels and waveforms: continuous, high-bandwidth
- One high-resolution image or one minute of audio dwarfs a 1,000-token paragraph
+++
**Order**

- Language: ordered left to right
- Images: 2-D layout. Audio: fine-grained time. Video: both
- Flattening into a sequence throws away structure unless positional information restores it
:::

The whole module answers these two tensions: **compress the high-bandwidth signal, keep its structure once flattened.** <!-- .element: class="text-lg" style="margin-top: 12px;" -->
