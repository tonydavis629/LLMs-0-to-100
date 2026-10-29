:::divider id="divider-sft" title="Instruction Tuning and Grounding" sub="Teaching assistant behavior"
:::

---

<!-- .slide: id="sft-interleaved" -->

## Interleaved Documents

Alignment gives matching, **not** assistant behavior. Modern pretraining adds **interleaved image-text documents**: web pages, papers, manuals, notebooks.

:::columns cols="2" gap="34px"
**A caption teaches**

Global matching: this whole image goes with this whole sentence.
+++
**A document teaches reference**

- "The **second** diagram," "**this** chart," "the example **above**"
- The data shape needed to reason across **multiple** images
:::

---

<!-- .slide: id="sft-unit" -->

## The Instruction-Tuning Unit

After alignment, the data unit is an **image plus a prompt-response pair**:

- "What color is the triangle?"
- "Read the axis label."
- "Describe the error message in this screenshot."
- "Which object is left of the chair?"

Chat templates gain a **placeholder** for the non-text input:

```text
<|user|> <image> What color is on top? <|end|>
<|assistant|> red <|end|>
```

`<image>` is **not** the image. The wrapper replaces it with the projected visual embeddings from the last section. <!-- .element: class="text-lg" style="margin-top: 8px;" -->

---

<!-- .slide: id="sft-data-sources" -->

## Where the Data Comes From

<div class="card-grid cols-3">
<div class="card"><h4>Captions and alt text</h4><p>Broad image-language alignment</p></div>
<div class="card"><h4>Interleaved documents</h4><p>PDFs, notebooks, and manuals with several images</p></div>
<div class="card"><h4>Visual QA</h4><p>Targeted answers to specific questions</p></div>
<div class="card"><h4>OCR and documents</h4><p>Reading text, forms, and layout in images</p></div>
<div class="card"><h4>Referring expressions</h4><p>Locating objects named by language</p></div>
<div class="card"><h4>Synthetic dialogue</h4><p>Written by a stronger model and filtered</p></div>
</div>

The synthetic route repeats Module 6: a stronger model writes the training data, a filter keeps the good parts. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="sft-grounding" -->

## Grounding: Words to the Right Pixels

**Grounding**: connect words to the right parts of the input. "The red square" should refer to *the red square in this image*, not to red squares being common in the dataset.

The stress test is **spatial language**:

<div class="card-grid cols-4">
<div class="card"><p>left / right</p></div>
<div class="card"><p>above / below</p></div>
<div class="card"><p>inside / outside</p></div>
<div class="card"><p>count</p></div>
<div class="card"><p>same / different</p></div>
<div class="card"><p>nearest / farthest</p></div>
</div>

These are easy for people. For a model, they show whether it **uses the visual layout** or guesses from language patterns. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="sft-failure-modes" -->

## Two Failure Modes

:::columns cols="2" gap="34px"
**OCR is separate from recognition**

- A model can recognize **objects** yet fail to **read** small text
- Or read the text but misunderstand the **layout** of a form, chart, or UI
+++
**Visual hallucination**

- The multimodal analog of text hallucination
- The model **confidently describes** objects, text, colors, or relationships **not present** in the image
:::

Instruction tuning makes the model more helpful. It can also teach **shortcut behavior**. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="sft-shortcuts" -->

## The Language-Prior Shortcut

If training answers carry **strong language priors**, the model learns to answer from text patterns instead of the image. If 90% of "what color is the banana?" answers are "yellow," a model scores well without looking.

:::columns grid="0.7fr 1.5fr" gap="22px" valign="center"
<div class="img-figure"><img src="images/vlm_biased_zebra.png" alt="A photo of a zebra edited to have five legs" style="max-height: 200px;"></div>

"How many legs does this animal have?" The edited zebra has **5**. Models tend to answer with the prior, **4**.
+++
<div class="img-figure"><img src="images/vlm_biased_overview.png" alt="VLMBias examples: a three-legged chicken, a four-pointed Mercedes star, a 14-stripe US flag; Gemini 2.5 Pro, Claude Sonnet 3.7, GPT-4.1, o3 and o4-mini mostly give the textbook count" style="max-height: 320px;"></div>
:::

Across 7 domains, frontier models averaged **17% accuracy** counting edited subjects (Vo et al., "Vision Language Models are Biased," 2025, <a href="https://arxiv.org/abs/2505.23941">arXiv:2505.23941</a>). The defense is **balanced data**, like the exercise's synthetic scenes. <!-- .element: class="text-lg" style="margin-top: 6px;" -->

---

<!-- .slide: id="sq-perception-reasoning" -->

## <span class="side-quest-badge">Side Quest</span> Perception or Reasoning?

:::columns grid="0.9fr 1.1fr" gap="30px" valign="center"
<div class="img-figure"><img src="images/chart_probe.svg" alt="Bar chart of revenue: Q1 42, Q2 47, Q3 61, Q4 58" style="max-height: 300px;"></div>
+++
"How much did revenue grow from Q2 to Q3?" The right answer is **14**. The model says **19**. Two different bugs give that answer:

- **Perception**: it read Q3 as 66, then subtracted correctly
- **Reasoning**: it read 47 and 61 correctly, then subtracted wrong

To tell them apart, ask it to list every bar's value first, or hand it the numbers as text. If the text version is right, the model never had the right numbers.
:::

The fixes differ: higher resolution or a tighter crop when perception failed, reasoning-focused training when the arithmetic failed. Calling every miss a "reasoning failure" hides which one you need. <!-- .element: class="text-lg" -->
