:::divider id="divider-tokenization" title="Images Are Not Words" sub="Visual tokenization"
:::

---

<!-- .slide: id="tok-raw" -->

## Start With the Raw Object

An image is a **tensor**, usually $H \times W \times C$, not a list of words.

:::columns cols="2" gap="34px"
**What makes images different**

- Neighboring pixels matter
- Two-dimensional position matters
- Small translations often should *not* change the meaning
+++
**Two recipes**

- **CNNs** build local edge and texture features with sliding filters (the older computer-vision path)
- **Transformers** treat an image as a **sequence of patches** (the recipe that unified vision with language models)
:::

To feed an image to a transformer, cut it into a sequence. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="fig-dosovitskiy" -->

:::figure img="images/dosovitskiy.png" name="Alexey Dosovitskiy and the ViT team" kicker="An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT, Google Research, 2020)"
- Split the image into fixed patches and treated each one as a token
- With enough training data, a plain transformer matches or beats convolutional networks
- The rest of the module builds on this framing
:::

---

:::manim id="tok-patchify" scene="patchify"
:::

---

<!-- .slide: id="tok-math" -->

## The Arithmetic of Patches

Split the image into non-overlapping $P \times P$ patches. The number of visual tokens is

$$N = \frac{H}{P} \cdot \frac{W}{P}$$

A $224 \times 224$ image with $16 \times 16$ patches gives $\textcolor{#4a9eff}{196}$ patch tokens. Each patch is projected, the visual analog of a token embedding:

$$z_i = W_{\text{patch}}\ \mathrm{flatten}(p_i) + b_{\text{patch}}$$

- Smaller patches: more detail, more tokens
- Attention cost grows with the **square** of sequence length (Module 4)

---

<!-- .slide: id="tok-position-resolution" -->

## Position and Resolution Are Not Optional

:::columns cols="2" gap="34px"
**Position embeddings**

- Flattening to 1-D **throws away layout**
- Row/column positions let the model tell "top" from "bottom"
+++
**Resolution**

- Many tasks live in **fine detail**: small text, tick labels, tiny objects, medical findings
- If the relevant pixels were never represented, the model fails
:::

A model can be "looking at" the image and still be blind to the part that matters. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="sq-registers" -->

## Side Quest: Vision Transformers Need Registers

<div class="img-figure">
  <img src="images/register_attention.png" alt="Two 16x16 patch-norm heatmaps: without registers a few patches have very high norm; with register tokens the features are clean and uniform" style="max-height: 380px;">
</div>

- Some ViTs hijack a few patches as high-norm scratch space (left)
- **Register tokens** add dedicated slots for that global computation, cleaning up the dense features (right)
- Attention over patches does not guarantee every patch token has a readable meaning

Darcet et al., "Vision Transformers Need Registers," <a href="https://arxiv.org/abs/2309.16588">arXiv:2309.16588</a> <!-- .element: class="text-lg" style="margin-top: 6px;" -->
