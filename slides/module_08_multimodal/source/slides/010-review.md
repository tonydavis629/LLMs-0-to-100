<!-- .slide: id="review-post-training" -->

## Review: Where Module 7 Left Us

The modern stack so far:

- **Pretraining** gives broad capability
- **SFT** teaches instruction-following
- **RL** / preference optimization shapes behavior with scalar feedback

:::columns cols="2" gap="34px"
**Unchanged since Modules 3-4**

- Embeddings enter a sequence model
- **Attention** mixes information
- **Logits** predict the next token
- **Gradients** update weights
+++
**What changes now**

- The **input space**
- Images, audio, video, and sensor streams must become representations the model can use
:::

This module's question: **how do we turn non-text data into tokens or embeddings that live beside words?** <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="review-callback" -->

## Review: What Carries Over

Module 5: **next-token pretraining**. Module 6: **response format**. Module 7: **behavior from feedback**. Module 8 adds **perceptual conditions** to the context.

:::columns cols="2" gap="34px"
**Reused**

- The Module 4 transformer, unchanged
- Cross-entropy and the masked SFT loss of Modules 5 and 6
- The same SFT, preference, and RLHF post-training of Module 7
+++
**New**

- Encoders and connectors for non-text input
- Harder evaluation: visual and audio mistakes can be small, spatial, or tied to timing
:::

---

<!-- .slide: id="review-slots" -->

## One Template for Every Architecture

<div class="img-figure"><img src="images/arch_text_only.svg" alt="The template: a modality lane (image or audio, encoder, connector) above the familiar text lane (prompt, token embedding), both feeding the language model" style="max-height: 360px;"></div>

Modules 4 to 7 used only the bottom lane. The top lane is new. Every architecture in this module is drawn on this template, so you can compare them by what sits in each box. <!-- .element: class="text-lg" -->
