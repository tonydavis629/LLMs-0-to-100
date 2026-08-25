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

## Review: The Through-Line Still Holds

Module 5: **next-token pretraining**. Module 6: **response format**. Module 7: **behavior from feedback**. Module 8 adds **perceptual conditions** to the context.

:::columns cols="2" gap="34px"
**Reused wholesale**

- The Module 4 transformer, unchanged
- Cross-entropy and the masked SFT loss of Modules 5 and 6
- The same SFT, preference, and RLHF post-training of Module 7
+++
**Newly hard**

- **Reward and evaluation**
- Visual and audio mistakes can be subtle, spatial, temporal, or safety-critical
:::
