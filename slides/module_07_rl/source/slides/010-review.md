<!-- .slide: id="review-sft" -->

## Review: Where Module 6 Left Us

SFT turned the base model into an **instruct model**: it treats a prompt as a request.

:::columns cols="2" gap="34px"
**What SFT gave us**

- Imitates good demonstrations
- Same machinery as pretraining: cross-entropy on fixed target tokens
+++
**Three motivations for RL**

- Optimizes imitation, rather than an explicit outcome score
- Ordinary SFT has no explicit chosen-versus-rejected signal
- **Exposure bias**: trained on ground-truth prefixes, generates from its own outputs
:::

This module changes the **training signal** while reusing the same optimization machinery. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="review-machinery" -->

## Review: The Machinery Does Not Change

:::columns cols="2" gap="34px"
**Same as Modules 2, 5, 6**

- Forward pass through the transformer (Module 4)
- **Backpropagation** and **AdamW** (Module 2)
- Gradient clipping, batching, learning-rate control
+++
**New in RL**

- No fixed correct token at each position
- Signal: a **scalar reward** on the model's **own samples**
- Objective: **expected reward**, not cross-entropy to a label
:::
