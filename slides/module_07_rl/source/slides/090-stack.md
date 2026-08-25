:::divider id="divider-stack" title="The Post-Training Stack"
:::

---

<!-- .slide: id="full-recipe" -->

## The Full Modern Recipe

:::columns cols="3" gap="22px"
**Pretrain** (Module 5)

Next-token prediction over broad text. Builds the **engine**: knowledge and fluency.
+++
**Supervised finetune** (Module 6)

Prompt&ndash;response pairs. Teaches the model to **follow intent**.
+++
**RL post-training** (Module 7)

Preference and verifiable rewards. Tunes **behavior and reasoning**.
:::

Each stage reuses the **same optimization machinery**. Only the data and the target change. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="through-line" -->

## The Through-Line of the Course

:::columns cols="2" gap="34px"
**What never changed**

Gradients, backpropagation, AdamW. From the perceptron in Module 2 to reasoning models here, the optimizer is the same.
+++
**What changed each module**

The **data** and the **target**.

- Module 5: raw text, next token
- Module 6: pairs, masked response
- Module 7: a **scalar reward** on the model's own samples
:::

---

<!-- .slide: id="handoff-module-8" -->

## Next Class: Multimodal Models

Same transformer, same training stack. The **inputs expand beyond text**.

- **Multimodal** models bring images, audio, and video into the token stream
- The optimization **stays**; the **modality grows**

You now know how a base model becomes a reasoning assistant. Next: how it learns to **see and hear**. <!-- .element: class="text-lg" style="margin-top: 12px;" -->
