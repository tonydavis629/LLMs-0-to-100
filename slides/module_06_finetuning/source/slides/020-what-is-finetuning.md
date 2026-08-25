:::divider id="divider-what" title="What Finetuning Is" sub="Specializing a pretrained model"
:::

---

<!-- .slide: id="finetuning-definition" -->

## Finetuning, Defined

**Finetuning**: continue training a pretrained model on narrower data to specialize its behavior or knowledge.

:::columns cols="2" gap="34px"
**Pretraining**

- Learns language from **billions** of tokens
- Weeks on thousands of GPUs
+++
**Finetuning**

- Nudges a capable model with **thousands to millions** of examples
- Hours on modest hardware
:::

This is **transfer learning**: pretraining supplies the capability, finetuning teaches only format and behavior. That is why it is cheap. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="same-algorithm" -->

## The Same Algorithm, Almost

Identical optimization to pretraining. Three dials move:

:::columns cols="3" gap="22px"
**Learning rate**

- Much **smaller**
- Large steps would destroy what the model learned
+++
**Steps and data**

- **Far fewer** steps, **far less** data
- Often 1&ndash;3 passes over the set
+++
**The objective**

- Still **cross-entropy** on next tokens
- Now masked to the response
:::

Same forward pass, same loss, same backprop &mdash; the **same algorithm pointed at a new distribution**. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="finetuning-taxonomy" -->

## A Short Taxonomy

"Finetuning" covers several distinct things:

:::columns cols="3" gap="22px"
**Continued pretraining**

- Next-token prediction on **domain text** (medicine, code)
- Shifts knowledge
+++
**Supervised finetuning (SFT)**

- Learn to **follow instructions** from prompt&ndash;response pairs
- **This module**
+++
**Preference optimization**

- Learn from **comparisons** of better vs worse responses
- RLHF, DPO &mdash; **Module 7**
:::

---

<!-- .slide: id="catastrophic-forgetting" -->

## Catastrophic Forgetting

Push the weights too hard on narrow data and the model **loses general capability**.

:::columns cols="2" gap="34px"
**The failure**

- The finetuning set is tiny and skewed vs the pretraining corpus
- Aggressive updates overwrite broad features with narrow ones
- Better at your 1,000 examples, worse at everything else
+++
**This one failure mode motivates:**

- the **small** learning rate
- the **few** epochs
- **parameter-efficient** methods, which freeze the base entirely
:::
