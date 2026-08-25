<!-- .slide: id="review-1" -->

## Review: From Module 4

:::columns cols="2" gap="34px"
**The machinery we built**

- Text is tokenized, embedded, and run through a stack of decoder blocks
- The model outputs **logits**: one score per vocabulary token, at every position
- A decoding strategy (greedy, temperature, top-k, top-p) turns logits into the next token
- Architecture is the **shape** of the computation, not the **weights**
+++
**The question this module answers**

- A fresh transformer has random weights, so its output is noise
- **Pretraining** turns random weights into useful ones
- The only input: raw text
:::

---

<!-- .slide: id="review-2" -->

## Review: Two Ideas We Reuse

:::columns cols="2" gap="34px"
**Causal masking (Module 3 and 4)**

- Each position attends only to itself and earlier positions
- So every position predicts the **next** token without seeing the answer
- One sequence = many prediction problems at once
+++
**Cross-entropy (Module 2)**

The same loss that trained the classifier:

$$\mathcal{L} = -\log p_\theta(\text{true next token})$$

Goal: assign **high probability to the token that actually comes next**.
:::

:::note
Causal masking + cross-entropy is the whole idea. The rest of the module &mdash; data pipelines, schedules, scaling laws, distributed training &mdash; is the engineering that makes it work at scale.
:::
