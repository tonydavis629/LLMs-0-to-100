<!-- .slide: id="review-1" -->

## Review: From Module 4

:::columns cols="2" gap="34px"
**The machinery we built**

- Text is tokenized, embedded, and run through a stack of decoder blocks
- The model outputs **logits**: one score per vocabulary token, at every position
- A decoding strategy (greedy, temperature, top-k, top-p) turns logits into the next token

Architecture is the **shape** of the computation. It does not yet say anything about the **weights**.
+++
**The question this module answers**

A freshly built transformer has random weights, so its output is noise.

**Pretraining** is the process that turns those random weights into ones that capture the statistical structure of language &mdash; using nothing but raw text.
:::

---

<!-- .slide: id="review-2" -->

## Review: Two Ideas We Reuse

:::columns cols="2" gap="34px"
**Causal masking (Module 3 and 4)**

Each position may attend only to itself and earlier positions.

So at every position the model can be asked to predict the **next** token without ever seeing the answer. One sequence becomes many prediction problems at once.
+++
**Cross-entropy (Module 2)**

The training signal is the same loss we used to train a classifier:

$$\mathcal{L} = -\log p_\theta(\text{true next token})$$

Assign **high probability to the token that actually comes next**. Averaged over a whole corpus, minimizing this is the entire objective.
:::

:::note
Causal masking plus cross-entropy is the whole idea of this module. Everything else &mdash; data pipelines, learning-rate schedules, scaling laws, distributed training &mdash; is engineering that makes this objective work at scale.
:::
