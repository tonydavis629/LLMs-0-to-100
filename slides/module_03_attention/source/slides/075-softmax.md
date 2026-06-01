:::divider id="divider-softmax" title="Softmax" sub="Turning scores into probabilities"
:::

---

<!-- .slide: id="softmax-purpose" -->

## From Logits to a Distribution

Softmax is the function that has been quietly doing the normalization step in every attention layer. It takes a vector of raw scores (**logits**) and turns them into a probability distribution:

$$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

:::columns cols="2" gap="30px"
**Why exponentiate?**

- $e^{z}$ is always positive, so every output is a valid probability
- It amplifies differences: a slightly larger logit gets a disproportionately larger share
- The outputs sum to exactly 1
+++
**Where it appears**

- **Inside attention:** over the keys, so each query's weights sum to 1
- **At the output:** over the whole vocabulary, to pick the next token
:::

---

<!-- .slide: id="softmax-interactive" -->

:::interactive id="softmax-explorer" widget="softmaxExplorer" title="Logits to Probabilities (and Temperature)"
:::

---

<!-- .slide: id="softmax-temperature" -->

## Temperature

Dividing the logits by a **temperature** $T$ before softmax controls how peaked the distribution is:

$$p_i = \text{softmax}(z / T)_i$$

:::columns cols="3" gap="20px"
**$T < 1$**

Sharper. The model commits to its top choice. More deterministic text.
+++
**$T = 1$**

The distribution exactly as the model produced it.
+++
**$T > 1$**

Flatter. More of the probability mass spreads to other tokens. More varied, riskier text.
:::

:::note
The same scaling idea appeared in attention: dividing $QK^T$ by $\sqrt{d_k}$ is a fixed temperature that keeps the softmax from saturating as the dimension grows.
:::
