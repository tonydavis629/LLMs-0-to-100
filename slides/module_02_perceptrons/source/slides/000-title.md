:::divider id="title" title="LLMs 0 to 100" sub="Module 2: Perceptrons and Optimization"
Neurons to Networks
:::

---

<!-- .slide: id="review-1" -->

## Review: Module 1

:::columns cols="2" gap="30px"
**Shannon Entropy**

Average surprise per symbol:

$$H(X) = -\sum_{i} p(x_i) \log_2 p(x_i)$$

- Good model: high probability on likely events
- High probability = low surprise = few bits
+++
**N-gram Models**

Predict the next symbol from the previous $n - 1$ symbols:

$$P(w_k \mid w_{k-n+1} \ldots w_{k-1})$$

- More context = better predictions
- But the number of contexts grows exponentially
:::

---

<!-- .slide: id="review-2" -->

## Review: Cross-Entropy as a Loss

$$H(p, q) = -\sum_{x} p(x) \log q(x)$$

- Same formula, new job: it is now the **training objective**
- Training adjusts the model to minimize it
- Smaller cross-entropy = predictions closer to the truth

:::note
**Key idea:** Module 1 used cross-entropy to *score* a model. Module 2 uses it to *train* one.
:::

---

:::figure img="images/rosenblatt.jpg" name="Frank Rosenblatt" kicker="Built the First Learning Machine"
- Psychologist at Cornell, not a mathematician
- Built the Mark I Perceptron (1958) &mdash; a physical machine with photocells and motor-driven weight updates
- The first machine that could learn to classify patterns from data
- Proved the perceptron convergence theorem: if data is linearly separable, training converges in finite steps
- **His equation is our starting point today**
:::
