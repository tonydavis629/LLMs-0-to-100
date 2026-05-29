:::divider id="title" title="LLMs 0 to 100" sub="Module 2: Perceptrons and Optimization"
Neurons to Networks
:::

---

<!-- .slide: id="review-1" -->

## Review: Module 1

:::columns cols="2" gap="30px"
**Shannon Entropy**

Information is measurable. The entropy of a source measures the average surprise per symbol:

$$H(X) = -\sum_{i} p(x_i) \log_2 p(x_i)$$

A good model assigns high probability to likely events — low surprise, few bits.
+++
**N-gram Models**

Predict the next symbol from the previous $n - 1$ symbols:

$$P(w_k \mid w_{k-n+1} \ldots w_{k-1})$$

Higher order = more context = better predictions, but the number of possible contexts explodes exponentially.
:::

---

<!-- .slide: id="review-2" -->

## Review: Cross-Entropy as a Loss

In Module 1, cross-entropy measured how well an n-gram model predicted text:

$$H(p, q) = -\sum_{x} p(x) \log q(x)$$

The same formula is now a **training objective**: we adjust the model to make this number as small as possible. Smaller cross-entropy means predictions closer to the truth.

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
