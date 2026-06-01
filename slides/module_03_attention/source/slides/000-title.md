:::divider id="title" title="LLMs 0 to 100" sub="Module 3: Attention Mechanisms"
From Fixed Windows to Learned Lookups
:::

---

<!-- .slide: id="review-1" -->

## Review: Module 2

:::columns cols="2" gap="30px"
**The MLP**

A multi-layer perceptron applies repeated linear transformations with nonlinear activations between them:

$$\mathbf{h} = \sigma(W\mathbf{x} + \mathbf{b})$$

An MLP with enough hidden units can approximate any continuous function (universal approximation theorem).
+++
**Limitations for Sequences**

An MLP over a fixed window of tokens treats each position identically, but:

- The context window is fixed: token 1 cannot directly inform token 50
- Flattening token vectors destroys the idea that the same pattern can appear in many positions
- Parameter count grows with context length
- There is no direct mechanism for one token to select information from another token
:::

---

<!-- .slide: id="review-2" -->

## Review: Why the Architecture Matters

:::columns cols="2" gap="30px"
**From Module 2**

The single neuron could not solve XOR because a linear boundary cannot separate entangled classes. The fix was a hidden layer with a nonlinearity &mdash; the architecture had to match the structure of the problem.
+++
**Same Principle, Bigger Scale**

Language has structure that an MLP does not match:

- Every token can depend on any earlier token, not just its neighbors
- The same syntactic pattern ("the ___") appears at every position
- The relevant context length varies per token

**The architecture must match the structure of the data.**
:::

