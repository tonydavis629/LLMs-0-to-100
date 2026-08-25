:::divider id="title" title="LLMs 0 to 100" sub="Module 3: Attention Mechanisms"
From Fixed Windows to Learned Lookups
:::

---

<!-- .slide: id="review-1" -->

## Review: Module 2

:::columns cols="2" gap="30px"
**The MLP**

Repeated linear transformations with nonlinearities between them:

$$\mathbf{h} = \sigma(W\mathbf{x} + \mathbf{b})$$

With enough hidden units, an MLP can approximate any continuous function (universal approximation theorem).
+++
**Limitations for Sequences**

- Fixed context window: token 1 cannot inform token 50
- A pattern learned at one position does not transfer to another
- Parameter count grows with context length
- No way for one token to select information from another
:::

---

<!-- .slide: id="review-2" -->

## Review: Why the Architecture Matters

:::columns cols="2" gap="30px"
**From Module 2**

- A single neuron cannot solve XOR: no linear boundary separates the classes
- The fix was architectural: a hidden layer with a nonlinearity
+++
**Same Principle, Bigger Scale**

Language structure an MLP does not match:

- Any token can depend on any earlier token
- The same pattern ("the ___") appears at every position
- Relevant context length varies per token

**The architecture must match the structure of the data.**
:::

