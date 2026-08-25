<!-- .slide: id="loss-masking-intro" -->

## Loss Masking

The example holds both prompt and response. We only want to teach the **response**.

- Compute cross-entropy **only over response tokens**
- Ignore prompt positions

:::columns cols="2" gap="34px"
**Why not train on the prompt?**

- The model would learn to **generate user turns**
- We want it to answer prompts, not write them
+++
**How**

- Set prompt targets to `-100`
- `ignore_index=-100` skips those terms
:::

Same cross-entropy as Module 5, restricted to the completion. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="masked-ce-equation" -->

## The Masked Objective

$$\mathcal{L} = -\frac{1}{|R|}\sum_{t \in R}\log p_\theta\left(x_t \mid x_{<t}\right)$$

The Module 5 loss with one change: the sum runs only over $R$, the **response-token positions**.

:::columns cols="2" gap="34px"
- $R$: the assistant's response tokens
- $|R|$: their count; the average is over these only
- $x_{<t}$: the full prefix &mdash; the model still **reads** the prompt
+++
- The prompt is **conditioning**, not a **target**
- One forward pass; the mask zeroes the prompt's loss terms
- Multi-turn: mask every assistant turn, train several responses per sequence
:::

---

:::interactive id="loss-mask-explorer" widget="lossMask" title="Which Tokens Carry Loss"
:::

---

:::manim id="loss-mask-anim" scene="loss-mask"
:::

---

<!-- .slide: id="side-quest-mask-ablation" -->

## Side Quest: What If You Skip the Mask?

Train on the **whole** sequence, prompt included, and the model learns to model user turns too.

:::columns cols="2" gap="34px"
**The symptom**

- **Hallucinates its own prompts** at generation time
- Invents a `<|user|>` question and answers it
- Echoes instruction-shaped text instead of responding
+++
**Why**

- You trained it to predict prompt tokens, so it generates them
- The mask tells the model **whose turn it is learning to write**
:::

The mask encodes **what behavior we teach**. The exercise extra credit shows this directly. <!-- .element: class="text-lg" style="margin-top: 12px;" -->
