:::divider id="divider-assistant" title="From Base Model to Assistant" sub="Why impressive pretraining is not yet a product"
:::

---

<!-- .slide: id="base-continues" -->

## A Base Model Continues Text

Pretraining produces a model that **continues text**, not one that reliably **follows your intent**. A base model is a pattern-matcher over its training distribution, so it will happily:

- Complete a chat transcript by inventing **both** sides of the conversation
- Continue a document, write code, or imitate a style
- Carry on harmful text, because that too is a pattern in the data

It is not refusing to help and it is not trying to help &mdash; it is just predicting the next token.

---

<!-- .slide: id="side-quest-base-vs-assistant" -->

## Side Quest: Same Prompt, Two Models

:::columns cols="2" gap="30px"
**Base model** sees a document to continue

```text
What is the capital of France?
What is the capital of Italy?
What is the largest ocean?
```
It continues the **pattern** &mdash; more quiz questions &mdash; instead of answering.
+++
**Assistant model** infers your intent

```text
What is the capital of France?

The capital of France is Paris.
```
It treats the text as a **request** and satisfies it.
:::

Same weights' worth of knowledge; completely different behavior. The difference is what comes after pretraining.

---

<!-- .slide: id="aligning-behavior" -->

## Shaping Behavior

:::columns cols="2" gap="34px"
**Instruction finetuning**

Continue training on a new data distribution: **prompts paired with desired responses**. This teaches the model that a prompt is something to satisfy, not merely continue.
+++
**Preference optimization and RL**

Further shape behavior toward **helpfulness, honesty, appropriate refusal, and tool use**, using human or model preferences as the signal.
:::

The optimization machinery is familiar &mdash; gradients, batches, a loss. What changes is the **data** and the **behavioral target**.

---

<!-- .slide: id="handoff" -->

## The Handoff to Module 6

:::columns cols="2" gap="34px"
**What pretraining gave us**

A base model that has absorbed the structure of language: grammar, facts, style, code, and latent skills &mdash; a powerful but unaligned foundation.
+++
**What comes next**

**Finetuning and alignment** turn that foundation into an assistant. Same optimizer, same backprop &mdash; new data, new target.
:::

Pretraining built the engine. Module 6 teaches it to drive.
