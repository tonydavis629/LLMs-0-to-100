:::divider id="divider-assistant" title="From Base Model to Assistant" sub="Why impressive pretraining is not yet a product"
:::

---

<!-- .slide: id="base-continues" -->

## A Base Model Continues Text

A base model **continues text**; it does not reliably **follow intent**. It will happily:

- Complete a chat transcript by inventing **both** sides
- Continue a document, write code, imitate a style
- Carry on harmful text, because that too is a pattern in the data

It is not refusing to help, and not trying to help. It is predicting the next token.

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

Same knowledge; different behavior. The difference is what comes after pretraining.

Try it (free Hugging Face account): paste the quiz prompt into the [Qwen3-4B-Base](https://huggingface.co/Qwen/Qwen3-4B-Base) inference widget, then into an instruction-tuned model on [HuggingChat](https://huggingface.co/chat), and compare.

---

<!-- .slide: id="aligning-behavior" -->

## Shaping Behavior

:::columns cols="2" gap="34px"
**Instruction finetuning**

- Continue training on **prompts paired with desired responses**
- Teaches: a prompt is something to satisfy, not continue
+++
**Preference optimization and RL**

- Shape behavior toward **helpfulness, honesty, appropriate refusal, tool use**
- Signal: human or model preferences
:::

Same machinery (gradients, batches, a loss). New **data**, new **behavioral target**.

---

<!-- .slide: id="handoff" -->

## Next Class: Finetuning

:::columns cols="2" gap="34px"
**What pretraining gave us**

- A base model with grammar, facts, style, code, latent skills
- Powerful but unaligned
+++
**What comes next**

- **Finetuning and alignment** turn it into an assistant
- Same optimizer, same backprop; new data, new target
:::

Pretraining built the engine. The next class teaches it to drive.
