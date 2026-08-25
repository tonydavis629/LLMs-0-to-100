<!-- .slide: id="review-base-model" -->

## Review: Where Module 5 Left Us

Pretraining ran next-token prediction over billions of tokens and produced a **base model**.

:::columns cols="2" gap="34px"
**Can**

- Continue text fluently
- Sample plausible continuations of any prefix
+++
**Cannot**

- Follow intent
- Asked to "list three colors," it may continue with more questions
- It was never trained to treat a prompt as a request
:::

The base model has the **knowledge**. It lacks the **behavior** of an assistant. <!-- .element: class="text-lg" style="margin-top: 14px;" -->

---

<!-- .slide: id="review-same-machinery" -->

## Review: The Machinery Does Not Change

Finetuning reuses the machinery from Modules 2 and 5.

:::columns cols="2" gap="34px"
**Same as pretraining**

- The forward pass through the transformer (Module 4)
- The **cross-entropy** loss (Module 5)
- **Backpropagation** and **AdamW** (Module 2)
- Batching, gradient clipping, learning-rate control
+++
**What is new**

- The **data**: prompt&ndash;response pairs, not raw text
- The **behavioral target**: follow intent, not just continue
- A **loss mask**: train on the response, not the prompt
- A much **smaller** learning rate and **far fewer** steps
:::
