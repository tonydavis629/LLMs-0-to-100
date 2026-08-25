<!-- .slide: id="chat-template-intro" -->

## Chat Templates and Special Tokens

- A conversation has **roles**: system, user, assistant
- The model sees only a **flat sequence of token ids** (Module 4)
- A **chat template** serializes roles into that stream with **special tokens**

:::columns cols="2" gap="34px"
**The markers (ChatML-style)**

- `<|user|>` starts a user turn
- `<|assistant|>` starts an assistant turn
- `<|end|>` ends a turn
+++
**Atomic, not spelled out**

- Each marker is **one token id**, not seven characters
- One embedding per marker, as in real templates
:::

Added to the Module 5 vocabulary: 65 characters become 69 tokens. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

:::manim id="chat-template-anim" scene="chat-template"
:::

---

<!-- .slide: id="generation-prompt" -->

## Prompting at Generation Time

To **ask** the model something: build the same template, stop right after the assistant marker.

```text
<|user|> uppercase: hello <|end|> <|assistant|>
```

- The model continues from there
- If it learned the format, it produces the **response**, then `<|end|>`
- The assistant marker now means **"your turn to answer"** &mdash; that is the whole difference between base model and assistant
