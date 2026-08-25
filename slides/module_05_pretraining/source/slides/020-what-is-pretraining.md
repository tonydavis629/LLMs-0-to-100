:::divider id="divider-what" title="What Pretraining Is" sub="Learning the structure of language from raw text"
:::

---

<!-- .slide: id="self-supervised" -->

## Self-Supervised Learning

Pretraining: learn from broad raw text **before** specializing for any task.

:::columns cols="2" gap="34px"
**Supervised learning**

- Humans label the data: image "cat", review "positive"
- Labels are expensive and scarce
+++
**Self-supervised learning**

- The label is already in the text: the **next token**
- No human annotation
- A sequence of length $T$ yields $T$ training examples
:::

Free supervision is why pretraining can consume trillions of tokens.

---

<!-- .slide: id="base-model" -->

## The Result Is a Base Model

<div class="stage-flow">
<svg viewBox="0 0 900 150" role="img" aria-label="Raw text becomes a base model through pretraining; the base model becomes an instruct model through finetuning"><defs><marker id="sf" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#f5a623"></path></marker><marker id="sg" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#8892a4"></path></marker></defs><rect x="20" y="50" width="180" height="56" rx="10" fill="rgba(74,158,255,0.10)" stroke="rgba(74,158,255,0.5)" stroke-width="1.4"></rect><text x="110" y="76" text-anchor="middle" font-size="16" fill="#e8eaf0">raw text</text><text x="110" y="95" text-anchor="middle" font-size="12" fill="#8892a4">web, books, code</text><line x1="206" y1="78" x2="354" y2="78" stroke="#f5a623" stroke-width="2.5" marker-end="url(#sf)"></line><text x="282" y="66" text-anchor="middle" font-size="15" fill="#f5a623">pretraining</text><text x="282" y="98" text-anchor="middle" font-size="11" fill="#8892a4">next-token, at scale</text><rect x="360" y="50" width="180" height="56" rx="10" fill="rgba(63,185,80,0.12)" stroke="rgba(63,185,80,0.6)" stroke-width="1.6"></rect><text x="450" y="76" text-anchor="middle" font-size="16" fill="#e8eaf0">base model</text><text x="450" y="95" text-anchor="middle" font-size="12" fill="#8892a4">this module</text><line x1="546" y1="78" x2="694" y2="78" stroke="#8892a4" stroke-width="2.5" stroke-dasharray="5 4" marker-end="url(#sg)"></line><text x="622" y="66" text-anchor="middle" font-size="15" fill="#8892a4">finetuning</text><text x="622" y="98" text-anchor="middle" font-size="11" fill="#8892a4">Module 6</text><rect x="700" y="50" width="180" height="56" rx="10" fill="rgba(136,146,164,0.10)" stroke="rgba(136,146,164,0.45)" stroke-width="1.4"></rect><text x="790" y="76" text-anchor="middle" font-size="16" fill="#8892a4">instruct model</text><text x="790" y="95" text-anchor="middle" font-size="12" fill="#8892a4">follows instructions</text></svg>
</div>

- Boxes: **states of the weights**. Arrows: **training processes**
- A base model knows grammar, facts, style, and code patterns
- It is **not** yet a helpful assistant &mdash; that is finetuning (Module 6)

---

<!-- .slide: id="why-enough" -->

## "Just" Predicting the Next Token

To push the loss lower, the model must learn everything that makes text predictable:

:::columns cols="2" gap="30px"
- **Grammar and syntax** &mdash; to keep sentences well-formed
- **Facts about the world** &mdash; "the capital of France is ___"
- **Style and register** &mdash; legal text, poetry, code comments
+++
- **Code structure** &mdash; balanced brackets, valid identifiers
- **Task formats** &mdash; question/answer, translation, summaries
- **Implicit reasoning** &mdash; arithmetic, deduction, multi-step chains

:::

One simple objective + enormous diverse data = broad capability.

---

<!-- .slide: id="few-shot" -->

## A Striking Side Effect: In-Context Learning

A large enough base model infers a task from **examples in the prompt**:

:::columns cols="2" gap="30px"
```text
sea -> mer
sky -> ciel
cat -> chat
dog ->
```
+++
- The model was only trained to continue text
- It continues the **pattern** and answers `chien`
- This is **few-shot / in-context learning**
- Emergent from scale, not explicitly trained
:::

:::note
Nobody designed a "few-shot mode" into the objective; it appears for free at scale. More on prompting in Module 10.
:::
