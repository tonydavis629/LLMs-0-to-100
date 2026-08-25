:::divider id="divider-data" title="Data Quality, Contamination, Memorization" sub="Pretraining data is a design decision, not neutral infrastructure"
:::

---

<!-- .slide: id="more-not-better" -->

## More Data Is Not Automatically Better

Beyond a point, **what** you train on matters more than how much:

:::columns cols="2" gap="34px"
- **Quality** &mdash; a smaller corpus of clean text can beat a larger pile of boilerplate and spam
- **Diversity** &mdash; many domains and styles generalize better than one repeated genre
+++
- **Deduplication** &mdash; removing near-duplicates improves generalization, even though it removes tokens
- **Domain balance** &mdash; the ratio of code, prose, and academic text shapes what the model is good at
:::

---

<!-- .slide: id="side-quest-memorization" -->

## Side Quest: Memorization vs Generalization

Repeat one passage many times in a tiny dataset, and the model **memorizes that exact string**.

:::columns cols="2" gap="34px"
**Memorization**

- "The model learned this exact character sequence"
- Verbatim recall: a privacy and copyright liability
- Not a transferable skill
+++
**Generalization**

- "The model learned a pattern it can reuse"
- The goal of pretraining
- Why we **deduplicate** and watch **validation loss**
:::

Duplicated text quietly turns generalization back into memorization.

---

<!-- .slide: id="contamination" -->

## Benchmark Contamination

Evaluation examples leak into pretraining data; the model scores by **recall**, not ability.

- The model may have literally seen the test questions and answers
- Benchmark numbers then overstate real capability
- Decontamination tries to remove known benchmark text; a suspiciously high score deserves scrutiny
- Easy to introduce by accident (benchmarks live on the scraped web), hard to fully rule out

---

<!-- .slide: id="data-mixture" -->

## Data Mixture Shapes Behavior

:::columns cols="3" gap="20px"
**Code-heavy**

- Improves coding ability
- Less obviously: some structured reasoning
+++
**Academic text**

- Shifts style
- Broadens factual knowledge
+++
**Conversational text**

- Changes how the model handles turns and informal language
:::

:::note
Gao, Biderman, and collaborators at EleutherAI built **The Pile**: 800GB, 22 curated sources, the standard reference corpus for open pretraining.
:::

---

<!-- .slide: id="side-quest-data-wall" -->

## Side Quest: The Data Wall

What if high-quality **human** text, not model size, is the scarce resource?

- The best public text is finite; the largest models already train on much of it
- Responses: aggressive **curation and deduplication**, careful **domain balance**, **synthetic data** from other models
- The new question is not "can we afford a bigger model?" but "is there enough good data to feed it?"

A recurring theme in later modules; "just scale up" is incomplete.
