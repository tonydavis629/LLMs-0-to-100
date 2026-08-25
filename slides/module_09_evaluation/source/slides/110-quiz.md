:::divider id="divider-quiz" title="Quiz"
:::

---

:::quiz id="quiz-base-model" title="Measuring a Model That Will Not Answer"
A base model cannot follow instructions. What can you still measure about it, and why is perplexity not enough?
+++
**Short answer: you can measure how well it predicts held-out text (loss and perplexity) and what it knows (multiple choice scored by likelihood). Perplexity is not enough because it rewards fluent continuation, not correct answers.**

- Neither requires the model to produce an answer on demand
- Perplexity rewards **plausible** next tokens, not **right** ones
- Near-identical perplexity can hide tens of points of task difference
- In the exercise, the lower-perplexity model is the one that collapsed on three of four tasks
:::

---

:::quiz id="quiz-tokenizer" title="Why Perplexity Does Not Travel"
Why are perplexity numbers not comparable between two models with different tokenizers, and what is reported instead?
+++
**Short answer: perplexity is per token, and a bigger vocabulary means fewer, larger tokens per unit of text. Report bits per byte instead.**

- Model A splits a word into five tokens, model B into two: five predictions versus two, over different vocabularies
- The per-token averages measure different things
- **Bits per byte** divides by text length instead of token count
- The denominator becomes a property of the data, so the number compares across model families
:::

---

:::quiz id="quiz-likelihood" title="Why Score by Likelihood at All"
Why do most academic benchmarks score multiple choice by likelihood rather than by reading the model's generated answer?
+++
**Short answer: likelihood scoring works on models that cannot follow instructions, and it removes answer extraction (a large source of protocol variance) from the measurement.**

- A base model asked "A, B, C, or D?" may not produce a letter at all
- Likelihood scoring appends each candidate, scores it, takes the highest: no generation
- Runs on any checkpoint; the number depends on weights, not a regex
- Cost: a model whose generation collapsed can still score perfectly, as in the exercise
:::

---

:::quiz id="quiz-protocol" title="Same Model, Different Numbers"
Two labs report different MMLU scores for the same open-weight model. Give three things about the protocol that could explain the gap.
+++
**Short answer: shot count, prompt template, and scoring shape. Also answer extraction, normalization, and length normalization of the likelihoods.**

Each of these moves the number by points on identical weights:

- Few-shot example count, and which examples
- Prompt formatting: chat markers, option labels A&ndash;D versus 1&ndash;4
- Likelihood scoring versus reading generated text, and length normalization
- The answer-extraction regex and normalization rules

A benchmark number belongs to a model *and* a protocol. Reproducible reports log the harness version.
:::

---

:::quiz id="quiz-reward-curve" title="The Reward Went Up"
The GRPO model's reward curve climbed steadily during training. Why is that not evidence that the model got better?
+++
**Short answer: the reward is exactly what was optimized, measured on the training prompts. A climbing curve shows the optimizer works, not that any ability generalized or survived.**

The curve is a **training diagnostic**, like a falling loss curve. It cannot tell you whether:

- The gain transfers to held-out prompts
- The model found a degenerate way to score
- Unrelated abilities were destroyed on the way

All three need separate held-out evaluations. In the exercise, the reward climbed from 0.5 to 0.93 and the suite score fell 33 points.
:::

---

:::quiz id="quiz-passk" title="pass@1 Up, pass@10 Flat"
A model's pass@1 improves after RL but pass@10 does not. What does that suggest about what RL changed?
+++
**Short answer: RL sharpened the sampling distribution rather than expanding the set of problems the model can solve.**

- pass@10 asks whether the answer is **anywhere** in ten tries; flat means the set of solvable problems did not grow
- pass@1 asks whether the answer is **on top**; rising means probability mass moved onto answers the model could already produce
- Reporting only pass@1 systematically overstates what RL added
:::

---

:::quiz id="quiz-contamination" title="What Contamination Does to a Score"
What does benchmark contamination do to the meaning of a high score, and why can it never be fully ruled out for a model trained on the web?
+++
**Short answer: it turns the score into a measure of recall rather than generalization, and web-scale training data cannot be exhaustively searched for paraphrases of test items.**

- A test item in training means a correct answer shows **memorization**, not problem-solving
- N-gram checks catch near-exact copies; a question discussed on a forum, translated, or reworded will not match
- Partial defenses: time-based splits, private held-out test sets
- The reliable defense: a test set you wrote yourself and never published
:::

---

:::quiz id="quiz-arena" title="Climbing Without Improving"
Chatbot Arena ranks models by human preference votes. Name a way a model can climb that ranking without becoming more correct.
+++
**Short answer: get longer, better formatted, and more agreeable. Preference is not correctness.**

- Voters reward answers that look thorough and feel accommodating
- More structure, less hedging, more agreement wins votes without changing a single fact
- Module 7's length and sycophancy bias, now in the **measurement** instead of the training signal
- Why AlpacaEval 2.0 explicitly controls for length
:::

---

:::quiz id="quiz-refusal" title="Why Two Numbers, Not One"
Why must refusal rate and false-refusal rate always be reported together?
+++
**Short answer: each one alone is trivially maximized by a degenerate model: refuse everything, or refuse nothing.**

- Refuse everything: perfect refusal rate, useless model
- Refuse nothing: perfect false-refusal rate, unsafe model
- Only the **pair** exposes the trade-off, same structure as precision and recall: one number can always be bought with the other
:::

---

:::quiz id="quiz-image-ablation" title="Did It Look at the Picture?"
A multimodal benchmark reports 60% accuracy. What single ablation would tell you how much of that came from actually looking at the image?
+++
**Short answer: run the same benchmark with the image removed. Whatever score survives came from language priors.**

- A text-only run scoring 45% (chance 25%) means most of the gap above chance came from the **question and options alone**
- Bananas are usually yellow; one option usually reads more plausibly
- The reported 60% is partly a language benchmark wearing a multimodal label
- Dataset fix: balancing, so every answer is equally likely a priori
:::
