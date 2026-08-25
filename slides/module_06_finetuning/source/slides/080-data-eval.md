:::divider id="divider-craft" title="The Craft" sub="Data and evaluation"
:::

---

<!-- .slide: id="data-quality" -->

## Data Quality Dominates Quantity

A small, clean, diverse set beats a large noisy one.

:::columns cols="2" gap="34px"
**LIMA** (Zhou et al., 2023)

- "Less Is More for Alignment"
- Roughly **1,000** curated examples
- Still produced a strong assistant
+++
**Why so few works**

- The base model already has the knowledge
- Alignment teaches **format and style**; a small clean set conveys that
- Noise teaches contradictions
:::

The expensive part of SFT is not compute. It is **curating good demonstrations**. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="data-sources" -->

## Where SFT Data Comes From

:::columns cols="3" gap="22px"
**Human-written**

- People write good responses
- Highest quality, slowest, most expensive
+++
**Distillation**

- Generate data from a **stronger model**
- Self-Instruct, Alpaca
+++
**Filtered logs**

- Mine real conversations, keep the good ones
- Abundant; needs filtering and consent
:::

Modern pipelines mix all three. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="sft-hyperparams" -->

## Hyperparameters That Matter

SFT only **nudges** a capable model. The settings are deliberately gentle.

:::columns cols="2" gap="34px"
**Move gently**

- **Small** learning rate: often 10x below pretraining
- **1 to 3 epochs**; more overfits and forgets
+++
**Watch generalization**

- Keep a **held-out set**: small data memorizes fast, training loss lies
- Stop when held-out quality stops improving
:::

The failure mode is always **catastrophic forgetting**: too large a step, or too many passes. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="sft-evaluation" -->

## Evaluating a Finetuned Model

Pretraining had a clean validation loss. **"Did it become a better assistant?"** has no single number.

:::columns cols="3" gap="22px"
**Preference judgments**

- Two responses, same prompt: which is better?
- Comparative, not absolute
+++
**Downstream benchmarks**

- MMLU, GSM8K
- Objective, repeatable
- Miss style, helpfulness, tone
+++
**Human or model graders**

- Rate quality directly
- LLM-as-judge: cheap, scalable, imperfect
:::

Every option is partial and noisy. (How to evaluate: Module 11.) <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="side-quest-superficial-alignment" -->

## Side Quest: The Superficial Alignment Hypothesis

LIMA's claim: the base model **already contains** the knowledge; alignment teaches **format, style, and which behaviors to surface**.

:::columns cols="2" gap="34px"
- If true, finetuning **reveals** capability more than it teaches
- It would explain why ~1,000 examples can align a model that saw trillions of tokens
+++
**Critical-thinking question** (compare Module 5's "emergence: real or mirage?"):

If 1,000 examples align a model, how much is finetuning **teaching** vs **unlocking**?
:::

---

<!-- .slide: id="side-quest-model-collapse" -->

## Side Quest: Synthetic Data and Model Collapse

Alpaca and Self-Instruct generate finetuning data from a **stronger model**. Cheap and scalable, with hard questions:

:::columns cols="2" gap="34px"
- **Quality and licensing**: the teacher's errors propagate; proprietary-model outputs may violate terms
- **Model collapse**: repeated training on model-generated text erodes diversity
+++
- Ties to Module 5's **data-wall** side quest: scarce human data pushes models toward synthetic text
- Open question: how much synthetic data before the snake eats its own tail?
:::
