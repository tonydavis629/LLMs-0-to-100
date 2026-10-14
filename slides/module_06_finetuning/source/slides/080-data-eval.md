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
**Synthetic**

- **Distillation**: a stronger model writes the data (Alpaca)
- **Self-instruction**: the model writes its own (Self-Instruct)
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

## <span class="side-quest-badge">Side Quest</span> The Superficial Alignment Hypothesis

LIMA's claim: the base model **already contains** the knowledge; alignment teaches **format, style, and which behaviors to surface**.

:::columns cols="2" gap="34px"
- If true, finetuning **reveals** capability more than it teaches
- It would explain why ~1,000 examples can align a model that saw trillions of tokens
+++
If 1,000 examples align a model, how much is finetuning **teaching** vs **unlocking**?
:::

---

<!-- .slide: id="side-quest-model-collapse" -->

## <span class="side-quest-badge">Side Quest</span> Two Kinds of Synthetic Data

:::columns cols="2" gap="34px"
**Distillation** (Alpaca, 2023)

- A **stronger teacher** (text-davinci-003) wrote 52K demonstrations; a **smaller student** (LLaMA 7B) imitated them
- The student copies the teacher's style more readily than its facts (Gudibande et al., 2023)
- The teacher's errors and terms of use come along with the data
+++
**Self-instruction** (Self-Instruct, 2022)

- The **same model** (GPT-3) wrote new tasks and answers from 175 seeds, filtered them, then trained on them
- No stronger model needed: the model bootstraps itself
- Loop it and you risk **model collapse**: each generation loses more of the rare outputs (Shumailov et al., 2024)
:::

Module 5's data wall pushes labs toward both. How much synthetic text can a model train on before the loop starts to degrade it? <!-- .element: class="text-lg" style="margin-top: 12px;" -->

