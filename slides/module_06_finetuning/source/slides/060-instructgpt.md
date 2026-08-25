:::divider id="divider-instructgpt" title="InstructGPT" sub="The birth of the assistant"
:::

---

<!-- .slide: id="instructgpt-recipe" -->

## The Three-Stage Recipe

InstructGPT (Ouyang et al., 2022) turned a base model into an instruction-follower in three stages.

:::columns cols="3" gap="22px"
**1. Supervised finetuning**

- Human-written demonstrations
- **This module**
+++
**2. Reward model**

- Score responses from human **preference rankings**
- Module 7
+++
**3. RL from human feedback**

- Optimize against the reward model with PPO
- Module 7
:::

---

<!-- .slide: id="instructgpt-result" -->

## The Result That Made the Point

Human raters preferred a **1.3B** InstructGPT model over the **175B** GPT-3. More than **100x smaller**.

:::columns cols="2" gap="34px"
- Raw scale did not win. **Alignment to intent** won.
- The smaller model was not more knowledgeable, just more **useful**: it did what was asked.
+++
- ChatGPT is the same SFT-plus-preference recipe, scaled up.
- A capable base model is necessary but not sufficient. **Finetuning** makes it an assistant.
:::

---

<!-- .slide: id="hhh" -->

## The Behavioral Target: HHH

What are we finetuning **toward**? Askell et al.: three properties.

:::columns cols="3" gap="22px"
**Helpful**

Does what the user asks; useful, relevant, complete.
+++
**Honest**

Accurate; does not fabricate; expresses uncertainty.
+++
**Harmless**

Declines to cause harm; safe, non-toxic.
:::

Finetuning turns these into a **data problem**: which demonstrations you show, which behaviors you reward. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="figure-ouyang" -->

:::figure img="images/ouyang.jpg" name="Long Ouyang" kicker="Training Language Models to Follow Instructions with Human Feedback (InstructGPT, 2022)"
- Led InstructGPT (OpenAI): turned a base model into an instruction-follower with the **SFT-plus-RLHF** recipe
- Showed a **1.3B** model could beat **175B** GPT-3 at following intent
- The direct technical ancestor of ChatGPT
:::

---

<!-- .slide: id="figure-askell" -->

:::figure img="images/askell.jpg" name="Amanda Askell" kicker="A General Language Assistant as a Laboratory for Alignment (2021)"
- Articulated the **helpful, honest, and harmless** framing for assistants
- Defined the behavioral target that finetuning and preference optimization aim at
- Made "alignment" a concrete, measurable engineering objective
:::

---

<!-- .slide: id="side-quest-alignment-tax" -->

## Side Quest: The Alignment Tax

**Alignment tax**: finetuning for helpfulness and safety can **reduce raw benchmark scores**.

:::columns cols="2" gap="34px"
- Base capabilities were tuned for next-token prediction, not aligned behavior
- Alignment can trade a few benchmark points for large gains in usefulness and safety
+++
**Mitigations**

- Mix **pretraining data** into the finetuning run
- Use parameter-efficient methods that perturb the base less
- Usually worth paying: a helpful, safe model beats a higher-scoring one nobody can use
:::
