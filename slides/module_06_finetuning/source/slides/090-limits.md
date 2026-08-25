:::divider id="divider-limits" title="The Limits of SFT" sub="And what reinforcement learning adds"
:::

---

<!-- .slide: id="sft-only-imitates" -->

## SFT Can Only Imitate

SFT learns **"produce responses like these"**, not **"this response is better than that one."**

:::columns cols="2" gap="34px"
**No negative signal**

- Every demonstration is a positive example
- Nothing says "not that"
+++
**Exposure bias**

- Trains on **ground-truth prefixes**
- At generation time it continues its **own** imperfect outputs
- Small early mistakes compound
:::

SFT's ceiling is the quality of its demonstrations. Going higher needs **comparisons**. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="handoff-module-7" -->

## Next Class: Reinforcement Learning

**Preference optimization** and **reinforcement learning** address these limits.

:::columns cols="3" gap="22px"
**RLHF**

- Reward model from human preferences, then optimize against it
- InstructGPT stages 2 and 3
+++
**DPO**

- Learn from preference pairs
- No separate reward model
+++
**GRPO**

- Group-relative policy optimization
- Used in recent reasoning models
:::

**SFT shaped the format; RL shapes the preferences.** Same optimizer, same backprop, new data and target. <!-- .element: class="text-lg" style="margin-top: 12px;" -->
