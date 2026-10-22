:::divider id="divider-rlvr" title="RLVR and the Reasoning Revolution"
:::

---

<!-- .slide: id="rlvr-definition" -->

## Reinforcement Learning from Verifiable Rewards

When an answer is **checkable**, the reward is a **deterministic program**, not a learned model.

:::columns cols="2" gap="34px"
- Math problems have **known answers**
- Code can be **run against tests**
- Formatted outputs can be **parsed**
+++
- No human preferences, no reward model
- The program defines the score; its checks must match the task
:::

The exercise uses exactly this: a Python function that reverses the string and compares. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="rlvr-breakthrough" -->

## The Breakthrough

:::columns cols="2" gap="34px"
**OpenAI o1** (September 2024)

- RL on chain-of-thought
- More **inference-time compute**
- Sharply better reasoning
+++
**DeepSeek-R1** (January 2025)

- Reproduced it **openly**
- **GRPO** + **verifiable rewards**: the algorithm you build in the exercise
:::

The scaling story shifts: from **pretraining** compute (Module 5g) to **RL and inference-time** compute. The model thinks longer before answering. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="reward-is-curriculum" -->

## The Reward Is the Curriculum

With a verifiable signal, the model **self-improves by exploration**, graded only by whether the final answer checks out.

:::columns cols="2" gap="34px"
- No one demonstrates the reasoning
- The reward selects whatever **chain of thought happens to work**
+++
- The **"exceed the demonstrations"** promise, realized
- The exercise shows it in miniature: reward climbs as the policy discovers reliable reversals
:::

---

<!-- .slide: id="why-cot-grows" -->

## Why Reward Alone Grows a Chain of Thought

Nothing in the reward says "reason step by step". It grades only the **final answer**. Selection grows the chain anyway:

:::columns cols="2" gap="34px"
**Selection**

- Some sampled completions work through intermediate steps
- Those are **correct more often**, so they earn positive advantage more often
+++
**Broadcast**

- The response-level loss applies one advantage to **every generated token**
- Reasoning tokens get pushed up with the answer they led to, though the verifier never read them
:::

Longer chains also buy **compute per problem**: each token is another forward pass, another chance to catch a mistake. Useful intermediate computation can therefore be reinforced. In DeepSeek-R1-Zero, response length climbs through training with **no length reward anywhere**. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="r1-zero" -->

## R1-Zero

DeepSeek-R1-Zero applied RL **directly to a base model, no SFT at all**:

- Emergent long chains of thought and self-correction (the **"aha moment"**)
- A capability nobody demonstrated, surfaced by reward alone

Is the model **discovering** new reasoning, or **surfacing latent ability** the reward selects for? The next slides take this apart. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="process-reward" -->

## Process vs Outcome Rewards

Reward only the **final answer** (outcome), or every **reasoning step** (process)?

:::columns cols="2" gap="34px"
**Outcome reward**

- One scalar at the end
- Cheap
- Credit assignment is hard: which step mattered?
+++
**Process reward**

- A **process reward model** grades the chain step by step
- Dense signal, easier credit assignment
- Far more expensive to label
:::

The credit-assignment trade-off again: denser reward, easier learning, higher labeling cost. <!-- .element: class="text-lg" style="margin-top: 12px;" -->
