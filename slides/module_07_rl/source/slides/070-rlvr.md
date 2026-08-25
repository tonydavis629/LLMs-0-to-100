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
- No learned proxy to hack: the verifier is ground truth
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

- The bandit collapse assigns that advantage to **every token**
- Reasoning tokens get pushed up with the answer they led to, though the verifier never read them
:::

Longer chains also buy **compute per problem**: each token is another forward pass, another chance to catch a mistake. So length itself is selected for. In DeepSeek-R1-Zero, response length climbs through training with **no length reward anywhere**. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="r1-zero" -->

## R1-Zero

DeepSeek-R1-Zero applied RL **directly to a base model, no SFT at all**:

- Emergent long chains of thought and self-correction (the **"aha moment"**)
- A capability nobody demonstrated, surfaced by reward alone

Is the model **discovering** new reasoning, or **surfacing latent ability** the reward selects for? The next slides take this apart. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="passk-definition" -->

## A Sharper Lens: pass@k

:::columns cols="2" gap="34px"
**pass@1**

- **One** attempt: is it correct?
- What improves most visibly after RL
+++
**pass@k**

- **k** samples: is **any** correct?
- Measures the **coverage** of the model's distribution, not just its top guess
:::

High pass@1, low pass@k: confident and narrow. Low pass@1, high pass@k: uncertain but broad. Optimizing one can shrink the other. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="yue-counterpoint" -->

## The Critical Counterpoint

Yue et al. (2025), *"Does RL Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?"*

:::columns cols="2" gap="34px"
**The finding**

- RLVR-trained models beat their base model at **small k**
- The base model **catches up and often overtakes at large k**
+++
**The interpretation**

- RLVR **sharpens the sampling distribution** toward paths the base model could **already** produce
- The reasoning boundary can even **narrow** as training proceeds
:::

**RL trades coverage for sampling efficiency.** Optimizing pass@1 shrinks the diversity pass@k rewards: **entropy collapse**. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

:::manim id="passk-anim" scene="passk"
:::

---

<!-- .slide: id="side-quest-r1-debate" -->

## Side Quest: R1-Zero and the Skeptic

:::columns cols="2" gap="34px"
**The R1-Zero claim**

RL on a base model produced **emergent** self-correction and lengthening chains of thought: new ability.
+++
**The Yue et al. rebuttal**

RLVR **amplifies** the base model rather than transcending it. The "new" reasoning was already in the base distribution.
:::

pass@1 and pass@k point in opposite directions because RL **reweights** probability mass; it does not add paths the base model never had. Compare Module 5's "emergence: real or mirage?" <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="agentic-rl" -->

## When the Bandit Becomes an MDP Again

**Recall:** an **MDP** is multi-step RL (each step has state, action, reward); a **bandit** is the one-step case. Single-turn LLM RL collapsed into a bandit. **Agentic RL** brings the MDP back.

:::columns cols="2" gap="34px"
**The loop returns**

- The model takes an action: a **tool call**, a search, a code execution
- It observes a real **intermediate state** (the tool's output) and acts again
+++
**Credit assignment returns**

- Credit spans **many steps**
- Genuine **intermediate rewards**
- The full MDP, not a bandit
:::

Yue et al. point here as the route **past the base-model ceiling** single-turn RLVR hits. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

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
