:::divider id="divider-why-rl" title="Why RL for LLMs"
:::

---

<!-- .slide: id="resolve-three-limits" -->

## Resolving the Three Limits of SFT

RL optimizes an **outcome** instead of copying a target:

:::columns cols="3" gap="22px"
**Exceed demonstrations**

A reward can select answers **better** than anything in the data.
+++
**Use negative signal**

Reward **ranks** better against worse, pushing probability **away** from bad answers.
+++
**Close exposure bias**

The model trains on its **own generations**: the same distribution it faces at inference.
:::

---

<!-- .slide: id="eval-easier-than-gen" -->

## Evaluation Is Easier Than Generation

We often **cannot write the ideal target token**, but we **can judge** an output after the fact.

:::columns cols="2" gap="34px"
- Writing a perfect essay is hard; **recognizing** one is easier
- Producing a correct proof is hard; **checking** it is easier
+++
- A **reward** is easier to specify than a full **demonstration**
- This is why RL can reach behaviors SFT cannot
:::

---

<!-- .slide: id="reward-spectrum" -->

## Where Does the Reward Come From?

:::columns cols="2" gap="30px"
- **Human preference comparisons**: RLHF
- **A learned reward model** imitating those preferences
- **A strong LLM scoring outputs directly**: LLM-as-judge, the dominant reward for tasks no program can check
+++
- **An AI judge guided by written principles**: RLAIF
- **A programmatic verifier** for checkable answers: RLVR
:::

The harder the answer is to check by program, the more the reward leans on **judgment**, and the more it can be gamed. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="rejection-sampling" -->

## The Gentlest RL: Rejection Sampling

Also called **Best-of-N**:

:::columns cols="2" gap="34px"
**The loop**

1. Sample $N$ completions
2. Score them with the reward
3. Keep only the **best** (or those above a threshold)
4. Finetune on those with ordinary **cross-entropy**
+++
**Why teach it first**

- No policy gradient, no critic
- **Offline** and **off-policy**
- The core RL move, made concrete: **reward selects among the model's own samples**
- Inference-time Best-of-N needs a **scorer**: verifier, reward model, LLM judge, or task metric
:::

Limitation: rejected samples provide no gradient. Policy gradient can push **down** on losers as well as up on winners. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="reward-cheaper-supervision" -->

## Reward Is Sparser but Cheaper Supervision

:::columns cols="2" gap="34px"
**Demonstrations**

- **Dense**: a target token at every position
- A human writes each ideal response, so the data is expensive and finite
+++
**Reward**

- **Sparse**: one scalar for a whole sequence, which is why credit assignment is hard
- Pairwise comparisons are **cheaper** to collect, and one reward grades **many** sampled outputs
:::

RL trades supervision density for supervision cost. Sampling is the **engine**: decoding is now part of the **training loop**, and exploration depends on sampling diversity. <!-- .element: class="text-lg" style="margin-top: 12px;" -->
