:::divider id="divider-dark-side" title="Failure Modes and Limits" sub="After the learning loop: what can go wrong?"
:::

---

<!-- .slide: id="goodhart" -->

## Goodhart's Law

> When a measure becomes a target, it ceases to be a good measure.
>
> — Marilyn Strathern (1997), summarizing Goodhart’s law

Every reward in this module, **even a verifier**, is a **proxy** for what we actually want.

- Optimize the proxy hard enough and the policy **games it** instead of achieving the real goal
- The better the optimizer, the more dangerous a **misspecified** reward

---

<!-- .slide: id="over-optimization" -->

## Reward Over-Optimization

As the policy's **KL from the reference grows**: **proxy** reward keeps rising, **true** (held-out human) reward eventually **falls**.

:::columns cols="2" gap="34px"
- Gao et al. (2023) fit **scaling laws** to this gap
- Proxy and truth agree at first, then **diverge**
+++
- The **KL penalty** holds the policy near the reference
- Too little leash: proxy-pleasing nonsense
:::

Evaluate quality independently of the training reward; a rising proxy score alone is insufficient. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="reward-hacking-modes" -->

## Concrete Reward-Hacking Failure Modes

:::columns cols="2" gap="34px"
- **Length / verbosity bias**: longer answers score higher, so the model pads
- **Sycophancy**: telling users what they want to hear rather than what is true
+++
- **Format exploitation**: gaming the rubric's surface features
- **Verifier loopholes**: satisfying the checker without solving the task
:::

Even a verifiable reward can be hacked: an output the **checker** accepts may not be what you **meant**. <!-- .element: class="text-lg" style="margin-top: 12px;" -->


---

<!-- .slide: id="entropy-collapse" -->

## The Failure Mode: Entropy Collapse

Training sharpens the policy toward a few high-reward outputs, so the distribution's **entropy falls**.

<div class="rl-loop-figure">
<svg viewBox="0 0 960 120" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Entropy collapse chain">
<defs>
<marker id="ec-ah" markerWidth="10" markerHeight="8" refX="7.5" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#8892a4"></path></marker>
</defs>
<rect x="10" y="30" width="200" height="60" rx="12" fill="#f5a623" fill-opacity="0.12" stroke="#f5a623" stroke-width="1.8"></rect>
<text x="110" y="66" text-anchor="middle" fill="#e8eaf0" font-size="17" font-weight="600">diversity dries up</text>
<path d="M212,60 L248,60" fill="none" stroke="#8892a4" stroke-width="2.2" marker-end="url(#ec-ah)"></path>
<rect x="254" y="30" width="200" height="60" rx="12" fill="#f5a623" fill-opacity="0.12" stroke="#f5a623" stroke-width="1.8"></rect>
<text x="354" y="66" text-anchor="middle" fill="#e8eaf0" font-size="17" font-weight="600">exploration stops</text>
<path d="M456,60 L492,60" fill="none" stroke="#8892a4" stroke-width="2.2" marker-end="url(#ec-ah)"></path>
<rect x="498" y="30" width="200" height="60" rx="12" fill="#f5a623" fill-opacity="0.12" stroke="#f5a623" stroke-width="1.8"></rect>
<text x="598" y="66" text-anchor="middle" fill="#e8eaf0" font-size="17" font-weight="600">narrow behavior</text>
<path d="M700,60 L736,60" fill="none" stroke="#8892a4" stroke-width="2.2" marker-end="url(#ec-ah)"></path>
<rect x="742" y="30" width="200" height="60" rx="12" fill="#ff6b6b" fill-opacity="0.14" stroke="#ff6b6b" stroke-width="2"></rect>
<text x="842" y="66" text-anchor="middle" fill="#e8eaf0" font-size="17" font-weight="600">learning stalls</text>
</svg>
</div>

**Countermeasures**

- An **entropy bonus** in the objective, rewarding spread
- The **KL-to-reference leash** used by RLHF, keeping the policy near the broad base distribution


---

<!-- .slide: id="side-quest-kl-leash" -->

## What the KL Leash Protects Against

The KL penalty keeps the policy near the frozen reference. What goes wrong without it depends on the scorer.

:::columns cols="2" gap="34px"
**Learned reward model (RLHF)**

The reward model only saw text near the SFT distribution, so its scores far from it are unchecked guesses. The optimizer finds regions where those guesses are too high, and the policy drifts into degenerate text that scores well but is not good.
+++
**Exact verifier (RLVR)**

A wrong answer scores 0, so gibberish cannot game it. The risk is collapse instead: correct answers on the rewarded task, less diversity, and nothing protecting behavior the reward never measures.
:::

The leash does not repair a misspecified reward. It limits how far the policy can travel to exploit one. Extra credit: set $\beta = 0$ in the exercise and compare held-out accuracy and diversity. <!-- .element: class="text-lg" style="margin-top: 12px;" -->


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

## <span class="side-quest-badge">Side Quest</span> R1-Zero and the Skeptic

:::columns cols="2" gap="34px"
**The R1-Zero claim**

RL on a base model produced **emergent** self-correction and lengthening chains of thought: new ability.
+++
**The Yue et al. rebuttal**

RLVR **amplifies** the base model rather than transcending it. The "new" reasoning was already in the base distribution.
:::

pass@1 and pass@k point in opposite directions because RL **reweights** probability mass; it does not add paths the base model never had. Compare Module 5's "emergence: real or mirage?" <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="alignment-framing" -->

## RL Is Where Alignment Becomes Explicit

RL makes **helpful, honest, and harmless** (Module 6's HHH target) an explicit **optimization objective**.

- The power: we can directly optimize behavior we want
- The danger: a **misspecified** objective gets **actively optimized** against us

Reward specification **is** value specification. What you reward is what you get, and stating human values as a scalar is hard. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="alignment-tax" -->

## The Alignment Tax

RLHF can **measurably degrade raw capability** on some benchmarks.

- The InstructGPT paper named this tax; it partly mitigated it by **mixing pretraining gradients** back into RL
- Compare pass@k: RL can **narrow the reasoning boundary** even as it improves single-shot behavior

The question is never "tax or no tax" but "is the trade worth it?" Usually it is. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="honest-limits" -->

## Honest Limits

:::columns cols="2" gap="34px"
- Reward models are **imperfect proxies** for messy, contested human values
- Verifiable rewards exist **only for checkable domains**
+++
- RL is **less stable**, more **compute-hungry**, and **harder to debug** than SFT, because training is a **distributed system**, not one loop
- None of this is solved; it is the active frontier
:::

---

<!-- .slide: id="rl-systems" -->

## RL Training Is a Distributed System

Pretraining and SFT run one loop: load a batch, forward, backward, step. RL must also **generate and score** fresh data at every step.

<div class="interaction-row">
<div><p class="example-label">Generators</p><p>An inference engine samples rollouts</p></div>
<p class="interaction-arrow">→</p>
<div><p class="example-label">Scorers</p><p>Verifiers, sandboxes, reward models</p></div>
<p class="interaction-arrow">→</p>
<div><p class="example-label">Trainer</p><p>Updates the policy against the reference</p></div>
<p class="interaction-arrow">→</p>
<div><p class="example-label">Weight sync</p><p>Sends new weights to the generators</p></div>
</div>

:::columns cols="2" gap="34px"
- **Generation dominates**: tokens come out one at a time and rollout lengths vary, so the batch waits on its longest response
- The parts often run on **separate GPU pools** with different software
+++
- **Async pipelines** keep GPUs busy by training on rollouts from slightly older weights, which needs off-policy corrections
- PPO-style RLHF holds **four models**: policy, reference, reward model, critic
:::
