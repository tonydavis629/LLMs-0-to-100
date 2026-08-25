:::divider id="divider-dark-side" title="The Dark Side" sub="Reward hacking, over-optimization, and alignment"
:::

---

<!-- .slide: id="goodhart" -->

## Goodhart's Law

> When a measure becomes a target, it ceases to be a good measure.

Every reward in this module, **even a verifier**, is a **proxy** for what we actually want.

- Optimize the proxy hard enough and the policy **games it** instead of achieving the real goal
- The better the optimizer, the more dangerous a **misspecified** reward

---

<!-- .slide: id="over-optimization" -->

## Reward Over-Optimization, Quantified

As the policy's **KL from the reference grows**: **proxy** reward keeps rising, **true** (held-out human) reward eventually **falls**.

:::columns cols="2" gap="34px"
- Gao et al. (2023) fit **scaling laws** to this gap
- Proxy and truth agree at first, then **diverge**
+++
- The **KL penalty** holds the policy near the reference
- Too little leash: proxy-pleasing nonsense
:::

The exercise reproduces this: set $\beta = 0$. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

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

<!-- .slide: id="rlaif" -->

## RLAIF and Constitutional AI

Bai et al. (2022): replace or augment human feedback with an **AI judge** guided by written principles (a **constitution**).

:::columns cols="2" gap="34px"
- Human labeling does **not scale** to every comparison
- **AI feedback does**: a route toward **scalable oversight**
+++
- The model critiques and revises its own outputs against the principles
- A learned judge for the open-ended tasks no verifier can check
:::

---

<!-- .slide: id="figure-bai" -->

:::figure img="images/anthropic.svg" name="Yuntao Bai and the Anthropic team" kicker="Constitutional AI: Harmlessness from AI Feedback (2022)" alt="Anthropic logo"
- Introduced **RLAIF**: AI feedback guided by written principles
- Showed an assistant could be made harmless with far less human labeling
- A concrete step toward **scalable oversight** of models that may exceed us
:::

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
- RL is **less stable**, more **compute-hungry**, and **harder to debug** than SFT
- None of this is solved; it is the active frontier
:::
