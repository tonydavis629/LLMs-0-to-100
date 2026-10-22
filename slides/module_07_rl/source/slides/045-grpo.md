:::divider id="divider-grpo" title="GRPO: A Baseline From the Group" sub="The algorithm you will build"
:::

---

<!-- .slide: id="grpo-insight" -->

## The GRPO Insight

Shao et al. (DeepSeekMath, 2024): **drop the critic entirely.** Use the **group itself** as the baseline.

<div class="critic-replacement">
<div class="replacement-kept"><p>Keep: <strong>LLM policy → sampled responses → reward scorer → policy update</strong></p></div>
<div class="replacement-row">
<div class="replacement-method"><p>PPO</p></div>
<div class="replacement-source removed-critic"><p class="example-label">Learned critic V<sub>ψ</sub></p><div class="removed-network" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i><i></i></div><p>Predict expected return</p></div>
<p class="replacement-arrow">→</p>
<div class="replacement-result"><p>Baseline = V<sub>ψ</sub>(x)</p><p>A = R − V<sub>ψ</sub>(x)</p></div>
</div>
<p class="replacement-change">↓ Remove critic weights, value loss, and critic optimizer state</p>
<div class="replacement-row">
<div class="replacement-method"><p>GRPO</p></div>
<div class="replacement-source"><p class="example-label">Score a group for one prompt</p><div class="reward-chips"><p class="positive">1</p><p class="negative">0</p><p class="positive">1</p><p class="negative">0</p></div><p>Mean = 0.5 · population std = 0.5</p></div>
<p class="replacement-arrow">→</p>
<div class="replacement-result"><p>Baseline = group mean</p><p>A = [+1, −1, +1, −1]</p></div>
</div>
</div>

The baseline and scale come from **group statistics**. No critic is trained.

---

<!-- .slide: id="grpo-baseline" -->

## Replace the Critic With Group Statistics

The baseline comes from comparing several sampled responses to the same prompt.

$$A_i = \frac{R_i - \textcolor{#f5a623}{\operatorname{mean}(R_1, \dots, R_G)}}{\operatorname{std}(R_1, \dots, R_G) + \epsilon}$$

:::columns cols="2" gap="34px"
**PPO**

Baseline = a **learned value estimate**, trained alongside the policy.
+++
**GRPO**

Baseline = the **group mean reward**, computed from samples you already drew.
:::

Full GRPO uses PPO-style ratio clipping and a KL penalty to a frozen reference, with **no value network**. A completion is rewarded only for beating its **peers** on the same prompt. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

:::interactive id="grpo-group" widget="grpoGroup" title="Advantages From the Group Alone"
:::

---

<!-- .slide: id="grpo-anim" -->

## GRPO: Follow the Same Training Batch

<div class="training-loop" data-algorithm="grpo"></div>

---

<!-- .slide: id="grpo-why-build" -->

## Why Build GRPO

:::columns cols="2" gap="34px"
- No critic to train or tune
+++
- It makes the **sample &rarr; score &rarr; update** cycle visible
- It is the method behind **DeepSeek-R1**
:::

GRPO collects fresh responses, compares their rewards within each prompt, and updates the LLM without a learned critic. The exercise uses one policy-gradient update per fresh batch; full GRPO also supports clipped batch reuse. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="figure-deepseek" -->

:::figure img="images/deepseek.png" name="The DeepSeek team (Zhihong Shao and collaborators)" kicker="GRPO (DeepSeekMath, 2024) and DeepSeek-R1 (2025)" alt="DeepSeek logo"
- Introduced **GRPO**: value-free policy optimization using group statistics
- Used it with **verifiable rewards** to train an open reasoning model rivaling closed ones
- Made the open-source reasoning recipe concrete and reproducible
:::
