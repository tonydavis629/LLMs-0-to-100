:::divider id="divider-grpo" title="Value-Free RL and GRPO" sub="The algorithm you will build"
:::

---

<!-- .slide: id="ppo-cost" -->

## The Cost of PPO

:::columns cols="2" gap="34px"
- The **critic** (value network) is roughly the **size of the policy**
- Doubles memory; a second model to train and tune
+++
- A bad value estimate destabilizes the whole run
- Can we get the baseline **without** a learned value network?
:::

---

<!-- .slide: id="grpo-insight" -->

## The GRPO Insight

Shao et al. (DeepSeekMath, 2024): **drop the critic entirely.** Use the **group itself** as the baseline.

:::columns cols="2" gap="34px"
**The loop**

1. For each prompt, sample a **group** of $G$ completions
2. Score them all
3. The group's $\textcolor{#4a9eff}{\operatorname{mean}}$ reward is the baseline
+++
**The advantage**

Each completion's advantage is its reward minus the group $\textcolor{#4a9eff}{\operatorname{mean}}$, normalized by the group $\textcolor{#50c878}{\operatorname{std}}$.
:::

$$A_i = \frac{\textcolor{#f5a623}{R_i} - \textcolor{#4a9eff}{\operatorname{mean}(R_1, \dots, R_G)}}{\textcolor{#50c878}{\operatorname{std}(R_1, \dots, R_G)}}$$

---

<!-- .slide: id="grpo-baseline" -->

## The Baseline Comes For Free

The same variance-reduction idea as the learned baseline, at **no extra cost**.

:::columns cols="2" gap="34px"
**PPO**

Baseline = a **learned value network**, as big as the policy, trained alongside it.
+++
**GRPO**

Baseline = the **group mean reward**, computed from samples you already drew.
:::

The update keeps PPO's ratio clipping and KL penalty, with **no value network**. A completion is rewarded only for beating its **peers** on the same prompt. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

:::interactive id="grpo-group" widget="grpoGroup" title="Advantages From the Group Alone"
:::

---

:::manim id="grpo-anim" scene="grpo"
:::

---

<!-- .slide: id="grpo-why-build" -->

## Why Build GRPO

:::columns cols="2" gap="34px"
- The implementation is **much simpler** than PPO
+++
- It makes the **sample &rarr; score &rarr; update** cycle visible
- It is the method behind **DeepSeek-R1**
:::

GRPO is **online and on-policy** (fresh samples from the live policy, unlike DPO) yet **value-free** (no critic, unlike PPO). That combination put it at the center of reasoning-model training. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="figure-deepseek" -->

:::figure img="images/deepseek.png" name="The DeepSeek team (Zhihong Shao and collaborators)" kicker="GRPO (DeepSeekMath, 2024) and DeepSeek-R1 (2025)" alt="DeepSeek logo"
- Introduced **GRPO**: value-free policy optimization using group statistics
- Used it with **verifiable rewards** to train an open reasoning model rivaling closed ones
- Made the open-source reasoning recipe concrete and reproducible
:::

---

<!-- .slide: id="dpo-ppo-grpo" -->

## DPO vs PPO vs GRPO

Two differences: **where the baseline comes from**, and whether the algorithm samples **online**.

<div class="axes-grid">
<table>
<thead>
<tr><th></th><th>On-policy</th><th>Off-policy</th></tr>
</thead>
<tbody>
<tr>
<th>Online</th>
<td><strong>PPO</strong> &mdash; critic baseline, expensive<br/><strong>GRPO</strong> &mdash; group baseline, value-free</td>
<td>(stale-batch PPO)</td>
</tr>
<tr>
<th>Offline</th>
<td>&mdash;</td>
<td><strong>DPO</strong> &mdash; no sampling, cheapest</td>
</tr>
</tbody>
</table>
</div>

DPO: offline, off-policy, cheapest. PPO: online, on-policy, critic. GRPO: online, on-policy, **value-free**. <!-- .element: class="text-lg" style="margin-top: 14px;" -->
