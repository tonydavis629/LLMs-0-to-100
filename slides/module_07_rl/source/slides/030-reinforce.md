:::divider id="divider-policy-optimization" title="Turning Reward Into Learning" sub="REINFORCE (1992) → PPO (2017) → GRPO (2024)"
:::

---

<!-- .slide: id="expected-reward" -->

## Improve the Average Reward

The policy is a distribution over responses, so the objective is the **average** reward of the responses it samples:

$$J(\theta) = \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)}[R(x,y)]$$

:::columns cols="2" gap="34px"
**Goal**

Change the weights $\theta$ so that future samples earn higher reward on average.
+++
**Problem**

Gradient descent needs $\nabla_\theta J$. But $R$ comes from sampling discrete tokens and running a checker such as a Python test, and neither can be backpropagated through.
:::

REINFORCE gets $\nabla_\theta J$ anyway, using only sampled rewards and the gradient of $\log \pi_\theta$. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="reward-baseline" -->

## Better or Worse Than Expected?

A **baseline** $b$ gives a reference score. Subtract it to get an **advantage**:

$$A = R - b$$

<div class="example-pair">
<div><p class="example-label">Better than baseline</p><p>R = 1, b = 0.5</p><p class="positive">A = +0.5 → increase likelihood</p></div>
<div><p class="example-label">Worse than baseline</p><p>R = 0, b = 0.5</p><p class="negative">A = −0.5 → decrease likelihood</p></div>
</div>

For REINFORCE, a fixed baseline or a running estimate from earlier samples can reduce gradient noise.

---

<!-- .slide: id="policy-gradient" -->

## REINFORCE: Weight Log-Probability by Advantage

Williams (1992): turn sampled rewards into a gradient of expected reward.

$$\nabla_\theta J(\theta) = \mathbb{E}_{y \sim \pi_\theta}\left[\textcolor{#f5a623}{(R-b)}\,\nabla_\theta \log \pi_\theta(y \mid x)\right]$$

Use gradient descent on the sampled loss:

$$\mathcal{L} = -\textcolor{#f5a623}{A}\sum_t \log \pi_\theta(y_t \mid x,y_{<t})$$

Treat **$A$ as a fixed weight**. Backpropagate through the LLM's log-probabilities, then let the optimizer update its parameters.

---

<!-- .slide: id="reinforce-anim" -->

## REINFORCE: Follow One Training Batch

<div class="training-loop" data-algorithm="reinforce"></div>

---

<!-- .slide: id="figure-williams" -->

:::figure img="images/williams.jpg" name="Ronald J. Williams" kicker="Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning (REINFORCE, 1992)"
- Introduced the REINFORCE family of gradient estimators
- The score supplies a weight; the policy supplies a differentiable log-probability
- This policy-gradient idea underlies the algorithms that follow
:::
