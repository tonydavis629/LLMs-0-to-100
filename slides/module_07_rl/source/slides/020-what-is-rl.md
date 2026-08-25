:::divider id="divider-what-is-rl" title="What Reinforcement Learning Is, Really"
:::

---

<!-- .slide: id="rl-vocabulary" -->

## The Classical RL Loop

An **agent** interacts with an **environment** in a loop.

<div class="rl-loop-figure">
<svg viewBox="0 0 880 220" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Agent-environment interaction loop">
<defs>
<marker id="rl-ah" markerWidth="10" markerHeight="8" refX="7.5" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#8892a4"></path></marker>
</defs>
<rect x="70" y="64" width="250" height="92" rx="14" fill="#4a9eff" fill-opacity="0.12" stroke="#4a9eff" stroke-width="2"></rect>
<text x="195" y="103" text-anchor="middle" fill="#e8eaf0" font-size="26" font-weight="700">Agent</text>
<text x="195" y="133" text-anchor="middle" fill="#8892a4" font-size="17">policy &#960;(a | s)</text>
<rect x="560" y="64" width="250" height="92" rx="14" fill="#f5a623" fill-opacity="0.12" stroke="#f5a623" stroke-width="2"></rect>
<text x="685" y="103" text-anchor="middle" fill="#e8eaf0" font-size="26" font-weight="700">Environment</text>
<text x="685" y="133" text-anchor="middle" fill="#8892a4" font-size="17">world / verifier</text>
<path d="M320,96 L552,96" fill="none" stroke="#8892a4" stroke-width="2.5" marker-end="url(#rl-ah)"></path>
<text x="436" y="82" text-anchor="middle" fill="#4a9eff" font-size="19" font-weight="600">action a</text>
<path d="M560,128 L328,128" fill="none" stroke="#8892a4" stroke-width="2.5" marker-end="url(#rl-ah)"></path>
<text x="436" y="152" text-anchor="middle" fill="#f5a623" font-size="19" font-weight="600">reward r &#43; next state s</text>
</svg>
</div>

:::columns cols="2" gap="34px"
- **State** $s$: what the agent observes
- **Action** $a$: what the agent does
- **Reward** $r$: a scalar score for the outcome
- **Policy** $\pi(a \mid s)$: the rule mapping states to actions
+++
- **Trajectory** (episode): a full sequence $s_0, a_0, s_1, a_1, \dots$
- **Return**: total reward over an episode
- Goal: **maximize expected return**
:::

No labeled "correct action." The agent learns only from the **reward** its own choices earn. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="rl-as-llm" -->

## Mapping RL Onto a Language Model

:::columns cols="2" gap="30px"
**Classical RL**

- Policy
- State
- Action
- Episode
- Reward
+++
**Language model**

- The **model** itself, $\pi_\theta$
- The **prompt + tokens so far**
- Emitting the **next token**
- One full **completion**
- A scalar **score on the finished output**
:::

The policy *is* the model. Training the policy *is* updating the weights. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="credit-assignment" -->

## The Defining Difference: Credit Assignment

:::columns cols="2" gap="34px"
**Supervised finetuning (Module 6)**

- Every position has a **known correct token**
- Minimize cross-entropy against it
- Dense, local signal
+++
**Reinforcement learning**

- **No correct token**
- One scalar reward on the **whole** output
- One number must become a per-token gradient
:::

Turning one end-of-sequence reward into an update for every token is the **credit-assignment problem**: the central challenge of this module. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="bandit-vs-mdp" -->

## The Resolution: Collapse to a Bandit

Classical RL is a multi-step **Markov decision process (MDP)**: every action lands in a new state and earns its own reward. LLM post-training **collapses this to a one-step contextual bandit**:

:::columns cols="2" gap="34px"
- The whole completion is a **single action**
- **One terminal reward** at the end
- **No** intermediate rewards; discount factor effectively **1**
+++
- This is why GRPO assigns **one advantage** to the sequence and broadcasts it to **every token**
- Multi-step RL returns later: **process rewards** and **agentic RL**
:::

The credit-assignment problem is sidestepped: no per-token credit at all. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="figure-sutton-barto" -->

:::figure img="images/sutton_barto.jpg" name="Richard Sutton and Andrew Barto" kicker="Reinforcement Learning: An Introduction (1998, 2018); 2024 Turing Award" alt="Richard Sutton and Andrew Barto"
- Wrote the foundational textbook that defined modern RL vocabulary
- Sutton's **"The Bitter Lesson"**: general methods that scale with **search and learning** beat hand-engineered knowledge
- RL post-training is that lesson applied to language models
:::

---

<!-- .slide: id="policy-gradient" -->

## Policy Gradient: REINFORCE

The objective: the policy's **expected reward** over its own samples.

$$J(\theta) = \mathbb{E}_{y \sim \textcolor{#4a9eff}{\pi_\theta(\cdot \mid x)}}\big[ \textcolor{#f5a623}{R(x, y)} \big]$$

- One sampled completion earns an **actual reward** $R(x,y)$
- Training changes a **distribution**, so we optimize the **average** reward over its possible samples

With a baseline $b$, the REINFORCE estimator:

$$\nabla_\theta J(\theta) = \mathbb{E}_{y \sim \pi_\theta}\big[ \textcolor{#50c878}{(R(x,y) - b)} \nabla_\theta \log \textcolor{#4a9eff}{\pi_\theta(y \mid x)} \big]$$

:::columns cols="2" gap="34px"
- $\textcolor{#50c878}{R(x,y) - b}$ is the **advantage**: better or worse than expected
- $\nabla_\theta \log \textcolor{#4a9eff}{\pi_\theta(y)}$: the direction that makes $y$ more likely
- Positive advantage pushes probability **up**; negative pushes it **down**
+++
- The baseline reduces variance **without adding bias**
- PPO learns $b$ with a value network; GRPO uses the group mean
- Same gradient machinery as Module 2g, now doing ascent on reward
:::

---

:::manim id="reinforce-anim" scene="reinforce"
:::

---

<!-- .slide: id="figure-williams" -->

:::figure img="images/williams.jpg" name="Ronald J. Williams" kicker="Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning (REINFORCE, 1992)"
- Introduced the **policy-gradient** estimator used by every method here
- The score-function trick: differentiate $\log \pi_\theta$, weight by reward
- Thirty years later it is the engine of RLHF, GRPO, and reasoning models
:::

---

<!-- .slide: id="two-axes" -->

## Two Axes, Often Conflated

Every method in this module sits on **two orthogonal axes**:

:::columns cols="2" gap="34px"
**Online vs offline** &mdash; *where does the data come from?*

- **Online**: generate fresh data during training (sample, score, update, repeat)
- **Offline**: learn from a **fixed dataset** collected ahead of time, like SFT
+++
**On-policy vs off-policy** &mdash; *who produced the data?*

- **On-policy**: the **current** policy (or one negligibly different)
- **Off-policy**: a **different or older** policy; the update must correct for the mismatch
:::

---

<!-- .slide: id="two-axes-grid" -->

## The Two Axes as a Grid

<div class="axes-grid">
<table>
<thead>
<tr><th></th><th>On-policy</th><th>Off-policy</th></tr>
</thead>
<tbody>
<tr>
<th>Online</th>
<td><strong>REINFORCE, PPO, GRPO</strong><br/>sample from the live policy, score, update</td>
<td>PPO with stale batches<br/>(importance-ratio clipping)</td>
</tr>
<tr>
<th>Offline</th>
<td>rare in practice</td>
<td><strong>DPO</strong> and SFT<br/>a fixed, pre-collected dataset</td>
</tr>
</tbody>
</table>
</div>

The common cases **pair up**. PPO's importance-ratio clipping exists to reuse each batch for a few steps **without drifting off-policy**. <!-- .element: class="text-lg" style="margin-top: 14px;" -->

---

<!-- .slide: id="why-axes-matter" -->

## Why The Axes Matter

:::columns cols="2" gap="34px"
**Online + on-policy**

- Explores **beyond** any fixed dataset
- Can **exceed the demonstrations** and discover reasoning strategies
- Cost: the data distribution **moves** as the model changes; less stable, harder to debug
+++
**Offline + off-policy**

- **Cheaper and more stable**: reuses the SFT machinery exactly
- Cost: can only **rank the responses already in its data**
- No exploration, no exceeding the dataset
:::

---

<!-- .slide: id="exploration-exploitation" -->

## Exploration vs Exploitation

The oldest tension in RL: try **varied** outputs to discover high-reward ones (explore), or lean on **what already works** (exploit).

:::columns cols="2" gap="34px"
- In LLM RL, exploration happens through **sampling**
- The decoding knobs from Module 4f (**temperature, top-k, top-p**) are now **training hyperparameters**
+++
- Too little exploration: the policy never finds better answers
- Too much: the gradient is pure noise
- The sampling distribution **is** the search strategy
:::

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

Entropy collapse returns when we reach reasoning models, behind a surprising result. <!-- .element: class="text-lg" style="margin-top: 12px;" -->
