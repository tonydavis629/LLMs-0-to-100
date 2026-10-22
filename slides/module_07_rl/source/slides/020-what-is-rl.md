:::divider id="divider-what-is-rl" title="Learning From Reward"
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

<!-- .slide: id="figure-sutton-barto" -->

:::figure img="images/sutton_barto.jpg" name="Richard Sutton and Andrew Barto" kicker="Reinforcement Learning: An Introduction (1998, 2018); 2024 Turing Award" alt="Richard Sutton and Andrew Barto"
- Established the vocabulary of agents, policies, rewards, and returns
- RL studies how choices improve through interaction and feedback
- We will apply that framework to a language model
:::

---

<!-- .slide: id="rl-as-llm" -->

## The Policy Is the LLM

<div class="interaction-row">
<div><p class="example-label">Input</p><p>Prompt x</p></div>
<p class="interaction-arrow">→</p>
<div><p class="example-label">Policy π<sub>θ</sub></p><p>LLM with weights θ</p></div>
<p class="interaction-arrow">→</p>
<div><p class="example-label">Output</p><p>Completion y</p></div>
</div>

$$y \sim \pi_\theta(\cdot \mid x)$$

The policy is a **distribution over possible completions**. Updating the policy means updating the LLM's weights.

---

<!-- .slide: id="what-is-bandit" -->

## Bandits: Choose, Get Paid, Repeat

The name comes from slot machines, once called **one-armed bandits**. Facing a row of machines with unknown payout rates, you pull an arm, see what it pays, and choose again.

:::columns cols="3" gap="25px"
**Multi-armed bandit**

- No state: the same arms every round
- Learn which arm pays best
+++
**Contextual bandit**

- See a **context** first, then act
- The best action depends on the context
- Your action does **not** affect the next context
+++
**MDP**

- Your action **changes the next state**
- Decisions chain into an episode, so credit spreads across steps
:::

Every bandit faces the same tension: keep pulling the arm that looks best (**exploit**), or try others that might pay more (**explore**). <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="bandit-vs-mdp" -->

## One Prompt, One Scored Response

For now, each training example has one prompt, one complete response, and one score.

<div class="interaction-row">
<div><p class="example-label">Context</p><p>reverse: cat</p></div>
<p class="interaction-arrow">→</p>
<div><p class="example-label">One response</p><p>tac</p></div>
<p class="interaction-arrow">→</p>
<div><p class="example-label">Reward</p><p>1</p></div>
</div>

The prompt is the context and the whole response is one action. The next prompt comes from the dataset no matter what the model wrote, so this is a **contextual bandit**, even though the LLM generates the response **one token at a time**.

---

<!-- .slide: id="credit-assignment" -->

## One Score Must Train Many Tokens

The checker scores the completed response. It does not label which generated tokens helped.

$$\log \pi_\theta(y \mid x) = \sum_t \log \pi_\theta(y_t \mid x, y_{<t})$$

A response-level learning signal can weight **every generated token's log-probability**.

:::note
This gives a trainable objective. It does not tell us which individual token caused success or failure.
:::

---

<!-- .slide: id="agentic-rl" -->

## Tools Add Decisions Between Observations

<div class="interaction-row">
<div><p class="example-label">LLM action</p><p>Run some code</p></div>
<p class="interaction-arrow">→</p>
<div><p class="example-label">Tool observation</p><p>Read the error</p></div>
<p class="interaction-arrow">→</p>
<div><p class="example-label">LLM action</p><p>Revise the code</p></div>
</div>

The next decision depends on what the environment returns. We model this as a **multi-step decision process (MDP)**, using the relevant interaction history as the state.

It can still receive **only a final reward**. Multiple decisions do not require intermediate rewards.

---

<!-- .slide: id="online-sampling" -->

## RL Collects Fresh Attempts During Training

<div class="interaction-row">
<div><p class="example-label">Sample</p><p>Current LLM</p></div>
<p class="interaction-arrow">→</p>
<div><p class="example-label">Score</p><p>Reward function</p></div>
<p class="interaction-arrow">→</p>
<div><p class="example-label">Update</p><p>LLM weights</p></div>
</div>

Then sample again from the updated LLM. This is **online data collection**.

**Offline training** reuses a fixed dataset. Standard SFT and the original DPO recipe follow that pattern.

---

<!-- .slide: id="on-policy-sampling" -->

## After an Update, Old Samples Become Stale

**On-policy:** the samples match the policy used in the learning objective.

Changing the weights changes the distribution. Samples from an older or different policy create a **policy mismatch**.

:::note
The methods here collect fresh samples online to stay close to on-policy. PPO and GRPO track the sampling policy when reusing a batch for several updates.
:::

Online collection and policy matching are **related properties**. An online system can also reuse old, off-policy data.

---

<!-- .slide: id="exploration-exploitation" -->

## Sampling Supplies Exploration

The model must try varied responses to find ones that earn higher reward.

:::columns cols="2" gap="34px"
**Explore**

Sample alternatives, including less likely answers.
+++
**Exploit**

Make answers that already score well more likely.
:::

Temperature and other decoding choices now affect **the training data the model discovers**.
