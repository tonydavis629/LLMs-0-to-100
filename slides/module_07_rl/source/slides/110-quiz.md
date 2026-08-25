:::divider id="divider-quiz" title="Quiz"
:::

---

:::quiz id="quiz-exceed-sft" title="Why Can RL Beat SFT?"
Why can RL improve on a task that SFT learned only imperfectly, when **both start from the same model**?
+++
**Short answer: RL optimizes an outcome and can push probability *down* on bad answers; SFT can only copy fixed target tokens.**

- SFT copies fixed target tokens; it cannot express that one answer beats another
- RL samples the model's **own** completions, scores them, and pushes probability **up** on winners and **down** on losers
- Negative signal plus search beyond the demonstrations reinforces answers SFT never reliably produced
- No new labeled data needed, just a reward
:::

---

:::quiz id="quiz-grpo-baseline" title="The GRPO Baseline"
You sample a group of four completions for one prompt and the verifier returns rewards $[1, 1, 0, 0]$. GRPO sets the baseline to the **group mean**. What advantage does each completion receive, and why does subtracting this baseline **reduce variance without biasing** the gradient?
+++
**Short answer: the advantages are $[+0.5, +0.5, -0.5, -0.5]$; a baseline independent of the sampled completion is unbiased, and recentering shrinks the update magnitude.**

- Group mean $= 0.5$, so $A_i = R_i - 0.5 = [+0.5, +0.5, -0.5, -0.5]$, then normalized by the group std
- Any baseline independent of the sampled completion leaves the gradient's **expectation unchanged**, because $\mathbb{E}[\nabla_\theta \log \pi_\theta] = 0$
- Recentering weights better-than-average completions positive and worse-than-average negative, shrinking update magnitude and cutting variance
:::

---

:::quiz id="quiz-why-group" title="Why a Group?"
Why sample a **group** of completions per prompt instead of just one?
+++
**Short answer: one sample has no reference point; a group supplies the baseline (its mean) and the scale (its spread), and guarantees a within-group learning signal.**

- One sample gives a reward but no reference: is that reward good or bad for this prompt?
- No baseline to subtract means a high-variance gradient
- A group lets completions **compete**: the mean is the baseline, the spread sets the scale
- When some succeed and others fail, the within-group variance is exactly the signal that says which behaviors to raise and which to lower
:::

---

:::quiz id="quiz-kl-leash" title="The KL Leash"
What is the KL-to-reference penalty protecting against, and what happens to the completions if you remove it?
+++
**Short answer: it protects against reward over-optimization; remove it and the policy drifts into high-reward gibberish that games the proxy.**

- The reward is only a proxy; a strong optimizer drifts far from the reference to find outputs that score highly but are not what we want
- The KL term penalizes moving away from the frozen reference, keeping the policy on fluent, sensible text
- Set $\beta = 0$ and the policy drifts into **degenerate, high-reward gibberish** that games the verifier or reward model
- The exercise extra credit shows exactly this
:::

---

:::quiz id="quiz-verifier-hack" title="Hacking a Verifier"
Why is a **verifiable** reward harder to hack than a learned reward model, and what can **still** be hacked?
+++
**Short answer: a verifier is deterministic ground truth with no learned proxy to fool, but the gap between what the checker accepts and what you meant can still be exploited.**

- A learned reward model is a neural network with blind spots; hard optimization finds inputs that light up the proxy without being good
- A verifier is **deterministic ground truth** for a checkable answer: no learned proxy to fool
- Still hackable: the **gap between the check and the intent** (passing visible tests without solving the problem, exploiting a parser)
- And verifiable rewards only exist for domains you can actually check
:::

---

:::quiz id="quiz-no-critic" title="No Critic"
How does GRPO avoid the separate **value network** that PPO requires?
+++
**Short answer: it replaces PPO's learned value network with an empirical baseline &mdash; the mean reward of a sampled group.**

- PPO's critic estimates the baseline (expected reward from a state) for computing advantages; it is roughly policy-sized and trained alongside
- GRPO replaces the learned estimate with an **empirical** one: the mean (and std) of a sampled group for the same prompt
- The baseline comes from samples you already drew: no value network to build, train, or tune
- That cuts memory and a major source of instability
:::

---

:::quiz id="quiz-instability" title="Why RL Is Harder Than SFT"
Why is RL training less stable and harder to debug than the supervised finetuning of Module 6?
+++
**Short answer: RL is online and on-policy, so the model generates its own training data and the data distribution moves as the weights change &mdash; there is no fixed loss curve to watch.**

- SFT: fixed data, stable cross-entropy against known labels
- RL is **online and on-policy**: the model generates its own training data, so the **distribution moves** as the weights change
- The reward is sparse and noisy; the gradient is high-variance; temperature, group size, KL weight, and learning rate all interact
- A bad setting triggers entropy collapse or reward hacking that surfaces only after many steps; you debug a moving target
:::

---

:::quiz id="quiz-passk" title="pass@1 Rises, pass@k Does Not"
A model's pass@1 rises after GRPO but its pass@k at large k does **not**. What does this tell you about whether RL created new ability or reweighted existing ability, and how does it connect to **entropy collapse**?
+++
**Short answer: RL reweighted existing ability rather than creating new ability; entropy collapse raises pass@1 while shrinking the diversity that large-k pass@k depends on.**

- Flat pass@k at large k means the set of problems solvable **with enough tries** has not grown
- RL **reweighted** probability toward paths the base model could already sample; one sample now lands on them more often
- The mechanism is entropy collapse: concentrating mass on a few high-reward outputs raises pass@1 but **shrinks the diversity** large-k pass@k depends on
- The Yue et al. (2025) finding: RL trades coverage for sampling efficiency
:::

---

:::quiz id="quiz-rejection-vs-grpo" title="What Rejection Sampling Throws Away"
Rejection-sampling finetuning and GRPO both use reward to improve the model. What information does GRPO use that rejection sampling **throws away**?
+++
**Short answer: GRPO uses the negative signal &mdash; pushing probability *down* on worse-than-average completions &mdash; which rejection sampling throws away along with all the losers.**

- Rejection sampling keeps only the **winners** and finetunes on them with cross-entropy; everything about the rejected samples is discarded
- GRPO uses the **full reward vector**: each completion's advantage relative to the group
- It pushes probability **up on better-than-average** completions **and down on worse-than-average** ones
- That **negative signal** is what rejection sampling cannot express: it never trains on the losers at all
:::

---

:::quiz id="quiz-bandit-mdp" title="Bandit vs MDP"
Why is single-turn LLM RL usually framed as a one-step **bandit** rather than a multi-step **MDP**, and what changes when the model can call **tools** mid-generation?
+++
**Short answer: single-turn generation is one action earning one terminal reward (a bandit); tool calls add real intermediate states and rewards, making it a multi-step MDP.**

- Single-turn generation: **one action**, **one terminal reward**, no intermediate states from the environment (the model just samples its own tokens)
- That is a contextual bandit, which is why GRPO assigns one advantage to the whole sequence
- A **tool call** produces a genuine **intermediate state** (the tool's output) the model must react to, plus possible **intermediate rewards**
- The problem becomes a real multi-step **MDP**: harder credit assignment, and the agentic frontier beyond single-turn RLVR
:::
