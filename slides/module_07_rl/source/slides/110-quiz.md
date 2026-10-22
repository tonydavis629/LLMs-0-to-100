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

:::quiz id="quiz-grpo-baseline" title="The Reward Did Not Change"
An answer earns reward **1** in two different groups. In one group every answer earns 1; in the other, most earn 0. Should GRPO give that answer the **same learning signal** in both groups?
+++
**No. Its advantage depends on how it compares with the other attempts at the same prompt.**

- In the all-correct group, it ties the mean: its advantage is **zero**.
- In the mixed group, it beats the mean: its advantage is **positive**.
- The checker supplies an absolute score; GRPO turns it into a **relative training signal**.
:::

---

:::quiz id="quiz-why-group" title="Why a Group?"
Why sample a **group** of completions per prompt instead of just one?
+++
**Short answer: a group supplies a reference point and a scale when its rewards differ.**

- One sample gives a reward but no reference: is that reward good or bad for this prompt?
- No baseline to subtract means a high-variance gradient
- A group lets completions **compete**: the mean is the baseline, the spread sets the scale
- When some succeed and others fail, the within-group variance is exactly the signal that says which behaviors to raise and which to lower
- When every reward ties, every relative advantage is zero
:::

---

:::quiz id="quiz-kl-leash" title="The KL Leash"
What is the KL-to-reference penalty protecting against? Does removing it guarantee that training will fail?
+++
**It discourages excessive drift from the reference. Removing it changes the trade-off; failure is not guaranteed.**

- The reward is only a proxy; a strong optimizer drifts far from the reference to find outputs that score highly but are not what we want
- The KL term penalizes moving away from the frozen reference, keeping the policy on fluent, sensible text
- With $\beta = 0$, the reward and other training choices determine how far the policy moves
- Measure held-out quality and diversity; a rising reward alone does not establish improvement
:::

---

:::quiz id="quiz-verifier-hack" title="Hacking a Verifier"
Why is a **verifiable** reward harder to hack than a learned reward model, and what can **still** be hacked?
+++
**A verifier implements explicit checks, but those checks can leave gaps between acceptance and the intended task.**

- A learned reward model is a neural network with blind spots; hard optimization finds inputs that light up the proxy without being good
- A verifier applies specified tests or comparisons: there is no learned reward model to fool
- Still hackable: the **gap between the check and the intent** (passing visible tests without solving the problem, exploiting a parser)
- And verifiable rewards only exist for domains you can actually check
:::

---

:::quiz id="quiz-no-critic" title="No Critic"
How does GRPO avoid the separate **value network** that PPO requires?
+++
**Short answer: it replaces PPO's learned value network with an empirical baseline &mdash; the mean reward of a sampled group.**

- PPO's critic estimates expected return for computing advantages and is trained alongside the policy
- GRPO replaces the learned estimate with an **empirical** one: the mean (and std) of a sampled group for the same prompt
- The baseline comes from samples you already drew: no value network to build, train, or tune
- That cuts memory and a major source of instability
:::

---

:::quiz id="quiz-instability" title="Why RL Is Harder Than SFT"
Why is RL training less stable and harder to debug than the supervised finetuning of Module 6?
+++
**The online RL methods here generate their own training data, so the sample distribution changes with the weights.**

- SFT: fixed data, stable cross-entropy against known labels
- Fresh data follows the changing policy, so reward and loss must be interpreted alongside held-out evaluation
- The reward is sparse and noisy; the gradient is high-variance; temperature, group size, KL weight, and learning rate all interact
- A bad setting triggers entropy collapse or reward hacking that surfaces only after many steps; you debug a moving target
:::

---

:::quiz id="quiz-passk" title="pass@1 Rises, pass@k Does Not"
A model's pass@1 rises after GRPO but its pass@k at large k does **not**. What does this tell you about whether RL created new ability or reweighted existing ability, and how does it connect to **entropy collapse**?
+++
**This is evidence of more reliable sampling without broader coverage. It does not establish that RL can never learn new abilities.**

- Flat pass@k at large k means the set of problems solvable **with enough tries** has not grown
- RL **reweighted** probability toward paths the base model could already sample; one sample now lands on them more often
- One possible mechanism is reduced diversity: concentrating probability can improve common successes while losing rare successful paths
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

:::quiz id="quiz-bandit-mdp" title="What Makes a Task Multi-Step?"
An assistant runs code, reads an error, revises its code, and receives **one score at the end**. Is this still a one-step problem because it has only one reward?
+++
**No. There are multiple decisions separated by observations from the environment.**

- The error changes what the assistant observes before its next action.
- A multi-step task can use only a **terminal reward**.
- Credit must still be assigned across the earlier tool calls and revisions.
:::
