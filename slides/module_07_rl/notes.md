# Module 7: Reinforcement Learning Post-Training — Lecture Notes

These notes give an explanation and a citation for every claim on the slides, map
the equations to the visuals they appear on, and record the historical context.
Module 7 owns the **preference and RL post-training** stage: it completes the
InstructGPT recipe Module 6 began (SFT), and carries it through reward models, DPO,
GRPO, RLVR, and the alignment concerns that come with optimizing a proxy.

## Review

- Module 6 produced an **instruct model**: supervised finetuning on prompt-response
  pairs taught it to follow intent, but it can only **imitate** the demonstrations it
  was shown. Three limits remain (from the Module 6 handoff): SFT cannot exceed its
  demonstrations; it has no clean way to use negative signal; and **exposure bias**
  means it trains on ground-truth prefixes but generates from its own outputs.
- The optimization machinery is unchanged from Modules 2, 5, and 6: gradients,
  backpropagation, AdamW (Loshchilov and Hutter, 2017, arXiv:1711.05101). What
  changes in RL is the **training signal**: instead of a fixed cross-entropy target
  at every position, the signal is a scalar **reward** on the model's own samples.
- Framing carried from the Module 6 handoff: pretraining built the engine, SFT taught
  it to drive, RL teaches it to drive well by trial, error, and feedback.

## a. What Reinforcement Learning Is, Really

### The classical loop
- The RL setting (Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd
  ed., 2018): an **agent** observes a **state** $s$, takes an **action** $a$ under a
  **policy** $\pi(a \mid s)$, receives a scalar **reward** $r$, and transitions to a
  new state. A **trajectory** (episode) is the sequence $s_0, a_0, s_1, \dots$; the
  **return** is the total reward; the objective is to maximize **expected return**.
- There is no labeled correct action; the only feedback is the reward the agent's own
  choices earn. This is the structural difference from supervised learning.
- The slide carries an **agent-environment loop diagram**: the agent emits an *action*
  $a$ to the environment, which returns a *reward* $r$ and a *next state* $s$, closing
  the loop. It is the canonical Sutton-and-Barto interaction diagram.

### Mapping onto a language model
- The slide "Mapping RL Onto a Language Model" makes the abstraction concrete: the
  **policy** is the model $\pi_\theta$; the **state** is the prompt plus tokens
  generated so far; an **action** is emitting the next token; the **episode** is one
  completion; the **reward** is a scalar score on the finished output.

### Credit assignment
- In SFT every position has a known target token and we minimize cross-entropy
  against it (Module 6). In RL there is no correct token, only a scalar reward on the
  whole output, which must be turned into a per-token gradient. This is the
  **credit-assignment problem**, the central challenge of the module.

### Bandit vs MDP
- Classical RL is a multi-step **Markov decision process (MDP)**: each action lands in
  a new state with its own reward. LLM post-training almost always collapses this to a
  one-step **contextual bandit**: the whole completion is a single action earning one
  terminal reward, with no intermediate rewards and discount $\gamma \approx 1$. This
  is why GRPO (section e) assigns **one advantage to the entire sequence** and
  broadcasts it to every token. Genuinely multi-step RL returns in section f's process
  rewards and agentic-RL slides. (This bandit framing of RLHF is discussed by,
  e.g., the TRL and OpenRLHF documentation and Ziegler et al., 2019, arXiv:1909.08593.)

### Policy gradient and REINFORCE
- REINFORCE (Williams, 1992, *Machine Learning* 8:229-256,
  doi:10.1007/BF00992696): the score-function estimator on the slide
  "Policy Gradient: REINFORCE" is
  $$\nabla_\theta J(\theta) = \mathbb{E}_{y \sim \pi_\theta}\big[ R(y) \nabla_\theta \log \pi_\theta(y) \big].$$
  Intuition: push up $\log \pi_\theta(y)$ in proportion to the reward $R(y)$. It is
  gradient **ascent** on expected reward, the same machinery as the gradient descent
  of Module 2g with a flipped sign and a different objective.
- The accompanying **REINFORCE animation** samples several completions, attaches a
  scalar reward to each, and draws the reward-weighted gradient as an up-arrow whose
  length is proportional to the reward. Because every reward here is positive, every
  arrow points up &mdash; visually motivating the baseline on the next slide.
- The estimator is unbiased but high-variance. The "Baseline: Taming Variance" slide
  introduces the **advantage** $A = R - b$:
  $$\nabla_\theta J(\theta) = \mathbb{E}_{y \sim \pi_\theta}\big[ (R(y) - b) \nabla_\theta \log \pi_\theta(y) \big].$$
  Subtracting any action-independent baseline leaves the gradient's expectation
  unchanged (the score-function term has zero expectation) but reduces variance. The
  **choice of baseline** is what most distinguishes PPO (a learned value network) from
  GRPO (the group mean). The slide also explains the term **advantage** literally: how
  much better than the baseline an action turned out &mdash; positive means push up,
  negative means push down.

### Two axes: online/offline and on/off-policy
- The slides "Two Axes Everyone Conflates" and "The Two Axes as a Grid" define two
  orthogonal axes:
  - **Online vs offline**: whether the algorithm generates fresh data during training
    (online) or learns from a fixed pre-collected dataset (offline, like SFT).
  - **On-policy vs off-policy**: whether the data came from the current policy
    (on-policy) or a different/older one (off-policy), which requires a correction.
- In LLM post-training the common cases pair up: REINFORCE, PPO, and GRPO are online
  and on-policy; DPO (section d) is offline and off-policy. PPO's importance-ratio
  clipping (Schulman et al., 2017, arXiv:1707.06347) exists precisely to reuse a batch
  of samples for a few steps without becoming dangerously off-policy. Online on-policy
  methods can explore beyond a fixed dataset (enabling section b and f), at the cost
  of a moving data distribution that makes RL less stable than offline SFT.

### Exploration, exploitation, and entropy collapse
- Exploration vs exploitation is the oldest tension in RL (Sutton and Barto). Because
  exploration in LLM RL happens through sampling, the decoding knobs from Module 4f
  (temperature, top-k, top-p) become **training hyperparameters**.
- **Entropy collapse** (slide "The Failure Mode"): as training sharpens the policy,
  the output entropy falls, sampling diversity dries up, exploration stops, and
  learning stalls. Practitioners counter it with an **entropy bonus** and the
  **KL-to-reference leash** (section c). This connects directly to the pass@k result
  in section f. (See discussions in the DeepSeek-R1 report and subsequent RLVR work on
  entropy regularization.)

## b. Why RL for LLMs

- **Resolving the three SFT limits**: RL optimizes an outcome, so it can exceed
  demonstrations; reward ranks better against worse, giving negative signal; and the
  model trains on its own generations, closing exposure bias. (Framed in the
  InstructGPT motivation, Ouyang et al., 2022, arXiv:2203.02155.)
- **Evaluation is easier than generation**: we often cannot write the ideal target but
  can judge an output after the fact, so a reward is easier to specify than a full
  demonstration. This is the core argument for preference-based learning (Christiano et
  al., 2017, arXiv:1706.03741).
- **The reward spectrum** (slide "Where Does the Reward Come From?"): human preference
  comparisons (RLHF); a learned reward model; an LLM scoring outputs directly
  (LLM-as-judge, e.g. Zheng et al., 2023, MT-Bench, arXiv:2306.05685); an AI judge
  guided by written principles (RLAIF, Bai et al., 2022, arXiv:2212.08073); and a
  programmatic verifier (RLVR, section f).
- **Rejection sampling / Best-of-N** (slide "The Gentlest RL"): sample $N$
  completions, score them, keep the best, and finetune with ordinary cross-entropy.
  It is offline, off-policy, and reuses SFT machinery. At inference, the same idea only
  works when there is a test-time scorer: a verifier, reward model, LLM judge, or task
  metric. Without such a scorer, there is no reward signal to decide which sample to
  keep. Named recipes: STaR (Zelikman et al., 2022, arXiv:2203.14465),
  rejection-sampling finetuning in Llama 2 (Touvron et al., 2023, arXiv:2307.09288),
  and the iterated ReST (Gulcehre et al., 2023, arXiv:2308.08998) and expert
  iteration. The limitation that motivates true policy gradient: rejection sampling
  discards all information in the rejected samples (a coarse keep-or-discard signal),
  whereas REINFORCE and GRPO push down on losers too.
- **Reward is sparser but cheaper supervision**: SFT gives a target token at every
  position, so the supervision is dense; a reward gives one scalar for an entire
  sequence, which is exactly why credit assignment is hard. The trade is cost and
  scale: demonstrations are expensive and finite, while pairwise comparisons are
  cheaper to collect and one reward grades many sampled outputs. Sampling is the
  engine of RL, so decoding is part of the training loop.

## c. RLHF and the Reward Model

- This completes the InstructGPT three-stage recipe (Ouyang et al., 2022): Module 6
  owned **stage 1** (SFT); this section owns **stage 2** (reward model) and **stage 3**
  (PPO). Origins: Christiano et al. (2017) learned from human preferences over
  trajectories; Stiennon et al. (2020, arXiv:2009.01325) applied it to summarization;
  InstructGPT made it the standard recipe and produced the lineage to ChatGPT.
- **Stage 2, the reward model** (slide "Stage 2"): collect human comparisons and fit a
  **Bradley-Terry** model (Bradley and Terry, 1952) that turns pairwise preferences
  into a scalar reward. The loss on the slide is
  $$\mathcal L_{\text{RM}} = -\mathbb E_{(x, y_w, y_l)}\big[\log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)\big],$$
  where $y_w$ is the preferred and $y_l$ the rejected response. The reward model is a
  learned, imperfect proxy for human judgment, which sets up section g.
- **Stage 3, policy optimization** (slide "Stage 3"): maximize reward under a KL
  penalty to the frozen SFT reference,
  $$\max_{\pi_\theta} \mathbb E_{y \sim \pi_\theta}\big[ r_\phi(x, y) \big] - \beta \mathrm{KL}\big(\pi_\theta(\cdot \mid x) \| \pi_{\text{ref}}(\cdot \mid x)\big).$$
  The KL term keeps the policy from drifting into degenerate text that games the
  proxy (Ziegler et al., 2019; Stiennon et al., 2020).
- **PPO conceptually** (slide "PPO, Conceptually"): an actor-critic method (Schulman et
  al., 2017) with a separate value network (critic), a clipped surrogate objective, and
  generalized advantage estimation (GAE, Schulman et al., 2015, arXiv:1506.02438). We
  name these but defer the algebra because the exercise uses the simpler GRPO. The
  Module 6 result returns: a 1.3B InstructGPT model was preferred over 175B GPT-3.
- The **PPO animation** shows the three moving parts: the actor samples a completion
  scored by the reward model ($R$); the critic estimates the value baseline ($V$); the
  advantage is $A = R - V$; and a number line illustrates the clipped surrogate
  clamping the policy/old-policy ratio into $[1-\epsilon, 1+\epsilon]$.

## d. Direct Preference Optimization (DPO)

- The insight (Rafailov et al., 2023, arXiv:2305.18290): the KL-regularized RL
  objective has a closed-form optimal policy, so the reward is implicit in the policy's
  own log-probabilities relative to the reference. No separate reward model and no RL
  loop are needed.
- The DPO loss (slide "The DPO Loss") is a single supervised classification loss on
  chosen-versus-rejected pairs:
  $$\mathcal L_{\text{DPO}} = -\mathbb E_{(x, y_w, y_l)}\Big[\log \sigma\Big(\beta \log \tfrac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \tfrac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\Big)\Big].$$
- The **DPO animation** shows a fixed chosen/rejected pair (nothing sampled), an
  implicit reward read off the policy's log-prob ratio to the frozen reference
  ($r = \beta \log(\pi_\theta / \pi_{\text{ref}})$), and the logistic loss widening the
  margin: raise the chosen, lower the rejected.
- DPO runs with the same machinery as SFT (no sampling, no reward model, no PPO
  instability), which made preference tuning reproducible and accessible. The honest
  trade-off, in the section-a vocabulary: **DPO is offline and off-policy**, so it
  cannot explore beyond the responses already in its dataset.
- The family (named, not detailed): IPO (Azar et al., 2023, arXiv:2310.12036), KTO
  (Ethayarajh et al., 2024, arXiv:2402.01306), ORPO (Hong et al., 2024,
  arXiv:2403.07691), and SimPO (Meng et al., 2024, arXiv:2405.14734).

## e. Value-Free RL and GRPO

- **The cost of PPO**: the critic is roughly the size of the policy, doubling memory
  and adding a second model to train and tune; a bad value estimate destabilizes the
  run.
- **The GRPO insight** (Shao et al., DeepSeekMath, 2024, arXiv:2402.03300): drop the
  critic. For each prompt sample a group of $G$ completions, score them, and use the
  group's own statistics as the baseline. The advantage on the slide is
  $$A_i = \frac{R_i - \operatorname{mean}(R_1, \dots, R_G)}{\operatorname{std}(R_1, \dots, R_G)}.$$
  This is the variance-reduction baseline from section a, but free from the group
  rather than a learned value function. The update keeps PPO-style ratio clipping and
  the KL-to-reference penalty, with no value network.
- **Why build it**: it is dramatically simpler than PPO, makes the
  sample-score-update cycle visible, is the method behind DeepSeek-R1, and is exactly
  what the exercise implements. In the section-a vocabulary, GRPO is online and
  on-policy yet value-free.

- **Interactive widget (`:::interactive widget="grpoGroup"`):** a group of $G = 8$ completions with binary rewards; clicking a bar flips that completion between correct and incorrect. It computes the group mean and population standard deviation and plots $A_i = (R_i - \operatorname{mean}) / \operatorname{std}$ directly. The mixed preset (3 of 8 correct) gives mean 0.375, std 0.484, advantages $+1.29$ and $-0.77$. The two degenerate presets are the point: when every completion scores the same, the standard deviation is zero, every advantage is zero, and the group contributes no gradient at all. That is the practical reason prompt difficulty has to be tuned for GRPO &mdash; prompts that are always solved or never solved are wasted rollouts.

## f. RLVR and the Reasoning Revolution

- **RLVR** (slide "Reinforcement Learning from Verifiable Rewards"): when an answer is
  checkable, the reward is a deterministic program, not a learned model. Math answers
  are known, code can be run against tests, formats can be parsed. No learned proxy
  means nothing for the policy to hack inside the reward (though see section g for what
  can still be hacked).
- **The breakthrough**: OpenAI o1 (September 2024,
  openai.com/index/learning-to-reason-with-llms) showed RL on chain-of-thought plus
  inference-time compute sharply improves reasoning; DeepSeek-R1 (January 2025,
  arXiv:2501.12948) reproduced it openly using GRPO with verifiable rewards. The
  scaling story shifts from pretraining compute (Module 5g) toward RL and
  inference-time compute.
- **Why reward alone grows a chain of thought** (slide "Why Reward Alone Grows a
  Chain of Thought"): the mechanism is selection plus broadcast, not instruction. The
  verifier grades only the final answer, but among sampled completions, those that
  happen to work through intermediate steps are correct more often and so earn positive
  advantage more often (selection). Because LLM RL collapses the episode to a one-step
  bandit (section a), that single advantage is broadcast to every token of the
  completion, so the reasoning tokens are reinforced alongside the answer they produced
  even though the reward never inspects them (broadcast). Length itself is selected
  for: each generated token is one more forward pass of computation spent on the
  problem, so longer chains give the model more chances to reach or check an answer.
  The DeepSeek-R1 report (arXiv:2501.12948, Figure 3) shows R1-Zero's average response
  length rising steadily over RL training despite the reward containing only accuracy
  and format terms, no length term. This is chain-of-thought prompting (Wei et al.,
  2022, arXiv:2201.11903) rediscovered by optimization rather than demonstration: the
  behavior CoT prompting had to elicit with examples, RL finds because it pays.
- **R1-Zero** (slide "The Most Striking Result"): RL applied directly to a base model
  with no SFT produced emergent long chains of thought and self-correction (the "aha
  moment") in the DeepSeek-R1 report. This is the "exceed the demonstrations" promise of
  section b, realized.
- **pass@k** (slide "A Sharper Lens"): pass@1 measures one attempt; pass@k measures
  whether any of $k$ samples succeeds, i.e. the coverage of the distribution. A model
  can have high pass@1 but low pass@k (confident and narrow) or the reverse.
- **The critical counterpoint** (slide "The Critical Counterpoint"): Yue et al. (2025,
  *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the
  Base Model?*, limit-of-rlvr.github.io) find RLVR-trained models beat the base model
  at small $k$ but the base model catches up and often overtakes at large $k$. RLVR
  sharpens the sampling distribution toward reasoning the base model could already
  produce, rather than creating new paths; the reasoning boundary can narrow. This is
  the entropy-collapse mechanism of section a: optimizing pass@1 can shrink the
  diversity pass@k rewards. RL trades coverage for sampling efficiency. The exercise
  demonstrates the mechanism in miniature (sampled accuracy, i.e. pass@1, rises sharply
  as the distribution concentrates).
## g. The Dark Side: Reward Hacking, Over-Optimization, and Alignment

- **Goodhart's law** (Goodhart, 1975; Strathern, 1997): when a measure becomes a
  target it ceases to be a good measure. Every reward here is a proxy; optimize it hard
  enough and the policy games it.
- **Reward over-optimization** (slide "Reward Over-Optimization, Quantified"): as the
  policy's KL from the reference grows, proxy reward keeps rising while true held-out
  reward eventually falls. Gao et al. (2023, arXiv:2210.10760) fit scaling laws to this
  gap. The KL penalty (section c) is the leash.
- **Failure modes**: length/verbosity bias, sycophancy (Perez et al., 2022,
  arXiv:2212.09251; Sharma et al., 2023, arXiv:2310.13548), format exploitation, and
  verifier loopholes. Even a verifier can be hacked through the gap between the check
  and the intent.
- **RLAIF and Constitutional AI** (Bai et al., 2022, arXiv:2212.08073): replace or
  augment human feedback with an AI judge guided by a written constitution, a route
  toward scalable oversight.
- **Alignment framing**: RL is where helpful, honest, and harmless (the HHH target of
  Module 6; Askell et al., 2021, arXiv:2112.00861) becomes an explicit optimization
  objective, and therefore where a misspecified objective becomes actively dangerous.
- **The alignment tax** (slide "The Alignment Tax"): RLHF can degrade raw capability on
  some benchmarks; the InstructGPT paper named this and mitigated it by mixing
  pretraining gradients back into RL. It connects to section f's narrowing of the
  reasoning boundary. Honest limits: reward models are imperfect proxies for contested
  values; verifiable rewards exist only for checkable domains; RL is less stable, more
  compute-hungry, and harder to debug than SFT.

## h. The Post-Training Stack

- The full modern recipe is now visible end to end: pretrain (Module 5), supervised
  finetune (Module 6), then preference and RL post-training (Module 7). The through-line
  of the course: the optimization machinery never changed; only the data and the target
  did. Module 7's target was a scalar reward on the model's own samples.
- Handoff to Module 8: the same transformer and training stack, but the inputs expand
  beyond text to images, audio, and video. The optimization stays; the modality grows.

## Notable Figures

- **Richard Sutton and Andrew Barto** (2024 Turing Award): wrote the foundational RL
  textbook; Sutton's "The Bitter Lesson" frames the search-and-learning view RL
  embodies. Introduced during section a.
- **Ronald J. Williams**: introduced REINFORCE (1992), the policy-gradient estimator
  underlying every method here. Section a.
- **Paul Christiano and collaborators**: deep RL from human preferences (2017), the
  origin of the reward-model recipe. Section c.
- **Long Ouyang and the InstructGPT team (OpenAI)**: made RLHF the standard recipe
  (2022); met in Module 6, returning to complete stages 2 and 3. Section c.
- **Rafael Rafailov and collaborators**: introduced DPO (2023). Section d.
- **The DeepSeek team (Zhihong Shao and collaborators)**: introduced GRPO (2024) and
  DeepSeek-R1 (2025). Sections e and f.
- **Yuntao Bai and the Anthropic team**: introduced RLAIF and Constitutional AI (2022).
  Section g.

## Side Quests

These are genuine asides &mdash; interesting but not on the critical path.

- **The KL penalty as a leash** (near section c): the same run with and without the KL
  term; without it the policy drifts into high-reward gibberish. Reproduced in the
  exercise extra credit ($\beta = 0$).
- **R1-Zero and the skeptic** (near section f): staged as a debate between the R1-Zero
  emergence claim and the Yue et al. (2025) pass@k rebuttal. pass@1 and pass@k can point
  in opposite directions because RL reweights probability mass rather than adding paths.

## Comparison and Cross-Cutting Slides

Regular lecture slides (not side quests), interleaved where relevant.

- **DPO vs PPO vs GRPO** (section e): a 2x2 grid on the section-a axes. DPO is
  offline/off-policy and cheapest; PPO is online/on-policy with a critic; GRPO is
  online/on-policy and value-free.
- **Agentic and multi-turn RL** (section f): tool calls reintroduce real
  intermediate states and rewards, turning the bandit back into an MDP. The slide
  opens with an explicit MDP-vs-bandit reminder. The frontier Yue et al. point to
  beyond the single-turn base-model ceiling.
- **Process vs outcome rewards** (section f): a process reward model grades each
  reasoning step (denser, easier credit assignment, more expensive) versus a single
  outcome reward (Lightman et al., 2023, "Let's Verify Step by Step", arXiv:2305.20050).
## Exercise: GRPO with a Verifiable Reward

- **Setup**: the policy is a small instruct model (a TinyGPT finetuned from the Module
  5/6 base) that can partly reverse strings. Its argmax is often right but its sampling
  distribution is broad, so sampled completions verify only about 16% of the time. The
  reward is a Python function (`verifiable_reward`): reverse the input and compare. No
  human labels, no reward model.
- **The loop**: for each of a batch of prompts, sample a group of $G = 8$ completions,
  score each with the verifier, standardize the rewards into group-relative advantages,
  compute per-token log-probabilities under the policy and the frozen reference, build
  the policy-gradient loss plus a KL penalty, and take one optimizer step. This is GRPO
  with the section-e advantage formula and the section-c KL leash.
- **Policy-gradient loss in the exercise**: the lecture presents gradient ascent on
  expected reward, but the code uses PyTorch's gradient descent convention. For one
  completion, Step 7 implements $\mathcal L_{\text{PG}} = -A_i \sum_t m_t \log
  \pi_\theta(y_t \mid x, y_{<t})$, where $A_i$ is the group-relative advantage and
  $m_t$ masks out prompt tokens.
- **Result** (actual solution output): over 400 steps the mean group reward climbs from
  about 0.15 to 0.96; held-out sampled accuracy rises from 15.9% to 73.1% and greedy
  from 22.5% to 92.5%. The sample prompt `reverse: sukgh` flips from `hgkuk` (wrong) to
  `hgkus` (correct). This is the module's thesis in miniature: RL **sharpens** the
  distribution onto reversals the model could already occasionally sample, which is why
  the sampled-accuracy (pass@1) gain is dramatic.
- **Extra credit** maps onto the lecture: the $\beta = 0$ ablation reproduces reward
  hacking (section g); the rejection-sampling baseline shows what the negative signal
  buys over keep-the-winner (section b); the pass@k probe reproduces the Yue et al.
  finding (section f) in miniature.

## References

- Williams, "Simple Statistical Gradient-Following Algorithms for Connectionist
  Reinforcement Learning," *Machine Learning*, 1992. doi:10.1007/BF00992696
- Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd ed., 2018.
- Christiano et al., "Deep Reinforcement Learning from Human Preferences," 2017.
  arXiv:1706.03741
- Ziegler et al., "Fine-Tuning Language Models from Human Preferences," 2019.
  arXiv:1909.08593
- Stiennon et al., "Learning to Summarize from Human Feedback," 2020. arXiv:2009.01325
- Ouyang et al., "Training Language Models to Follow Instructions with Human Feedback"
  (InstructGPT), 2022. arXiv:2203.02155
- Schulman et al., "Proximal Policy Optimization Algorithms," 2017. arXiv:1707.06347
- Schulman et al., "High-Dimensional Continuous Control Using GAE," 2015.
  arXiv:1506.02438
- Rafailov et al., "Direct Preference Optimization," 2023. arXiv:2305.18290
- Shao et al., "DeepSeekMath" (GRPO), 2024. arXiv:2402.03300
- DeepSeek-AI, "DeepSeek-R1," 2025. arXiv:2501.12948
- OpenAI, "Learning to Reason with LLMs" (o1), 2024.
- Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models,"
  2022. arXiv:2201.11903
- Yue et al., "Does RL Really Incentivize Reasoning Capacity in LLMs Beyond the Base
  Model?", 2025. limit-of-rlvr.github.io
- Gao et al., "Scaling Laws for Reward Model Overoptimization," 2023. arXiv:2210.10760
- Bai et al., "Constitutional AI: Harmlessness from AI Feedback," 2022. arXiv:2212.08073
- Lightman et al., "Let's Verify Step by Step," 2023. arXiv:2305.20050
- Zelikman et al., "STaR: Bootstrapping Reasoning with Reasoning," 2022.
  arXiv:2203.14465
- Gulcehre et al., "Reinforced Self-Training (ReST)," 2023. arXiv:2308.08998
- Touvron et al., "Llama 2," 2023. arXiv:2307.09288
- HuggingFace TRL (PPO, DPO, GRPO reference implementations), huggingface.co/docs/trl
