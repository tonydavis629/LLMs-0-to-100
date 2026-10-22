# Module 7: Reinforcement Learning Post-Training

These notes follow the editable slide partials in `source/slides/` in presentation
order. They update the framing, algorithm visuals, and limitations in the earlier
module notes. Numbers inside the algorithm diagrams are illustrative, not measured
training output; the exercise terminal slides retain captured solution output.

## Review and motivation

`review-sft` and `review-machinery`: ordinary supervised finetuning minimizes
cross-entropy on demonstrated response tokens. The model can generalize beyond
those examples, but the objective does not explicitly rank chosen and rejected
answers or directly optimize a task score. During generation it conditions on its
own earlier outputs; teacher-forced training conditions on dataset prefixes.
RL post-training changes the objective and data collection while reusing forward
passes, backpropagation, and an optimizer. See Ouyang et al. (2022),
[InstructGPT](https://arxiv.org/abs/2203.02155), and Sutton and Barto (2018),
[Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html).

`eval-easier-than-gen` and `resolve-three-limits`: a checker or preference judgment
can be available even when writing an ideal response is expensive. Running tests
is the concrete example; passing a finite test suite is evidence of correctness,
not a proof that arbitrary code is correct. A reward can select among responses
absent from the demonstrations. This is an opportunity for improvement, not a
guarantee of novel reasoning. See Christiano et al. (2017),
[Deep RL from Human Preferences](https://arxiv.org/abs/1706.03741).

## RL context before algorithms

`rl-vocabulary`: the agent chooses actions under a policy, and the environment
returns observations and rewards. The objective is expected return. In a
multi-step problem the return may be $G_t=\sum_{k\geq0}\gamma^k r_{t+k+1}$;
intermediate rewards may all be zero. The interaction diagram represents this
information flow. Sutton and Barto (2018), chapters 1 and 3.

`figure-sutton-barto`: Sutton and Barto wrote the standard textbook (first edition
1998, second edition 2018) and received the 2024 ACM A. M. Turing Award. See the
[ACM award announcement](https://amturing.acm.org/).

`rl-as-llm`: the policy is the conditional distribution $\pi_\theta(y\mid x)$
implemented by the transformer. Its parameters $\theta$ are the model's weights.
Changing those weights changes probabilities of future generated responses.

`bandit-vs-mdp`: treating a complete response as the action gives a contextual
bandit abstraction: context $x$, action $y$, terminal score $R(x,y)$. At token
resolution the same response can be represented as a sequential process with
prefix states. The distinction is the modeling granularity, not whether the LLM
generates multiple tokens. A terminal reward alone does not establish that a
problem is a bandit. See Ziegler et al. (2019),
[Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593).

`credit-assignment`: autoregressive factorization gives

$$\log\pi_\theta(y\mid x)=\sum_t\log\pi_\theta(y_t\mid x,y_{<t}).$$

Weighting this sum by one scalar trains the generated tokens, but does not reveal
which token causally determined the result. Do not describe the response-level
view as eliminating credit assignment.

`agentic-rl`: tool results create observations between decisions. With a sufficient
state representation, such as the relevant interaction history, this fits an MDP;
partially observed environments may require a POMDP formulation. The code/error/
revision example can receive only a terminal score and still be multi-step.
Sutton and Barto (2018), chapter 3.

`online-sampling` and `on-policy-sampling`: online means collecting new interaction
data during training. On-policy concerns whether samples match the policy in the
learning objective. They are related, not a Cartesian taxonomy of the methods in
this lecture. Online off-policy methods exist, such as replay-based Q-learning.
PPO and GRPO collect recent batches and retain a sampling-policy snapshot for
importance ratios during limited batch reuse. Ordinary fixed-dataset DPO is best
introduced as offline preference optimization rather than forced into an RL grid.

`exploration-exploitation`: stochastic decoding supplies varied candidates;
concentrating probability on successful candidates exploits observed feedback.
Temperature and truncation affect the training distribution. Exact on-policy
identities assume sampling from the policy in the objective; changing that
distribution can require correction. Sutton and Barto, chapters 2 and 13.

## Reward sources

`reward-cheaper-supervision`: response-level scores are sparser than demonstrated
token targets. Preference comparisons can be cheaper than writing ideal answers;
reusable programmatic checks can scale across many samples. Cost depends on task.

`reward-spectrum`: the table includes classical and language-model settings.
Programmed environment rewards include game scores and simulator goal conditions
(Williams, 1992; Schulman et al., 2017). A programmatic verifier combined with LLM
policy optimization is commonly called RLVR (Shao et al., 2024; DeepSeek-AI, 2025).
A reward model learned from human preferences combined with policy optimization
is the RLHF recipe (Christiano et al., 2017; Ouyang et al., 2022). AI feedback may
score responses directly or train a reward model (Bai et al., 2022,
[Constitutional AI](https://arxiv.org/abs/2212.08073)). REINFORCE, PPO, and GRPO are
update methods; these reward sources are compatible with multiple such methods.
The pairings in the table are examples, not restrictions or algorithm definitions.

## REINFORCE

`expected-reward` conditions on a prompt for clarity; training also averages over
the prompt distribution:

$$J(\theta)=\mathbb E_{y\sim\pi_\theta(\cdot\mid x)}[R(x,y)].$$

`reward-baseline`, `policy-gradient`, and `figure-williams`: Williams (1992),
[Simple Statistical Gradient-Following Algorithms](https://doi.org/10.1007/BF00992696),
gives the score-function estimator. For a baseline independent of the sampled
action, the expected gradient is

$$\nabla_\theta J=\mathbb E[(R-b)\nabla_\theta\log\pi_\theta(y\mid x)].$$

The baseline term has zero expectation because
$\mathbb E[\nabla_\theta\log\pi_\theta]=0$. A suitable baseline can reduce variance;
an arbitrary one is not guaranteed to do so. A fixed or earlier-batch running
baseline is used in the teaching example. Gradient descent minimizes the sampled
surrogate $\mathcal L=-A\sum_t\log\pi_\theta(y_t\mid x,y_{<t})$, treating $A$ as
detached. No derivative through a Python reward checker is needed.

`reinforce-anim`: the full loop remains visible. Five clicks trace sampling with
fixed weights, checking with fixed weights, advantage calculation, forward/backward
computation and the optimizer step, then new sampling. The batch
`[tac, tca, tac, cta]` earns `[1, 0, 1, 0]` from exact reversal comparison. With
$b=0.5$, advantages are $[0.5,-0.5,0.5,-0.5]$. The colored weight cells mark the
optimizer phase; they are a schematic rather than recorded tensor values.

## PPO and GRPO

`ppo-conceptually`: the actor is the language policy; the critic predicts expected
return. It can share a backbone or use another model, so its cost is not always a
second full policy. Full token-level PPO commonly uses generalized advantage
estimation, not simply a single response reward minus one prompt value. The
response-level $R-V_\psi(x)$ example isolates the role of the baseline.
Schulman et al. (2015), [GAE](https://arxiv.org/abs/1506.02438).

`ppo-clipping` and `ppo-anim`: Schulman et al. (2017),
[PPO](https://arxiv.org/abs/1707.06347), use a clipped surrogate:

$$L^{\mathrm{CLIP}}=\mathbb E_t\left[\min\left(r_t A_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t\right)\right],\qquad
r_t=\frac{\pi_\theta(y_t\mid s_t)}{\pi_{\mathrm{old}}(y_t\mid s_t)}.$$

The old policy is the snapshot that sampled the batch. Clipping removes incentives
for sufficiently large ratio changes in the favorable direction; it does not
enforce a hard trust region or guarantee every ratio remains inside the interval.
The critic learns from observed returns. The PPO diagram retains the same prompt,
responses, reward, and weights as REINFORCE, replacing its running baseline with
a toy critic prediction of 0.5 and adding critic training and clipped batch reuse.

`ppo-cost`, `grpo-insight`, `grpo-baseline`, and `grpo-why-build`: Shao et al.
(2024), [DeepSeekMath](https://arxiv.org/abs/2402.03300), introduce GRPO. For one
prompt, sample $G$ responses and compute

$$A_i=\frac{R_i-\bar R}{\sigma_R+\epsilon},\qquad
\bar R=\frac1G\sum_{i=1}^G R_i.$$

The group supplies an empirical comparison and scale without a learned critic.
Sampling a group still costs computation; the baseline is not literally free.
Full GRPO uses clipped probability ratios and a reference-policy KL regularizer.
The DeepSeekMath and DeepSeek-R1 papers describe the specific objectives used.

The within-group mean includes each response's own reward. It is therefore not
independent of that response's sampled action: the simple unbiased-baseline proof
for REINFORCE does not apply unchanged. Without standardization the inclusive
mean introduces a $(G-1)/G$ factor for independent samples. Normalization by the
sample-dependent standard deviation further changes the estimator. A leave-one-out
baseline has a different independence property. The lecture teaches relative
weighting, not an exact unbiasedness claim for normalized GRPO.

`grpo-anim` and `grpo-group`: the diagram uses population standard deviation 0.5
for rewards $[1,0,1,0]$, giving advantages $[1,-1,1,-1]$. The interactive widget
also uses population standard deviation. The exercise uses PyTorch's sample
standard deviation: for four such rewards it is about 0.577, yielding magnitudes
about 0.866. These conventions differ by scale; the signs are identical. An
all-tied group has zero advantages with a guarded denominator and provides no
relative policy-gradient signal. A separate KL term can still contribute a loss.

The `grpo-anim` loss box explicitly minimizes
$\mathcal L=-J_{\mathrm{clip}}+\beta D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}})$.
Here $J_{\mathrm{clip}}$ denotes the clipped, group-advantage-weighted policy
objective, averaged over sampled responses and tokens. The KL term is likewise
averaged over token states; implementations estimate it from token probabilities.
The frozen reference scores the same response tokens and supplies a separate
regularization signal to the loss. Its weights never receive optimizer updates.
The reference $\pi_{\mathrm{ref}}$ differs from the sampling snapshot
$\pi_{\mathrm{old}}$ used in clipping. This schematic shows the original
KL-regularized GRPO objective (Shao et al., 2024); variants may set $\beta=0$.

`figure-deepseek`: DeepSeekMath introduced GRPO in 2024; DeepSeek-R1 reported its
use for reasoning post-training in 2025. See
[DeepSeek-R1](https://arxiv.org/abs/2501.12948). These are representative modern
methods, not a claim that every contemporary reasoning model uses GRPO.

## RLHF, reward models, and AI feedback

`rlhf-completes-instructgpt`, `rlhf-pipeline`, `figure-christiano`, and
`figure-ouyang-7`: the historical pipeline is SFT, human comparison collection and
reward-model fitting, then PPO against that model with reference regularization.
Christiano et al. (2017), Stiennon et al. (2020),
[Learning to Summarize](https://arxiv.org/abs/2009.01325), and Ouyang et al. (2022)
are the primary sources. InstructGPT is a precursor to ChatGPT, not the identical
training run. The three-stage SVG shows information flow across these stages.

`reward-model` uses the Bradley–Terry preference model:

$$\mathcal L_{\mathrm{RM}}=-\mathbb E\log\sigma(r_\phi(x,y_w)-r_\phi(x,y_l)).$$

Here $y_w$ is preferred and $y_l$ rejected. This trains the reward model's weights
$\phi$; during subsequent policy optimization the scorer is held fixed. The reward
model is distinct from the PPO critic: one evaluates responses, the other predicts
expected returns for a variance-reduction baseline.

`rlhf-objective` uses the conceptual objective

$$\mathbb E_{y\sim\pi_\theta}[r_\phi(x,y)]-
\beta\operatorname{KL}(\pi_\theta(\cdot\mid x)\Vert\pi_{\mathrm{ref}}(\cdot\mid x)).$$

The frozen reference regularizes drift from an initial model. It is distinct from
$\pi_{\mathrm{old}}$, refreshed to record which policy generated a training batch.
KL discourages drift but does not guarantee quality or prevent all reward hacking.
Ziegler et al. (2019), Stiennon et al. (2020), and Ouyang et al. (2022).

`kl-vs-cross-entropy`: for distributions over the same support,
$H(p,q)=H(p)+\operatorname{KL}(p\Vert q)$. In the RLHF penalty the changing policy
occupies the $p$ slot, so its entropy is not constant. A uniform binary reference
has cross-entropy one bit for both a uniform and a collapsed policy, but their KL
values are zero and one bit respectively. See Kullback and Leibler (1951),
[On Information and Sufficiency](https://doi.org/10.1214/aoms/1177729694), and Cover
and Thomas (2006), chapter 2. Logarithms use bits in that table and nats in the
usual training losses; the conceptual identities do not depend on the base.

`rlaif` and `figure-bai`: Constitutional AI combines critique/revision and AI
preference feedback guided by written principles. The principles, judge, and
reward-model training are additional choices, not a new policy-gradient identity.
Bai et al. (2022), [Constitutional AI](https://arxiv.org/abs/2212.08073).

## Selection, DPO, and comparison

`rejection-sampling`: finetuning on selected high-scoring responses is supervised
learning on reward-filtered data. A fixed collection is offline; repeating
collection and fitting makes an iterative online recipe. Inference-time Best-of-N
selects a response without changing weights. References: Zelikman et al. (2022),
[STaR](https://arxiv.org/abs/2203.14465); Touvron et al. (2023),
[Llama 2](https://arxiv.org/abs/2307.09288); Gulcehre et al. (2023),
[ReST](https://arxiv.org/abs/2308.08998).

`dpo-insight`, `dpo-loss`, `dpo-why-matters`, and `figure-rafailov`:
Rafailov et al. (2023), [DPO](https://arxiv.org/abs/2305.18290), derive a preference
loss from a KL-regularized reward objective. Define
$m=\beta[\log\pi_\theta(y_w\mid x)-\log\pi_{\mathrm{ref}}(y_w\mid x)
-\log\pi_\theta(y_l\mid x)+\log\pi_{\mathrm{ref}}(y_l\mid x)]$;
then $\mathcal L_{\mathrm{DPO}}=-\mathbb E\log\sigma(m)$.
There is no explicit return baseline, online sampling, or separate reward checker
in the standard fixed-pair update. Generalization beyond dataset responses is
possible; it is incorrect to claim offline training cannot do that.

`dpo-anim` keeps the same visual arrangement but labels the input responses as a
fixed preference pair. The frozen reference contributes log-probabilities, the
preference margin supplies the loss, and only the policy weights are updated.
The final phase loads another pair rather than sampling a fresh response.
`dpo-ppo-grpo` compares these actual training loops rather than an orthogonal grid.

The detailed DPO animation uses a two-response toy distribution with $\beta=1$.
Initially both models assign probabilities $(0.25,0.75)$, so both log-probability
ratios are zero, $m=0$, and $\mathcal L=\log 2\approx0.693$. An illustrative later
policy assigns $(0.60,0.40)$ while the reference stays fixed. Its relative scores
are $\log(0.60/0.25)\approx0.875$ and $\log(0.40/0.75)\approx-0.629$, hence
$m\approx1.504$ and $\mathcal L\approx0.201$. These are consistent hypothetical
before/after values, not the result of a specified single optimizer step.
The diagram renders the margin, loss, and optimizer equations with KaTeX.

## Verifiable rewards and reasoning

`rlvr-definition`: answer comparison, executable tests, and format validation can
provide scores. The checker must still encode the intended task correctly.
`rlvr-breakthrough`: OpenAI's
[Learning to Reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/)
(2024) reports reasoning gains from RL and inference-time computation; DeepSeek-R1
(2025) publishes an open reasoning-training recipe using GRPO. These are different
systems and do not establish that their internal algorithms are identical.

`reward-is-curriculum`, `why-cot-grows`, and `r1-zero`: DeepSeek-R1 reports that
R1-Zero, trained directly from a base model without preliminary SFT, developed
longer responses and self-correction. A final-answer reward can reinforce all
tokens of successful responses, including useful intermediate reasoning. More
tokens provide more computation, but unnecessary length need not improve reward;
longer reasoning is not guaranteed across tasks or models. The report documents
response-length growth without an explicit length reward.

`process-reward`: outcome rewards evaluate final results; process supervision
evaluates intermediate reasoning steps and requires different labels or a process
reward model. Adding intermediate scores is a reward-design choice, independent
of whether the environment has multiple decisions. Lightman et al. (2023),
[Let's Verify Step by Step](https://arxiv.org/abs/2305.20050).

## Failure modes and limits, grouped after the RL explanations

`goodhart`, `over-optimization`, and `reward-hacking-modes`: the familiar wording
of Goodhart's law is Marilyn Strathern's (1997) formulation. A proxy can diverge
from the intended objective as optimization intensifies. Gao et al. (2023),
[Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760),
measure this in specific reward-model setups; it is not a theorem that quality
must fall in every RL run. Verbosity bias, format gaming, sycophancy, and gaps in
test coverage are examples. See Sharma et al. (2023),
[Towards Understanding Sycophancy](https://arxiv.org/abs/2310.13548).

`entropy-collapse`: excessive concentration can reduce useful exploration and
cause learning to stall. Entropy regularization rewards spread; reference KL can
discourage loss of diversity relative to a broad reference but is not itself a
universal entropy guarantee. `side-quest-kl-leash` compares regularization choices:
setting $\beta=0$ is an ablation, not proof that reward hacking will occur.

`passk-definition`: for a task whose independent attempts each succeed with
probability $p$, $\operatorname{pass@k}=1-(1-p)^k$. For a fixed model and task,
pass@k cannot decrease as $k$ increases. Across training, improving average pass@1
can coincide with reduced coverage on other problems at larger $k$.

`yue-counterpoint`, `passk-anim`, and `side-quest-r1-debate`: Yue et al. (2025),
[Does RL Really Incentivize Reasoning Capacity Beyond the Base Model?](https://limit-of-rlvr.github.io/),
report small-$k$ gains and large-$k$ crossover in their evaluated setups. The Manim
curves are schematic illustrations of that finding, not a new measurement.
This is evidence of redistribution toward existing successful paths, not a proof
that RL can never acquire new capabilities or that every gain is entropy collapse.

`alignment-framing`, `alignment-tax`, and `honest-limits`: reward design makes
behavioral preferences explicit. Ouyang et al. (2022) report capability regressions
on some tasks and mitigation by mixing pretraining gradients into PPO. Trade-offs
should be evaluated for the intended use, rather than assumed beneficial. Helpful,
honest, and harmless framing: Askell et al. (2021),
[A General Language Assistant as a Laboratory](https://arxiv.org/abs/2112.00861).

## Stack, exercise, and retrieval questions

`full-recipe`, `through-line`, and `handoff-module-8`: pretraining, SFT, and
preference/RL optimization are a useful training-stack summary. Actual systems
may repeat stages, combine losses, or omit a stage. The common optimization
machinery extends to multimodal inputs; that is the next module's topic.

The exercise walkthrough remains aligned with `exercises/module_07_rl/exercise.py`.
It uses fresh groups, sample-standardized relative advantages, a generated-token
mask, a policy-gradient loss, a frozen reference, and one optimizer update per
fresh batch. It omits the old-policy ratio needed for multiple PPO-style updates
on stale batches. The recorded solution output reports sampled accuracy
14.4% → 54.1% and greedy accuracy 22.5% → 75.0%; these are results from the bundled
run, not guarantees for arbitrary training settings. See the existing module
notes for the captured-run setup. Extra-credit KL and pass@k experiments should
measure outcomes rather than promise a particular failure or crossover.

The quizzes ask students to connect score versus relative advantage, diversity
versus coverage, scorer versus critic, KL versus guaranteed quality, fixed-data
training versus online exploration, and multi-step observations versus reward
timing. Answers deliberately avoid claiming that GRPO normalization is unbiased,
that all RL is on-policy, or that terminal-only reward implies a bandit.

`ppo-cost` depicts the actor and critic as two training jobs, each with a loss,
backpropagation, gradients, and optimizer state. A separate critic backbone can
be expensive; a shared value head can reduce that overhead. The diagram is a
conceptual accounting of work, not a memory-to-scale chart.

The final `resources` slide lists the primary papers and textbook cited above.
