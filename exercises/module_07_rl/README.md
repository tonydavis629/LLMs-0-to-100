# Module 7: GRPO on the Instruct Model with a Verifiable Reward

## Overview

Take the small **instruct model** from Module 6 and improve it with **GRPO** (Group
Relative Policy Optimization) on a task it can **verify itself**: reversing a string.
The model, the sampler, the tokenizer, the verifiable-task data, and the runner are
all provided &mdash; your job is the **RL loop**: sample a group of completions, score
each with a Python verifier, turn the rewards into group-relative advantages, compute
per-token log-probabilities under the policy and the frozen reference, build the
policy-gradient loss plus a KL penalty, and take one optimizer step.

The goal is not a useful model. The goal is to make the RL loop **visible**: the
sample &rarr; score &rarr; advantage &rarr; update cycle, a **reward curve that climbs**,
and a held-out accuracy that rises &mdash; driven by **reward**, not imitation.

The starting policy is an instruct model that can *partly* reverse strings. Its
argmax (greedy) answer is often right, but its sampling distribution is broad, so
**sampled** completions are correct only about a fifth of the time. GRPO **sharpens**
that distribution: sampled accuracy climbs from roughly 20% to about 90%. This is the
module's thesis in miniature &mdash; RL concentrates probability on reasoning the model
could already occasionally produce.

## Setup

There is one shared environment for the whole repo. From the repo root:

```bash
uv sync
```

## Running

```bash
uv run python exercises/module_07_rl/src/main.py
```

The runner detects which steps you have implemented and skips the rest, so you can
fill in one step at a time and re-run immediately. It prints the held-out accuracy
**before** (sampled and greedy), the mean reward at each checkpoint during training,
the held-out accuracy **after**, and a sample completion before and after. It also
saves a **reward-curve image** to `output/reward_curve.png`.

## What to implement

Open `exercise.py` and fill in each `raise NotImplementedError(...)` line. Each needs
only one expression or one short block.

| Step | Function | What it does |
|------|----------|--------------|
| 1 | `sample_group()` | Draw `G` completions for one prompt from the policy |
| 2 | `verifiable_reward()` | `1.0` if the completion matches the answer, else `0.0` |
| 3 | `score_group()` | Apply the reward to every completion &rarr; a reward vector |
| 4 | `group_relative_advantages()` | Standardize: `(r - mean) / (std + eps)` |
| 5 | `completion_mask()` | Which positions are generated tokens (train on these only) |
| 6 | `gather_token_log_probs()` | Per-token log-prob via `log_softmax` + `gather` |
| 7 | `pg_loss()` | `-advantage * sum(masked log-probs)` for one completion |
| 8 | `kl_penalty()` | `sum(masked (log pi_policy - log pi_ref))` |
| 9 | `grpo_step()` | Zero gradients, backpropagate (clip + step provided) |
| 10 | `mean_reward()` | Mean of a reward vector, as a float (the curve metric) |

The model (`src/model.py`), tokenizer (`src/tokenizer.py`), data (`src/data.py`),
plotting (`src/visualization.py`), and runner (`src/main.py`) are all provided. The
runner orchestrates the loop and calls the functions you write. You only edit
`exercise.py`.

## Data

- `data/instruct_model.pt` &mdash; the starting policy: a TinyGPT that has been
  finetuned (from the Module 5/6 base) to *partly* reverse strings. It is loaded as
  both the **policy** (trained) and the **frozen reference** (for the KL penalty). It
  is never re-pretrained. (Regenerate it with
  `solution/src/make_instruct_checkpoint.py`.)
- `data/verify_prompts.jsonl` &mdash; reverse-string prompts split into `train`
  (GRPO learns from these) and `eval` (held-out, used only to measure before/after
  accuracy). Each record is `{prompt, answer, split}`; the words are disjoint from the
  ones the instruct model was finetuned on. (Regenerate with `src/data.py`.)

The tokenizer is the Module 6 vocabulary: 65 characters plus four atomic special
tokens (`<|user|>`, `<|assistant|>`, `<|end|>`, `<|pad|>`), for a vocabulary of 69.

## Extra credit

- **KL-penalty ablation.** Set `BETA = 0` and watch reward climb while the completions
  drift into gibberish that games the verifier &mdash; the reward-hacking demo from the
  lecture's "dark side" section.
- **Group-size sweep.** Vary `GROUP_SIZE` and observe how a larger group gives a less
  noisy advantage estimate at higher cost.
- **Reward shaping.** Replace the binary reward with a partial-credit reward (fraction
  of characters correct) and compare learning speed.
- **Length-bias probe.** Add a tiny per-token bonus to the reward and watch the model
  learn to pad its answers &mdash; a concrete Goodhart demonstration.
- **Rejection-sampling baseline.** Instead of a policy-gradient step, keep only the
  highest-reward completion in each group and SFT on it (Best-of-N finetuning). Compare
  its learning curve to GRPO: this shows what the **negative signal** in policy
  gradient buys over keep-the-winner.
- **pass@k probe.** Track pass@1 and pass@k (best of k samples) on the held-out prompts
  across training. Check whether GRPO improves pass@1 while pass@k stays flat &mdash;
  the Yue et al. (2025) claim that RL sharpens rather than expands the base model.
- **DPO comparison.** Build chosen/rejected pairs from the sampled completions and take
  one DPO step, contrasting the off-policy loss with the on-policy GRPO loop.
