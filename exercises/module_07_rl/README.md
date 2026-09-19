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
argmax (greedy) answer is right on 22.5% of the held-out prompts, and its sampling
distribution is broad, so **sampled** completions are correct only 15.9% of the time.
GRPO **sharpens** that distribution: held-out sampled accuracy climbs to 73.1% and
greedy accuracy to 92.5%. This is the module's thesis in miniature: RL
concentrates probability on reasoning the model could already occasionally produce.

## Setup

There is one shared environment for the whole repo. From the repo root:

```bash
uv sync
```

## Running

From the `exercises/` directory:

```
uv run python module_07_rl/src/main.py
```

The runner goes through the ten steps in order. Each step's header line carries a tag, and the step's output follows: what your function produced on the real model (a sampled group, its rewards and advantages, a completion mask, per-token log-probs), then one line per test from `tests/`. The tags are:

| Tag | Meaning |
|-----|---------|
| `CORRECT` | every test for the step passed |
| `INCORRECT` | your code ran but a test failed; the expected and actual values are printed under it |
| `INCOMPLETE` | the function still raises `NotImplementedError` |

```
=== Step 4: group_relative_advantages() === CORRECT
Rewards for the Step 1 group: [ 0.00,  0.00,  0.00,  0.00,  1.00,  0.00,  1.00,  0.00]
Advantages:                   [-0.54, -0.54, -0.54, -0.54, +1.62, -0.54, +1.62, -0.54]
  CORRECT    rewards [1, 0, 1, 0] give [0.866, -0.866, 0.866, -0.866] (mean 0.5, std 0.577)
  CORRECT    advantages have mean 0 and std 1 within a group: rewards [1, 0, 0, 0, 1, 0, 0, 0]
  CORRECT    a group where every reward ties gives all-zero advantages, not NaN: [1, 1, 1, 1]
```

After Step 10 comes the payoff, `GRPO training (Steps 1-10 together)`, which needs all ten steps. It prints the held-out accuracy **before** training (sampled and greedy), the mean group reward every 20 steps, the held-out accuracy **after**, and the example prompt's greedy answer before and after. It saves a **reward-curve image** to `output/reward_curve.png`, then its tests check that the reward climbed and held-out accuracy rose. Training takes a few minutes on a laptop CPU, and a progress line shows the current step while it runs.

Run a single step with `--step` (1 to 10, or `train` for the training run):

```
uv run python module_07_rl/src/main.py --step 4
```

`exercise.py` at the module root is the only file you edit. Everything already written for you lives in `src/`. Run the finished answers with `--solution`:

```
uv run python module_07_rl/src/main.py --solution
```

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
runner orchestrates the loop and calls the functions you write. The tests live in
`tests/`, one file per step (`test_step1_sample_group.py` through
`test_step10_mean_reward.py`, plus `test_training.py` for the training run). Each calls
your function on small tensors with a known answer, so you can read the test for the
step you are on to see exactly what is expected. You only edit `exercise.py`.

## Data

- `data/instruct_model.pt` &mdash; the starting policy: a TinyGPT that has been
  finetuned (from the Module 5/6 base) to *partly* reverse strings. It is loaded as
  both the **policy** (trained) and the **frozen reference** (for the KL penalty). It
  is never re-pretrained. (Regenerate it with
  `src/make_instruct_checkpoint.py`.)
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
