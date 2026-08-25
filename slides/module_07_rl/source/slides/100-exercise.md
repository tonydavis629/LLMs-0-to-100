:::divider id="divider-exercise" title="Exercise" sub="GRPO with a verifiable reward"
:::

---

<!-- .slide: id="exercise-run" -->

## Running the Exercise

Open `module_07_rl/exercise.py` and fill in the ten `NotImplementedError` lines.

- Provided: model, sampler, tokenizer, task data, runner
- Run after each step; unfinished steps are skipped automatically

```bash
# Improve the instruct model with GRPO on a verifiable reverse task
cd exercises
uv run python module_07_rl/src/main.py
```

`data/instruct_model.pt` ships with the repo: the Module 6 result, built for you by `solution/src/make_instruct_checkpoint.py`. Nothing is downloaded. <!-- .element: class="text-md" style="margin-top: 22px;" -->

The runner prints held-out accuracy **before and after** (sampled and greedy), mean reward at each checkpoint, a sample completion before and after, and saves a **reward-curve image**. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="exercise-overview" -->

## Exercise: Make the RL Loop Visible

- Starting policy: an instruct model that can **partly** reverse strings
- Reward: a **Python function** that reverses the input and compares
- No human labels, no reward model

:::columns cols="2" gap="30px"
**The payoff**

- Before: only ~16% of sampled completions verify
- GRPO **sharpens** the distribution
- After: sampled accuracy ~73%, greedy ~92%
+++
**Ten one-line steps**

Sample a group, write the verifier, score the group, standardize advantages, mask the completion, gather log-probs, build the policy-gradient loss, add the KL penalty, take a step, track reward.
:::

---

:::step id="exercise-step1" title="Step 1: sample_group()"
```python
def sample_group(policy, prompt_ids, group_size, max_new_tokens,
                 block_size, temperature, generate_fn, generator):
    """Draw group_size completions for one prompt from the policy."""
    # TODO: Return a list of group_size completions, each from generate_fn(policy,
    #       prompt_ids, max_new_tokens, block_size, temperature=temperature,
    #       generator=generator)[0].
    raise NotImplementedError("TODO: sample a group of completions from the policy")
```
+++
**Hint:** a list comprehension over `range(group_size)`; index `[0]` to drop the batch dim.
+++
**Answer:**

```python
return [
    generate_fn(policy, prompt_ids, max_new_tokens, block_size,
                temperature=temperature, generator=generator)[0]
    for _ in range(group_size)
]
```
:::

---

:::step id="exercise-step2" title="Step 2: verifiable_reward()"
```python
def verifiable_reward(response: str, target: str) -> float:
    """Return 1.0 if the response exactly matches the answer, else 0.0."""
    # TODO: Return 1.0 when response equals target, otherwise 0.0.
    raise NotImplementedError("TODO: return 1.0 for an exact match else 0.0")
```
+++
**Hint:** a single comparison; return a float.
+++
**Answer:**

```python
return 1.0 if response == target else 0.0
```
:::

---

:::step id="exercise-step3" title="Step 3: score_group()"
```python
def score_group(responses, target) -> torch.Tensor:
    """Apply the verifiable reward to every completion."""
    # TODO: Return a tensor of verifiable_reward(r, target) for each r in responses.
    raise NotImplementedError("TODO: score every completion into a reward vector")
```
+++
**Hint:** build a Python list with a comprehension, wrap it in `torch.tensor(...)`.
+++
**Answer:**

```python
return torch.tensor([verifiable_reward(r, target) for r in responses])
```
:::

---

:::step id="exercise-step4" title="Step 4: group_relative_advantages()"
```python
def group_relative_advantages(rewards, eps=1e-6) -> torch.Tensor:
    """Standardize the group's rewards: (r - mean) / (std + eps)."""
    # TODO: Return (rewards - mean) / (std + eps).
    raise NotImplementedError("TODO: standardize rewards into group-relative advantages")
```
+++
**Hint:** `rewards.mean()` and `rewards.std()`; add `eps` to the std before dividing.
+++
**Answer:**

```python
return (rewards - rewards.mean()) / (rewards.std() + eps)
```
:::

---

:::step id="exercise-step5" title="Step 5: completion_mask()"
```python
def completion_mask(prompt_len, seq_len) -> torch.Tensor:
    """True at the next-token positions that predict generated tokens."""
    # TODO: Return a bool tensor of length seq_len - 1 that is True at positions
    #       t >= prompt_len - 1 and False elsewhere.
    raise NotImplementedError("TODO: build the completion mask over target positions")
```
+++
**Hint:** `torch.arange(seq_len - 1)` gives the positions; compare it to `prompt_len - 1`.
+++
**Answer:**

```python
return torch.arange(seq_len - 1) >= (prompt_len - 1)
```
:::

---

:::step id="exercise-step6" title="Step 6: gather_token_log_probs()"
```python
def gather_token_log_probs(logits, target_ids) -> torch.Tensor:
    """Log-prob the model assigns to each actually-taken token."""
    # TODO: Return the log-probability of each target token: log_softmax the logits
    #       over the vocab dimension, then gather the entry at each target id.
    raise NotImplementedError("TODO: gather the per-token log-probabilities")
```
+++
**Hint:** `F.log_softmax(logits, dim=-1)`, then `.gather(-1, target_ids.unsqueeze(-1))` and `.squeeze(-1)`.
+++
**Answer:**

```python
log_probs = F.log_softmax(logits, dim=-1)
return log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
```
:::

---

<!-- .slide: id="exercise-pg-loss-bridge" -->

## From Policy Gradient to `pg_loss`

The lecture used gradient **ascent** on expected reward. PyTorch optimizers do **descent**, so the exercise negates:

$$\mathcal L_{\text{PG}} = -\textcolor{#50c878}{A_i} \sum_t \textcolor{#4a9eff}{m_t}\log \pi_\theta(y_t \mid x, y_{<t})$$

:::columns cols="2" gap="34px"
- `advantage` is $\textcolor{#50c878}{A_i}$: better or worse than the group baseline
- `mask` is $\textcolor{#4a9eff}{m_t}$: 1 for completion tokens, 0 for prompt tokens
+++
- `token_log_probs` are the per-token $\log \pi_\theta$ values from Step 6
- `pg_loss()` returns one scalar for one sampled completion
:::

---

:::step id="exercise-step7" title="Step 7: pg_loss()"
```python
def pg_loss(token_log_probs, advantage, mask) -> torch.Tensor:
    """Advantage-weighted negative log-prob over the completion tokens."""
    # TODO: Return -advantage times the sum of the masked per-token log-probs.
    raise NotImplementedError("TODO: build the advantage-weighted policy-gradient loss")
```
+++
**Hint:** multiply `token_log_probs` by `mask`, `.sum()` it, multiply by `-advantage`.
+++
**Answer:**

```python
return -advantage * (token_log_probs * mask).sum()
```
:::

---

:::step id="exercise-step8" title="Step 8: kl_penalty()"
```python
def kl_penalty(policy_log_probs, ref_log_probs, mask) -> torch.Tensor:
    """A per-token estimate of KL(policy || reference) over the completion."""
    # TODO: Return the sum over masked positions of (policy_log_probs - ref_log_probs).
    raise NotImplementedError("TODO: build the KL-to-reference penalty")
```
+++
**Hint:** subtract the two log-prob tensors, multiply by `mask`, then `.sum()`.
+++
**Answer:**

```python
return ((policy_log_probs - ref_log_probs) * mask).sum()
```
:::

---

:::step id="exercise-step9" title="Step 9: grpo_step()"
```python
    # TODO: Clear last step's gradients, then backpropagate this step's loss.
    raise NotImplementedError("TODO: zero the gradients and backpropagate the loss")

    # Provided: clip the global gradient norm for stability, then take the step.
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return loss.item()
```
+++
**Hint:** the optimizer has a method to zero gradients (use `set_to_none=True`); the loss tensor has a method that backpropagates.
+++
**Answer:**

```python
optimizer.zero_grad(set_to_none=True)
loss.backward()
```
:::

---

:::step id="exercise-step10" title="Step 10: mean_reward()"
```python
def mean_reward(rewards) -> float:
    """Average reward over a vector of rewards, as a plain float."""
    # TODO: Return the mean of rewards as a Python float.
    raise NotImplementedError("TODO: return the mean reward as a float")
```
+++
**Hint:** `rewards.mean()` gives a tensor; `.item()` converts it to a float.
+++
**Answer:**

```python
return rewards.mean().item()
```
:::

---

:::terminal id="exercise-output-before" title="Before GRPO" cmd="uv run python module_07_rl/src/main.py" caption="The argmax is wrong here, and sampled completions verify only 16% of the time. The reward is a Python function, not a learned model."
<span class="header">MODULE 7: GRPO with a verifiable reward</span>
TinyGPT: 4 layers, 4 heads, width 128
Policy parameters: 818,560   (reference is a frozen copy)
Task: reverse a string, verified by a Python function (no reward model)
Train prompts: 256   Held-out prompts: 40
Group size G=8, temperature=1.0, beta(KL)=0.01, lr=0.0001

<span class="header">BEFORE GRPO</span>
  Held-out accuracy, sampled (temp 1.0): 15.9%   &lt;- what GRPO optimizes
  Held-out accuracy, greedy (argmax):     22.5%
  sample: 'reverse: sukgh' -&gt; 'hgkuk'  (want 'hgkus': wrong)
:::

---

:::terminal id="exercise-output-after" title="The Reward Climbs, the Policy Improves" cmd="uv run python module_07_rl/src/main.py" caption="Sampled accuracy rises from 16% to 73% and greedy from 22% to 92%, driven by reward, not imitation. The same prompt now reverses correctly."
<span class="header">GRPO TRAINING</span>
  step   mean reward
    20         0.147
    80         0.280
   140         0.452
   200         0.655
   260         0.717
   320         0.839
   380         0.912
   400         0.955

<span class="header">AFTER GRPO</span>
  <span class="success">Held-out accuracy, sampled (temp 1.0): 73.1%   (was 15.9%)</span>
  <span class="success">Held-out accuracy, greedy (argmax):     92.5%   (was 22.5%)</span>
  <span class="success">sample: 'reverse: sukgh' -&gt; 'hgkus'  (want 'hgkus': correct)</span>
:::

---

<!-- .slide: id="exercise-reward-curve" -->

## The Reward Curve

<div class="curve-figure">
  <img src="images/reward_curve.png" alt="Mean group reward climbing from 0.15 to 0.96 over 400 GRPO steps">
</div>

Reward climbs as the policy concentrates probability on reversals it could already occasionally sample. Dashed lines: held-out accuracy before and after. (Actual exercise output.) <!-- .element: class="text-lg" style="margin-top: 10px;" -->

---

<!-- .slide: id="exercise-extra-credit" -->

## Extra Credit

- **KL-penalty ablation.** Set `BETA = 0`: reward climbs while completions drift into gibberish that games the verifier. The reward-hacking demo, reproduced.
- **Group-size sweep.** Vary `GROUP_SIZE`: a larger group gives a less noisy advantage estimate at higher cost.
- **Rejection-sampling baseline.** Keep only the best completion per group and SFT on it (Best-of-N). Compare to GRPO: what does the **negative signal** buy?
- **pass@k probe.** Track pass@1 and pass@k across training. Does GRPO raise pass@1 while pass@k stays flat? The Yue et al. claim, in miniature. <!-- .element: class="text-lg" style="margin-top: 10px;" -->
