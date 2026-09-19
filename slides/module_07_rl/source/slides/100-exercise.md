:::divider id="divider-exercise" title="Exercise" sub="GRPO with a verifiable reward"
:::

---

<!-- .slide: id="exercise-run" -->

## Running the Exercise

Open `module_07_rl/exercise.py`, the only file you edit, and fill in the ten `NotImplementedError` lines. The model, sampler, tokenizer, task data, and runner are provided.

<!-- .element: class="text-lg" -->

```bash
# Run every step; each is tagged CORRECT, INCORRECT, or INCOMPLETE
cd exercises
uv run python module_07_rl/src/main.py

# Run a single step (1-10, or train for the GRPO training run)
uv run python module_07_rl/src/main.py --step 4
```

`data/instruct_model.pt` ships with the repo: the Module 6 result, built for you by `src/make_instruct_checkpoint.py`. Nothing is downloaded.

<!-- .element: class="text-md" style="margin-top: 22px;" -->

After Step 10 the runner trains the policy with GRPO. It prints held-out accuracy **before and after** (sampled and greedy) and the mean reward every 20 steps, and it saves a **reward-curve image**.

<!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="exercise-results" -->

## Reading the Results

For each step the runner prints what your function produced on the real model (a sampled group, its rewards and advantages, a mask, per-token log-probs), then runs the step's **tests** from `tests/`. Each test calls your function on small tensors whose correct answer is known.

<!-- .element: class="text-lg" -->

- **CORRECT**: every test for the step passed
- **INCORRECT**: your code ran but a test failed; the expected and actual values are printed beneath
- **INCOMPLETE**: the function still raises `NotImplementedError`

The tag sits on the step's header line. Policy-gradient bugs rarely crash: the reward curve just stalls or falls, and you notice minutes into training. The tests catch sign and masking mistakes in seconds. <!-- .element: class="text-lg" style="margin-top: 15px;" -->

---

:::terminal id="exercise-results-example" title="What an INCORRECT Step Looks Like" cmd="uv run python module_07_rl/src/main.py --step 7" maxw="1000px" caption="Here <code>pg_loss()</code> dropped the minus sign. The mask is right, so that test passes, but the loss and its gradient both come out with the wrong sign: a descent step would lower the winner's log-probs. The first message says where to look."
<span class="header">=== Step 7: pg_loss() ===</span> <span class="t-fail">INCORRECT</span>
  <span class="t-fail">INCORRECT</span>  log-probs [-1, -2, -3], mask [F, T, T], advantage 2 give loss 10.0
             expected 10.0, got -10.0000 (optimizers minimize, so negate: -advantage * sum)
  <span class="t-fail">INCORRECT</span>  descent raises a winner's log-probs: advantage 2 gives d loss / d log p = [0, -2, -2]
             expected [0.0, -2.0, -2.0], got [0.0, 2.0, 2.0] (descent would push the winner's tokens down)
  <span class="success">CORRECT</span>    masked prompt positions contribute nothing to the loss
:::

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

Sample a group, write the verifier, score the group, standardize advantages, mask the completion, gather log-probs, build the policy-gradient loss, add the KL penalty, take a step, track reward. Every step has its own tests in `tests/`.
:::

---

:::step id="exercise-step1" title="Step 1: sample_group()"
```python
def sample_group(
    policy,
    prompt_ids: torch.Tensor,
    group_size: int,
    max_new_tokens: int,
    block_size: int,
    temperature: float,
    generate_fn,
    generator: torch.Generator,
) -> list[torch.Tensor]:
    """Draw `group_size` completions for one prompt from the policy.

    GRPO scores a whole *group* of samples against each other, so the first move is
    to generate several completions for the same prompt. `generate_fn` is the
    provided sampler; call it once per group member, each at `temperature` so the
    group is diverse. Each call returns shape (1, L); take row [0].

    Returns:
        A list of `group_size` tensors, each the full prompt+completion ids of one sample.
    """
    # TODO: Return a list of `group_size` completions, each from generate_fn(policy,
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
    """Return 1.0 if the response exactly matches the verified answer, else 0.0.

    This is the whole point of RLVR: no human label and no learned reward model, just
    a deterministic check. For "reverse: cat" the runner passes target="tac".
    """
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
def score_group(responses: list[str], target: str) -> torch.Tensor:
    """Apply the verifiable reward to every completion, returning a reward vector.

    Returns:
        A 1-D float tensor of length G holding each completion's reward.
    """
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
def group_relative_advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Standardize the group's rewards into advantages: (r - mean) / (std + eps).

    The group mean is GRPO's baseline (the role PPO's value network plays): a
    completion is "good" only relative to its peers on the same prompt. Subtracting
    the mean keeps the gradient unbiased while cutting variance; dividing by the std
    keeps the update scale stable. The eps avoids a divide-by-zero when every
    completion scored the same.

    Returns:
        A 1-D tensor of advantages, shape (G,).
    """
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

:::terminal id="exercise-output-group" title="Steps 1&ndash;4: One Group, Scored" cmd="uv run python module_07_rl/src/main.py" maxw="1000px" caption="The greedy answer to 'jphkq' is right, yet only two of eight samples verify. The group mean is 0.25, so each winner gets advantage +1.62 and each loser -0.54. The rare successes are pushed up hard and the failures are pushed down gently."
<span class="header">=== Step 1: sample_group() ===</span> <span class="success">CORRECT</span>
Held-out prompt 'reverse: jphkq'; the verifier wants 'qkhpj'
A group of G=8 completions sampled at temperature 1.0:
  'qkhpV'  'qkhpp'  'qkhpq'  'qkhpb'  <span class="success">'qkhpj'</span>  'qkhpp'  <span class="success">'qkhpj'</span>  'qkhpq'
<span class="skipped">  ...</span>
<span class="header">=== Step 2: verifiable_reward() ===</span> <span class="success">CORRECT</span>
The starting model's greedy answers on 4 held-out prompts:
  'reverse: jphkq' -&gt; 'qkhpj'  want 'qkhpj'  reward 1.0
  'reverse: sukgh' -&gt; 'hgkuk'  want 'hgkus'  reward 0.0
  'reverse: nlfqb' -&gt; 'bqfnl'  want 'bqfln'  reward 0.0
  'reverse: gzyps' -&gt; 'spyzg'  want 'spyzg'  reward 1.0
<span class="skipped">  ...</span>
<span class="header">=== Step 3: score_group() ===</span> <span class="success">CORRECT</span>
Rewards for the Step 1 group: [0, 0, 0, 0, 1, 0, 1, 0]
<span class="skipped">  ...</span>
<span class="header t-green">=== Step 4: group_relative_advantages() ===</span> <span class="success">CORRECT</span>
Rewards for the Step 1 group: [ 0.00,  0.00,  0.00,  0.00,  1.00,  0.00,  1.00,  0.00]
Advantages:                   [-0.54, -0.54, -0.54, -0.54, <span class="success">+1.62</span>, -0.54, <span class="success">+1.62</span>, -0.54]
  <span class="success">CORRECT</span>    rewards [1, 0, 1, 0] give [0.866, -0.866, 0.866, -0.866] (mean 0.5, std 0.577)
  <span class="success">CORRECT</span>    advantages have mean 0 and std 1 within a group: rewards [1, 0, 0, 0, 1, 0, 0, 0]
  <span class="success">CORRECT</span>    a group where every reward ties gives all-zero advantages, not NaN: [1, 1, 1, 1]

<span class="header">=== Step 5: completion_mask() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">TODO: build the completion mask over target positions</span>
:::

---

:::step id="exercise-step5" title="Step 5: completion_mask()"
```python
def completion_mask(prompt_len: int, seq_len: int) -> torch.Tensor:
    """Mark which next-token-prediction positions belong to the completion.

    A sequence is prompt + completion. We score predictions at positions 0..seq_len-2
    (each predicts the next token). Position t predicts token t+1, which is a
    generated token only when t + 1 >= prompt_len. So positions t >= prompt_len - 1
    are completion positions; the rest are prompt context we must NOT train on. This
    is the credit-assignment mask, the RL cousin of Module 6's loss mask.

    Returns:
        A 1-D bool tensor of length seq_len - 1, True at completion positions.
    """
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
def gather_token_log_probs(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Log-probability the model assigns to each actually-taken token.

    Softmax the logits into a distribution, take the log, then pick out the entry for
    the token that was really sampled at each position. The runner calls this twice
    per completion: once on the policy (with gradients) and once on the frozen
    reference (under no_grad). Same gather, different model.

    Args:
        logits: Model outputs for the input positions, shape (T, vocab_size).
        target_ids: The token id actually taken at each position, shape (T,).

    Returns:
        A 1-D tensor of per-token log-probabilities, shape (T,).
    """
    # TODO: Return the log-probability of each target token: log_softmax the logits
    #       over the vocab dimension, then gather the entry at each target id.
    raise NotImplementedError("TODO: gather the per-token log-probabilities")
```
+++
**Hint:** take `F.log_softmax` over the vocabulary dimension, then `.gather` along that same dimension using `target_ids`; `unsqueeze` gives the index the extra dim that `gather` needs, and `squeeze` drops it afterwards.
+++
**Answer:**

```python
log_probs = F.log_softmax(logits, dim=-1)
return log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
```
:::

---

:::terminal id="exercise-output-mask" title="Steps 5&ndash;6: Mask and Log-Probs" cmd="uv run python module_07_rl/src/main.py" maxw="1000px" caption="Sixteen positions predict prompt tokens and are masked out; the six that predict the answer are trained on. The model is sure of 'hgk', less sure of 'u', and picks the wrong final 'k' with log-prob -1.00 (probability 0.37)."
<span class="header">=== Step 1: sample_group() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 2: verifiable_reward() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 3: score_group() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 4: group_relative_advantages() ===</span> <span class="success">CORRECT</span>
<span class="skipped">  ...</span>

<span class="header t-green">=== Step 5: completion_mask() ===</span> <span class="success">CORRECT</span>
Greedy answer to 'reverse: sukgh': 'hgkuk' + &lt;|end|&gt;, so 17 prompt tokens + 6 generated
Mask over the 22 next-token positions: 0000000000000000<span class="success">111111</span>
  <span class="success">CORRECT</span>    prompt_len=3, seq_len=6 gives [F, F, T, T, T]: position t predicts token t + 1
  <span class="success">CORRECT</span>    returns a bool tensor of length seq_len - 1: prompt_len=4, seq_len=10 gives length 9
  <span class="success">CORRECT</span>    one True per generated token: prompt_len=17, seq_len=23 has 6 True positions

<span class="header t-cyan">=== Step 6: gather_token_log_probs() ===</span> <span class="success">CORRECT</span>
Log-prob of each token of the greedy answer to 'reverse: sukgh' (want 'hgkus'):
  h -0.00   g -0.00   k -0.00   u -0.42   <span class="t-fail">k -1.00</span>   &lt;|end|&gt; 0.00
  <span class="success">CORRECT</span>    logits [[0, 0, 0], [0, ln 2, 0]] with targets [2, 1] give [ln 1/3, ln 1/2] = [-1.099, -0.693]
  <span class="success">CORRECT</span>    returns one log-prob per position: logits (5, 69) and targets (5,) give shape (5,)
  <span class="success">CORRECT</span>    matches -F.cross_entropy(logits, targets, reduction='none') on random (5, 69) logits

<span class="header">=== Step 7: pg_loss() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">TODO: build the advantage-weighted policy-gradient loss</span>
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
def pg_loss(token_log_probs: torch.Tensor, advantage: float, mask: torch.Tensor) -> torch.Tensor:
    """Advantage-weighted negative log-probability over the completion tokens.

    REINFORCE in one line: push up the log-prob of a completion in proportion to its
    advantage. A positive advantage (better than the group) makes the loss reward
    raising those tokens' probability; a negative advantage pushes them down. The
    mask restricts the sum to generated tokens. We negate because optimizers minimize.

    Returns:
        A scalar loss tensor for this completion.
    """
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
def kl_penalty(
    policy_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """A per-token estimate of KL(policy || reference) over the completion tokens.

    The reward only says "produce correct answers"; nothing stops the policy from
    drifting into degenerate text that happens to score. The KL term is the leash:
    summed over the generated tokens, log pi_policy - log pi_ref measures how far the
    policy has moved from the frozen reference, and the runner scales it by beta.

    Returns:
        A scalar KL estimate for this completion.
    """
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
def mean_reward(rewards: torch.Tensor) -> float:
    """Average reward over a vector of rewards, as a plain float.

    The runner records this each step to plot the reward curve, and reuses it on the
    held-out prompts to report before/after accuracy.
    """
    # TODO: Return the mean of rewards as a Python float.
    raise NotImplementedError("TODO: return the mean reward as a float")
```
+++
**Hint:** averaging a tensor gives a one-element tensor; `.item()` converts it to a float.
+++
**Answer:**

```python
return rewards.mean().item()
```
:::

---

:::terminal id="exercise-output-before" title="Before GRPO" cmd="uv run python module_07_rl/src/main.py" maxw="1000px" caption="All ten steps pass their tests, so training starts. Sampled completions verify only 15.9% of the time, and the argmax gets 'sukgh' wrong. The reward is a Python function, not a learned model."
<span class="header">=== Step 1: sample_group() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 2: verifiable_reward() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 3: score_group() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 4: group_relative_advantages() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 5: completion_mask() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 6: gather_token_log_probs() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 7: pg_loss() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 8: kl_penalty() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 9: grpo_step() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 10: mean_reward() ===</span> <span class="success">CORRECT</span>
<span class="skipped">  ...</span>

<span class="header t-green">=== GRPO training (Steps 1-10 together) ===</span> <span class="success">CORRECT</span>
TinyGPT: 4 layers, 4 heads, width 128, 818,560 parameters (the reference is a frozen copy)
Task: reverse a string, verified by a Python function (no reward model)
Train prompts: 256   Held-out prompts: 40
Group size G=8, temperature=1.0, beta(KL)=0.01, lr=0.0001

Before GRPO:
  Held-out accuracy, sampled (temp 1.0): 15.9%   &lt;- what GRPO optimizes
  Held-out accuracy, greedy (argmax):     22.5%
  <span class="t-fail">sample: 'reverse: sukgh' -&gt; 'hgkuk'  (want 'hgkus': wrong)</span>
:::

---

:::terminal id="exercise-output-after" title="The Reward Climbs, the Policy Improves" cmd="uv run python module_07_rl/src/main.py" maxw="1000px" caption="Sampled accuracy rises from 15.9% to 73.1% and greedy from 22.5% to 92.5%, driven by reward, not imitation. The same prompt now reverses correctly, and the training tests check that the reward curve climbed."
<span class="header t-green">=== GRPO training (Steps 1-10 together) ===</span> <span class="success">CORRECT</span>
<span class="skipped">  ...</span>
  step   mean reward
    20         0.147
    40         0.172
    60         0.270
<span class="skipped">   ...</span>
   200         0.655
<span class="skipped">   ...</span>
   360         0.875
   380         0.912
   400         0.955

After GRPO:
  <span class="success">Held-out accuracy, sampled (temp 1.0): 73.1%   (was 15.9%)</span>
  <span class="success">Held-out accuracy, greedy (argmax):     92.5%   (was 22.5%)</span>
  <span class="success">sample: 'reverse: sukgh' -&gt; 'hgkus'  (want 'hgkus': correct)</span>
  Reward curve saved to output/reward_curve.png
  <span class="success">CORRECT</span>    the mean group reward climbs by at least 0.3 from the first checkpoint to the last
  <span class="success">CORRECT</span>    held-out sampled accuracy rises by at least 20 points
  <span class="success">CORRECT</span>    held-out greedy accuracy does not fall
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
