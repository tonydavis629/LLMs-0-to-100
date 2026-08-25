"""
Module 7 Exercise: GRPO on the instruct model with a verifiable reward

You implement the ten GRPO pieces below. Everything else (the TinyGPT model, the
sampler, the tokenizer, the verifiable-task data, the plotting, and the runner) is
provided. Each blank is one line or one short expression.

The goal is not a useful model; it is to make the RL loop visible: sample a group of
completions, score each with a Python verifier, turn rewards into group-relative
advantages, and take one policy-gradient step that pushes up the winners and down the
losers. The payoff is a reward curve that climbs and a held-out accuracy that rises.

Run after each step; unfinished steps are skipped automatically:
    uv run python module_07_rl/src/main.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Step 1: Sample a group of completions from the current policy
# ---------------------------------------------------------------------------


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
    # HINT: a list comprehension over range(group_size); index [0] to drop the batch dim.
    raise NotImplementedError("TODO: sample a group of completions from the policy")


# ---------------------------------------------------------------------------
# Step 2: The verifiable reward
# ---------------------------------------------------------------------------


def verifiable_reward(response: str, target: str) -> float:
    """Return 1.0 if the response exactly matches the verified answer, else 0.0.

    This is the whole point of RLVR: no human label and no learned reward model, just
    a deterministic check. For "reverse: cat" the runner passes target="tac".
    """
    # TODO: Return 1.0 when response equals target, otherwise 0.0.
    # HINT: a single comparison; return a float.
    raise NotImplementedError("TODO: return 1.0 for an exact match else 0.0")


# ---------------------------------------------------------------------------
# Step 3: Score every completion in the group
# ---------------------------------------------------------------------------


def score_group(responses: list[str], target: str) -> torch.Tensor:
    """Apply the verifiable reward to every completion, returning a reward vector.

    Returns:
        A 1-D float tensor of length G holding each completion's reward.
    """
    # TODO: Return a tensor of verifiable_reward(r, target) for each r in responses.
    # HINT: build a Python list with a comprehension, wrap it in torch.tensor(...).
    raise NotImplementedError("TODO: score every completion into a reward vector")


# ---------------------------------------------------------------------------
# Step 4: Group-relative advantages (the GRPO baseline)
# ---------------------------------------------------------------------------


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
    # HINT: rewards.mean() and rewards.std(); add eps to the std before dividing.
    raise NotImplementedError("TODO: standardize rewards into group-relative advantages")


# ---------------------------------------------------------------------------
# Step 5: The completion mask (which tokens we train on)
# ---------------------------------------------------------------------------


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
    # HINT: torch.arange(seq_len - 1) gives the positions; compare it to prompt_len - 1.
    raise NotImplementedError("TODO: build the completion mask over target positions")


# ---------------------------------------------------------------------------
# Step 6: Per-token log-probabilities (used for both policy and reference)
# ---------------------------------------------------------------------------


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
    # HINT: F.log_softmax(logits, dim=-1), then .gather(-1, target_ids.unsqueeze(-1))
    #       and .squeeze(-1) to drop the gathered dimension.
    raise NotImplementedError("TODO: gather the per-token log-probabilities")


# ---------------------------------------------------------------------------
# Step 7: The policy-gradient loss for one completion
# ---------------------------------------------------------------------------


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
    # HINT: multiply token_log_probs by mask, .sum() it, multiply by -advantage.
    raise NotImplementedError("TODO: build the advantage-weighted policy-gradient loss")


# ---------------------------------------------------------------------------
# Step 8: The KL-to-reference penalty for one completion
# ---------------------------------------------------------------------------


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
    # HINT: subtract the two log-prob tensors, multiply by mask, then .sum().
    raise NotImplementedError("TODO: build the KL-to-reference penalty")


# ---------------------------------------------------------------------------
# Step 9: One GRPO optimizer step
# ---------------------------------------------------------------------------


def grpo_step(
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor,
    model: torch.nn.Module,
    grad_clip: float = 1.0,
) -> float:
    """Clear gradients, backpropagate the combined GRPO loss, and step.

    The runner has already summed the per-completion policy-gradient losses and KL
    penalties into one scalar `loss`. Same train-step shape as Module 6: zero the
    gradients, backpropagate, then (provided) clip and step.

    Returns:
        The loss value as a plain Python float.
    """
    # TODO: Clear last step's gradients, then backpropagate this step's loss.
    # HINT: the optimizer has a method to zero gradients (use set_to_none=True);
    #       the loss tensor has a method that backpropagates.
    raise NotImplementedError("TODO: zero the gradients and backpropagate the loss")

    # Provided: clip the global gradient norm for stability, then take the step.
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return loss.item()


# ---------------------------------------------------------------------------
# Step 10: Mean group reward (the training-curve metric)
# ---------------------------------------------------------------------------


def mean_reward(rewards: torch.Tensor) -> float:
    """Average reward over a vector of rewards, as a plain float.

    The runner records this each step to plot the reward curve, and reuses it on the
    held-out prompts to report before/after accuracy.
    """
    # TODO: Return the mean of rewards as a Python float.
    # HINT: rewards.mean() gives a tensor; .item() converts it to a float.
    raise NotImplementedError("TODO: return the mean reward as a float")
