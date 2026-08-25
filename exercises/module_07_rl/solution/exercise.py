"""
Module 7 Solution: GRPO on the instruct model with a verifiable reward

Complete reference implementation of the ten GRPO pieces: sample a group of
completions, score each with a programmatic verifier, turn the rewards into
group-relative advantages, compute per-token log-probabilities under the policy and
the frozen reference, build the policy-gradient loss plus a KL penalty, and take one
optimizer step.

The model (`src/model.py`), tokenizer (`src/tokenizer.py`), dataset
(`src/data.py`), plotting (`src/visualization.py`), and runner (`src/main.py`) are
all provided. This file is the only one a student edits.
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
    provided sampler (`model.generate`); call it once per group member, each time at
    `temperature` so the group is diverse.

    Args:
        policy: The model being trained (the current policy).
        prompt_ids: Token ids of the prompt (the generation prefix), shape (1, P).
        group_size: How many completions to draw (G).
        max_new_tokens: How many tokens to generate per completion.
        block_size: The model's context length.
        temperature: Sampling temperature (higher = more diverse).
        generate_fn: The provided autoregressive sampler.
        generator: A torch.Generator for reproducible sampling.

    Returns:
        A list of `group_size` tensors, each the full prompt+completion ids of one sample.
    """
    return [
        generate_fn(policy, prompt_ids, max_new_tokens, block_size,
                    temperature=temperature, generator=generator)[0]
        for _ in range(group_size)
    ]


# ---------------------------------------------------------------------------
# Step 2: The verifiable reward
# ---------------------------------------------------------------------------


def verifiable_reward(response: str, target: str) -> float:
    """Return 1.0 if the response exactly matches the verified answer, else 0.0.

    This is the whole point of RLVR: no human label and no learned reward model, just
    a deterministic check. For "reverse: cat" the runner passes target="tac".

    Args:
        response: The text the model produced (already stripped of markers).
        target: The known-correct answer.

    Returns:
        1.0 for a correct completion, 0.0 otherwise.
    """
    return 1.0 if response == target else 0.0


# ---------------------------------------------------------------------------
# Step 3: Score every completion in the group
# ---------------------------------------------------------------------------


def score_group(responses: list[str], target: str) -> torch.Tensor:
    """Apply the verifiable reward to every completion, returning a reward vector.

    Args:
        responses: The decoded response text of each group member.
        target: The known-correct answer for this prompt.

    Returns:
        A 1-D float tensor of length G holding each completion's reward.
    """
    return torch.tensor([verifiable_reward(r, target) for r in responses])


# ---------------------------------------------------------------------------
# Step 4: Group-relative advantages (the GRPO baseline)
# ---------------------------------------------------------------------------


def group_relative_advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Standardize the group's rewards into advantages: (r - mean) / (std + eps).

    The group mean is GRPO's baseline (the role PPO's value network plays): a
    completion is "good" only relative to its peers on the same prompt. Subtracting
    the mean keeps the gradient unbiased while cutting variance; dividing by the std
    keeps the update scale stable across easy and hard prompts. The eps avoids a
    divide-by-zero when every completion scored the same.

    Args:
        rewards: The group's reward vector, shape (G,).
        eps: Small constant for numerical stability.

    Returns:
        A 1-D tensor of advantages, shape (G,).
    """
    return (rewards - rewards.mean()) / (rewards.std() + eps)


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

    Args:
        prompt_len: Number of prompt tokens (the generation prefix length).
        seq_len: Total length of the prompt+completion sequence.

    Returns:
        A 1-D bool tensor of length seq_len - 1, True at completion positions.
    """
    return torch.arange(seq_len - 1) >= (prompt_len - 1)


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
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Step 7: The policy-gradient loss for one completion
# ---------------------------------------------------------------------------


def pg_loss(token_log_probs: torch.Tensor, advantage: float, mask: torch.Tensor) -> torch.Tensor:
    """Advantage-weighted negative log-probability over the completion tokens.

    REINFORCE in one line: push up the log-prob of a completion in proportion to its
    advantage. A positive advantage (better than the group) makes the loss reward
    raising those tokens' probability; a negative advantage pushes them down. The
    mask restricts the sum to generated tokens (prompt positions contribute nothing).
    We negate because optimizers minimize.

    Args:
        token_log_probs: Per-token log-probs under the policy, shape (T,).
        advantage: This completion's scalar group-relative advantage.
        mask: Bool completion mask, shape (T,).

    Returns:
        A scalar loss tensor for this completion.
    """
    return -advantage * (token_log_probs * mask).sum()


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

    Args:
        policy_log_probs: Per-token log-probs under the policy, shape (T,).
        ref_log_probs: Per-token log-probs under the frozen reference, shape (T,).
        mask: Bool completion mask, shape (T,).

    Returns:
        A scalar KL estimate for this completion.
    """
    return ((policy_log_probs - ref_log_probs) * mask).sum()


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
    penalties into one scalar `loss`. This is the same train-step shape as Module 6:
    zero the gradients, backpropagate, then (provided) clip and step.

    Args:
        optimizer: The optimizer over the policy parameters.
        loss: The combined GRPO loss for this group.
        model: The policy (for gradient clipping).
        grad_clip: Max global gradient norm.

    Returns:
        The loss value as a plain Python float.
    """
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

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

    Args:
        rewards: A 1-D reward tensor.

    Returns:
        The mean reward as a Python float.
    """
    return rewards.mean().item()
