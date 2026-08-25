"""
Module 6 Solution: Finetuning NanoGPT into an instruct model

Complete reference implementation: chat-template formatting, loss masking,
LoRA layer, adapter injection / freezing / merging, and the SFT training step.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Step 1: Format one prompt-response pair into a token stream
# ---------------------------------------------------------------------------


def format_example(
    prompt: str,
    response: str,
    special: dict[str, int],
    encode_fn,
) -> list[int]:
    """Assemble a chat-template token sequence for one prompt-response pair.

    The template is:
        [user] + encode(prompt) + [end] + [assistant] + encode(response) + [end]

    Args:
        prompt: The user's instruction text.
        response: The assistant's desired response text.
        special: Mapping from special token strings to their integer IDs.
        encode_fn: Function that maps a plain string to a list of token IDs.

    Returns:
        A flat list of token IDs representing the full formatted example.
    """
    return (
        [special["<|user|>"]]
        + encode_fn(prompt)
        + [special["<|end|>"]]
        + [special["<|assistant|>"]]
        + encode_fn(response)
        + [special["<|end|>"]]
    )


# ---------------------------------------------------------------------------
# Step 2: Build next-token targets with prompt positions masked
# ---------------------------------------------------------------------------


def build_targets(ids: list[int], prompt_span: int) -> list[int]:
    """Build next-token targets where prompt predictions are set to -100 (ignored).

    Target t is the token that position t should predict, i.e. ids[t + 1]. We keep
    only the predictions of the response tokens. The first response token is
    predicted at position prompt_span - 1 (the assistant marker), so positions
    0 .. prompt_span - 2 are masked, the response targets are ids[prompt_span:],
    and the final position is -100 (no token follows the last one).

    Args:
        ids: The full formatted token sequence.
        prompt_span: Number of leading tokens that belong to the prompt (user turn + markers).

    Returns:
        A list of target IDs the same length as ids.
    """
    return [-100] * (prompt_span - 1) + ids[prompt_span:] + [-100]


# ---------------------------------------------------------------------------
# Step 3: Masked cross-entropy loss
# ---------------------------------------------------------------------------


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Average cross-entropy over the response tokens only (-100 positions are ignored).

    Args:
        logits: Model outputs, shape (batch, time, vocab_size).
        targets: Target IDs with -100 for prompt positions, shape (batch, time).

    Returns:
        Scalar loss tensor.
    """
    vocab_size = logits.shape[-1]
    return F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), ignore_index=-100)


# ---------------------------------------------------------------------------
# Step 4: Build an optimizer over adapters only
# ---------------------------------------------------------------------------


def build_optimizer(model: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
    """Return an AdamW optimizer over only the trainable (adapter) parameters.

    Args:
        model: The model with LoRA injected and base weights frozen.
        lr: The small finetuning learning rate.

    Returns:
        A torch.optim.Optimizer configured for the adapter parameters.
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(trainable, lr=lr)


# ---------------------------------------------------------------------------
# Step 5: LoRA forward delta (filled inside LoRALinear in src/model.py)
# ---------------------------------------------------------------------------


def lora_forward_delta(
    x: torch.Tensor,
    A: torch.nn.Parameter,
    B: torch.nn.Parameter,
    scale: float,
    dropout: torch.nn.Module,
) -> torch.Tensor:
    """Compute the low-rank update: scale * (dropout(x) @ A.t() @ B.t()).

    This is the expression the student fills inside LoRALinear.forward.
    It is extracted here so the step list is complete.
    """
    return scale * (dropout(x) @ A.t() @ B.t())


# ---------------------------------------------------------------------------
# Step 6: Freeze base parameters (filled inside freeze_base_ in src/model.py)
# ---------------------------------------------------------------------------


def freeze_base_param(p: torch.nn.Parameter) -> None:
    """Set requires_grad=False on a single parameter. Student step 6."""
    p.requires_grad = False


# ---------------------------------------------------------------------------
# Step 7: One SFT training step
# ---------------------------------------------------------------------------


def sft_train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    grad_clip: float = 1.0,
) -> float:
    """Run one supervised-finetuning step and return the scalar loss.

    Args:
        model: The model (with LoRA adapters).
        optimizer: The optimizer over adapter parameters.
        x: Input token IDs, shape (batch, time).
        y: Target IDs with -100 for prompt positions, shape (batch, time).
        grad_clip: Max global gradient norm.

    Returns:
        The loss for this batch as a plain Python float.
    """
    logits = model(x)
    loss = masked_cross_entropy(logits, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return loss.item()


# ---------------------------------------------------------------------------
# Step 8: Count trainable parameters
# ---------------------------------------------------------------------------


def count_trainable_params(model: torch.nn.Module) -> int:
    """Count parameters where requires_grad is True."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Step 9: Build a generation prompt (assistant turn prefix)
# ---------------------------------------------------------------------------


def build_generation_prompt(
    prompt: str,
    special: dict[str, int],
    encode_fn,
) -> list[int]:
    """Assemble the token stream up to (and including) the assistant marker.

    Template:
        [user] + encode(prompt) + [end] + [assistant]

    Args:
        prompt: The user's instruction text.
        special: Mapping from special token strings to their integer IDs.
        encode_fn: Function mapping a plain string to a list of token IDs.

    Returns:
        A flat list of token IDs ready for autoregressive generation.
    """
    return (
        [special["<|user|>"]]
        + encode_fn(prompt)
        + [special["<|end|>"]]
        + [special["<|assistant|>"]]
    )


# ---------------------------------------------------------------------------
# Step 10: Merge LoRA weight
# ---------------------------------------------------------------------------


def merge_lora_weight(base_W: torch.Tensor, A: torch.Tensor, B: torch.Tensor, scale: float) -> torch.Tensor:
    """Return base_W + scale * (B @ A).

    Args:
        base_W: The frozen pretrained weight matrix.
        A: LoRA A matrix (r x in).
        B: LoRA B matrix (out x r).
        scale: alpha / r.

    Returns:
        The merged weight matrix of the same shape as base_W.
    """
    return base_W + scale * (B @ A)
