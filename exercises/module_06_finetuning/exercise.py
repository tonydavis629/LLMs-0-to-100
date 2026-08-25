"""
Module 6 Exercise: Finetuning NanoGPT into an instruct model

You implement the ten finetuning pieces below. Everything else (the TinyGPT
model, the LoRA injection/merge plumbing, the tokenizer, the dataset, and the
runner) is provided. Each blank is one line or one short expression.

Run after each step; unfinished steps are skipped automatically:
    uv run python module_06_finetuning/src/main.py
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
    # TODO: Return the chat-template token ids:
    #       [user] + encode(prompt) + [end] + [assistant] + encode(response) + [end].
    # HINT: look up the marker ids in `special` (e.g. special["<|user|>"]) and call
    #       encode_fn(prompt) / encode_fn(response) for the text; join lists with +.
    raise NotImplementedError("TODO: assemble the chat-template token sequence")


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
    # TODO: Return next-token targets: the first prompt_span - 1 positions are -100
    #       (ignored), the response targets are ids[prompt_span:], and the final
    #       position is -100 (no token follows the last one).
    # HINT: [-100] * (prompt_span - 1) masks the prompt predictions; ids[prompt_span:]
    #       are the response targets; append one more -100 for the final position.
    raise NotImplementedError("TODO: build the masked next-token targets")


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
    # TODO: Return the average cross-entropy over the response tokens, ignoring -100.
    # HINT: flatten logits to (-1, vocab_size) and targets to (-1), then call
    #       F.cross_entropy with ignore_index=-100.
    raise NotImplementedError("TODO: masked cross-entropy with ignore_index=-100")


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
    # TODO: Return an AdamW optimizer over only the trainable (requires_grad) params.
    # HINT: collect [p for p in model.parameters() if p.requires_grad], then pass
    #       that list to torch.optim.AdamW with lr=lr.
    raise NotImplementedError("TODO: build AdamW over the trainable adapter params")


# ---------------------------------------------------------------------------
# Step 5: LoRA forward delta (called inside LoRALinear.forward in src/model.py)
# ---------------------------------------------------------------------------


def lora_forward_delta(
    x: torch.Tensor,
    A: torch.nn.Parameter,
    B: torch.nn.Parameter,
    scale: float,
    dropout: torch.nn.Module,
) -> torch.Tensor:
    """Compute the low-rank update added to the frozen layer's output.

    The update is:  scale * (dropout(x) @ A.t() @ B.t())

    Args:
        x: The layer input, shape (..., in_features).
        A: LoRA A matrix, shape (r, in_features).
        B: LoRA B matrix, shape (out_features, r).
        scale: alpha / r.
        dropout: A dropout (or Identity) module applied to x.

    Returns:
        The low-rank delta, shape (..., out_features).
    """
    # TODO: Return the low-rank update scale * (dropout(x) @ A.t() @ B.t()).
    # HINT: apply dropout to x, matrix-multiply by A.t() then B.t(), scale the result.
    raise NotImplementedError("TODO: compute the LoRA low-rank delta")


# ---------------------------------------------------------------------------
# Step 6: Freeze a base parameter (called inside freeze_base_ in src/model.py)
# ---------------------------------------------------------------------------


def freeze_base_param(p: torch.nn.Parameter) -> None:
    """Freeze a single base parameter so the optimizer never updates it."""
    # TODO: Freeze this parameter so it receives no gradient updates.
    # HINT: set the parameter's requires_grad attribute to False.
    raise NotImplementedError("TODO: freeze this base parameter")


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

    # TODO: Clear last step's gradients, then backpropagate this step's loss.
    # HINT: the optimizer has a method to zero gradients (use set_to_none=True);
    #       the loss tensor has a method that backpropagates.
    raise NotImplementedError("TODO: zero the gradients and backpropagate the loss")

    # Provided: clip the global gradient norm for stability, then take the step.
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return loss.item()


# ---------------------------------------------------------------------------
# Step 8: Count trainable parameters
# ---------------------------------------------------------------------------


def count_trainable_params(model: torch.nn.Module) -> int:
    """Count parameters where requires_grad is True."""
    # TODO: Return the number of parameters with requires_grad=True.
    # HINT: sum p.numel() over model.parameters() where p.requires_grad is True.
    raise NotImplementedError("TODO: count the trainable parameters")


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
    # TODO: Return the token ids up to the assistant marker (no response yet):
    #       [user] + encode(prompt) + [end] + [assistant].
    # HINT: same as format_example but stop right after special["<|assistant|>"].
    raise NotImplementedError("TODO: assemble the generation prompt up to the assistant marker")


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
    # TODO: Return the merged weight base_W + scale * (B @ A).
    # HINT: matrix-multiply B @ A (shape out x in), scale it, add to base_W.
    raise NotImplementedError("TODO: merge the LoRA update into the base weight")
