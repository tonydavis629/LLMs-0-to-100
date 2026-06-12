"""
Module 5 Exercise: Pretraining NanoGPT

Train a tiny decoder-only language model FROM SCRATCH on a bundled text file.
The model architecture is provided (src/model.py) — your job is the PRETRAINING
LOOP that turns random weights into useful ones.

Fill in each line marked with `raise NotImplementedError(...)`. Each blank needs
only one expression or one short block. Run after each step; the runner skips any
step you have not finished yet:

    uv run python module_05_pretraining/src/main.py
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Step 1: Encode text into token IDs
# ---------------------------------------------------------------------------


def encode(text: str, stoi: dict[str, int]) -> torch.Tensor:
    """Turn a string into a 1-D LongTensor of token IDs.

    Args:
        text: The raw text.
        stoi: A "string-to-index" map from each character to its integer ID.

    Returns:
        A 1-D tensor of dtype torch.long with one ID per character.
    """
    # TODO: Return a 1-D LongTensor with one integer ID per character in `text`.
    # HINT: look up each character in stoi, then wrap the list with torch.tensor(..., dtype=torch.long).
    raise NotImplementedError("TODO: encode text into a LongTensor of token IDs")


# ---------------------------------------------------------------------------
# Step 2: Split the token stream into train and validation sets
# ---------------------------------------------------------------------------


def train_val_split(data: torch.Tensor,
                    val_fraction: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a 1-D token stream into a training prefix and a validation suffix.

    Args:
        data: The full 1-D tensor of token IDs.
        val_fraction: Fraction of tokens to hold out for validation.

    Returns:
        (train_data, val_data) as two 1-D tensors.
    """
    # TODO: Return (train_data, val_data): the first (1 - val_fraction) of the
    #       tokens for training, and the remaining tail for validation.
    # HINT: n_train = int(len(data) * (1 - val_fraction)), then slice data[:n_train] and data[n_train:].
    raise NotImplementedError("TODO: split the token stream into train and validation")


# ---------------------------------------------------------------------------
# Step 3: Build one batch of inputs x and shifted targets y
# ---------------------------------------------------------------------------


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of contexts and their next-token targets.

    For each example we pick a random start index i and take a block of
    `block_size` tokens as the input x. The target y is that SAME block shifted
    one position to the left: the token the model should predict at each step.

        x = data[i      : i + block_size]
        y = data[i + 1  : i + block_size + 1]

    Args:
        data: A 1-D tensor of token IDs to sample from.
        block_size: Context length (number of tokens per example).
        batch_size: How many examples to stack into the batch.
        generator: Optional RNG for reproducible sampling.

    Returns:
        (x, y), each of shape (batch_size, block_size).
    """
    # Random start indices, leaving room for the +1 shift on y.
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in ix])

    # TODO: Build y, the targets: the same blocks as x but shifted one step left,
    #       so y[:, t] is the token that should follow x[:, t].
    # HINT: mirror the line that builds x, but start each slice one index later (i + 1 ... i + 1 + block_size).
    y = None
    if y is None:
        raise NotImplementedError("TODO: build the shifted target batch y")
    return x, y


# ---------------------------------------------------------------------------
# Step 4: Cross-entropy loss from logits and targets
# ---------------------------------------------------------------------------


def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Average cross-entropy between predicted logits and the true next tokens.

    Args:
        logits: Model outputs, shape (batch, time, vocab_size).
        targets: True next-token IDs, shape (batch, time).

    Returns:
        A scalar loss tensor (negative log-likelihood averaged over all tokens).
    """
    batch_size, seq_len, vocab_size = logits.shape
    # TODO: Return the average cross-entropy between logits and targets.
    # HINT: use .view to flatten logits to (batch_size * seq_len, vocab_size) and targets to (batch_size * seq_len,), then F.cross_entropy.
    raise NotImplementedError("TODO: cross-entropy loss from logits and targets")


# ---------------------------------------------------------------------------
# Step 5: One optimizer step
# ---------------------------------------------------------------------------


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    grad_clip: float = 1.0,
) -> float:
    """Run one training step and return the scalar loss value.

    Args:
        model: The model to train.
        optimizer: The optimizer holding the model's parameters.
        x: Input token IDs, shape (batch, time).
        y: Target token IDs, shape (batch, time).
        grad_clip: Max global gradient norm (clipping is provided below).

    Returns:
        The loss for this batch as a plain Python float.
    """
    logits = model(x)
    loss = compute_loss(logits, y)

    # TODO: Clear last step's gradients, then backpropagate this step's loss.
    # HINT: gradients accumulate across steps unless cleared — the optimizer has a method to reset them, and the loss tensor has a method that backpropagates.
    raise NotImplementedError("TODO: clear old gradients and backpropagate the loss")

    # Provided: clip the global gradient norm for stability, then take the step.
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return loss.item()


# ---------------------------------------------------------------------------
# Learning-rate schedule (provided): warmup, then cosine decay
# ---------------------------------------------------------------------------


def lr_at_step(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Return the learning rate for a given step: warmup then cosine decay.

    Args:
        step: The current training step (0-indexed).
        warmup_steps: Steps spent linearly ramping the LR up to max_lr.
        max_steps: Total steps; LR reaches min_lr at the end.
        max_lr: Peak learning rate (reached at the end of warmup).
        min_lr: Final/minimum learning rate.

    Returns:
        The learning rate to use at this step.
    """
    # 1) Warmup: ramp linearly from ~0 up to max_lr.
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    # 2) After the schedule ends, hold at the floor.
    if step > max_steps:
        return min_lr
    # 3) Cosine decay from max_lr down to min_lr.
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# ---------------------------------------------------------------------------
# Step 7: Estimate average loss over several batches
# ---------------------------------------------------------------------------


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    n_batches: int,
    generator: torch.Generator | None = None,
) -> float:
    """Average the loss over several random batches (a less noisy estimate).

    Args:
        model: The model to evaluate.
        data: The token stream to sample evaluation batches from.
        block_size: Context length.
        batch_size: Examples per batch.
        n_batches: How many batches to average over.
        generator: Optional RNG for reproducible sampling.

    Returns:
        The mean loss across the batches, as a float.
    """
    model.eval()
    total = 0.0
    for _ in range(n_batches):
        x, y = get_batch(data, block_size, batch_size, generator)
        logits = model(x)
        # TODO: Add this batch's loss (as a Python float) to `total`.
        # HINT: compute_loss(logits, y) returns a tensor; use .item() to get a float.
        raise NotImplementedError("TODO: accumulate the batch loss into total")
    model.train()
    return total / n_batches


# ---------------------------------------------------------------------------
# Step 8: Convert loss into perplexity and bits per token
# ---------------------------------------------------------------------------


def loss_to_perplexity_and_bits(loss: float) -> tuple[float, float]:
    """Convert an average cross-entropy loss (in nats) into two readouts.

    Perplexity is exp(loss): roughly the effective number of equally likely
    next-token choices. Bits per token is loss / ln(2): the same loss expressed
    in Shannon's units, i.e. the average number of bits to encode each token.

    Args:
        loss: Average cross-entropy loss in nats.

    Returns:
        (perplexity, bits_per_token).
    """
    # TODO: Return (perplexity, bits_per_token) from the average loss (in nats).
    # HINT: perplexity = exp(loss); bits_per_token = loss / ln(2). Use math.exp and math.log.
    raise NotImplementedError("TODO: perplexity and bits per token")


# ---------------------------------------------------------------------------
# Step 10: Generate text by sampling one token at a time
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    block_size: int,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Autoregressively extend `idx` by sampling `max_new_tokens` new tokens.

    Each step: crop to the last block_size tokens, run the model, take the
    logits at the final position, divide by temperature, softmax into a
    probability distribution, and sample one token from it.

    Args:
        model: The model to sample from.
        idx: Seed token IDs, shape (batch, time).
        max_new_tokens: How many tokens to append.
        block_size: Maximum context the model can attend to.
        temperature: Softmax temperature (lower is greedier).
        generator: Optional RNG for reproducible sampling.

    Returns:
        The extended sequence, shape (batch, time + max_new_tokens).
    """
    model.eval()
    for _ in range(max_new_tokens):
        # The model only has positions for block_size tokens, so crop the context.
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)
        # Take the last position's logits and scale by temperature.
        logits = logits[:, -1, :] / temperature

        # TODO: Turn these logits into probabilities and sample ONE token id.
        # HINT: use F.softmax over the -1 dimension, then torch.multinomial to sample (pass generator= for reproducibility).
        next_id = None
        if next_id is None:
            raise NotImplementedError("TODO: sample the next token id from the logits")

        idx = torch.cat([idx, next_id], dim=1)
    model.train()
    return idx
