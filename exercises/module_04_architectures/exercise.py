"""
Module 4 Exercise: LLM Architectures — Assemble GPT-2 and Generate Text

Wire attention into a complete decoder-only transformer, load real GPT-2
weights, and generate text. No training yet (that is Module 5); the goal is
to see a model you assembled produce real output.

Fill in the lines marked with `raise NotImplementedError(...)`.
Each blank needs only one expression or one short assignment.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Provided for you - see src/attention.py and src/sampling.py
from src.attention import CausalSelfAttention
from src.sampling import _sample_topk_token


# ---------------------------------------------------------------------------
# Step 1: Embedding lookup
# ---------------------------------------------------------------------------


class EmbeddingLayer(nn.Module):
    """Map token IDs to vectors and add positional embeddings."""

    def __init__(self, vocab_size: int, d_model: int, max_pos: int = 1024) -> None:
        super().__init__()
        self.d_model = d_model
        # One learned vector per vocabulary token.
        self.token_embed = nn.Embedding(vocab_size, d_model)
        # One learned vector per position in the sequence.
        self.pos_embed = nn.Embedding(max_pos, d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look up token and positional embeddings and add them.

        Args:
            token_ids: Integer token IDs, shape (batch_size, seq_len).

        Returns:
            Embeddings of shape (batch_size, seq_len, d_model).
        """
        b, t = token_ids.size()
        # Create a tensor [0, 1, 2, ..., t-1] for the positions in this batch.
        position = torch.arange(t, dtype=torch.long, device=token_ids.device)
        # Broadcast so every row in the batch gets the same position indices.
        position = position.unsqueeze(0).expand(b, t)

        # TODO: Look up token embeddings and position embeddings, then add them.
        # HINT: there are two embedding tables on self (one for tokens, one for positions); call each on its index tensor and add the two results.
        raise NotImplementedError("TODO: token embedding lookup + positional embedding")


# ---------------------------------------------------------------------------
# Step 2: The feed-forward network
# ---------------------------------------------------------------------------


class FeedForward(nn.Module):
    """Position-wise feed-forward network: linear, GELU, linear."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the two-layer FFN with GELU nonlinearity.

        Args:
            x: Input tensor, shape (batch_size, seq_len, d_model).

        Returns:
            Output tensor, shape (batch_size, seq_len, d_model).
        """
        # TODO: Apply fc1, then F.gelu(...), then fc2, then dropout.
        # HINT: compose the layers in order; the tensor should end at d_model, not d_ff.
        raise NotImplementedError("TODO: FFN forward pass")


# ---------------------------------------------------------------------------
# Step 3: A transformer block
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """One decoder block: pre-norm attention + pre-norm FFN, each with residual."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-norm transformer block with residual connections.

        Args:
            x: Input tensor, shape (batch_size, seq_len, d_model).

        Returns:
            Output tensor, shape (batch_size, seq_len, d_model).
        """
        # TODO: Pre-norm attention with residual, then pre-norm FFN with residual.
        # HINT: use the same pattern twice: normalize, apply the sub-layer, then add the result back to x.
        raise NotImplementedError("TODO: transformer block forward pass")


# ---------------------------------------------------------------------------
# Step 4: Stack the blocks and add the output head
# ---------------------------------------------------------------------------


class GPT2Model(nn.Module):
    """Complete decoder-only model."""

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        d_ff: int = 3072,
        max_pos: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        self.embed = EmbeddingLayer(vocab_size, d_model, max_pos)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        # The language-modeling head maps back to vocabulary-sized logits.
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Run a full forward pass from token IDs to logits.

        Args:
            token_ids: Integer token IDs, shape (batch_size, seq_len).

        Returns:
            Logits over the vocabulary, shape (batch_size, seq_len, vocab_size).
        """
        # TODO: embed -> run every block -> final layer norm -> LM head.
        # HINT:
        #   x = self.embed(token_ids)
        #   for block in self.blocks:
        #       x = block(x)
        #   return self.lm_head(self.ln_f(x))
        raise NotImplementedError("TODO: full GPT-2 forward pass")


# ---------------------------------------------------------------------------
# Step 5: Greedy decoding
# ---------------------------------------------------------------------------


def greedy_decode(model: GPT2Model, tokenizer, prompt: str, max_new: int = 10) -> str:
    """Generate text by always picking the most likely next token.

    Args:
        model: A GPT2Model with loaded weights.
        tokenizer: A HuggingFace GPT-2 tokenizer.
        prompt: The text prompt to condition generation on.
        max_new: How many new tokens to generate.

    Returns:
        The prompt plus the generated text as a single string.
    """
    model.eval()
    # Encode the prompt into token IDs; unsqueeze to add a batch dimension of 1.
    token_ids = tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        for _ in range(max_new):
            # Run a forward pass to get logits for every position.
            logits = model(token_ids)
            # The logits for the very last position predict the next token.
            next_logits = logits[:, -1, :]

            # TODO: Pick the token with the highest logit (argmax).
            # HINT: use torch.argmax(next_logits, dim=-1, keepdim=True)
            next_token = None
            if next_token is None:
                raise NotImplementedError("TODO: greedy argmax next token")

            # Append the new token to the sequence.
            token_ids = torch.cat([token_ids, next_token], dim=1)

    return tokenizer.decode(token_ids[0])


# ---------------------------------------------------------------------------
# Step 6: Temperature and top-k sampling
# ---------------------------------------------------------------------------


def sample_with_temperature_topk(
    model: GPT2Model,
    tokenizer,
    prompt: str,
    max_new: int = 10,
    temperature: float = 1.0,
    top_k: int = 50,
) -> str:
    """Generate text by sampling from a temperature-scaled, top-k truncated distribution.

    Args:
        model: A GPT2Model with loaded weights.
        tokenizer: A HuggingFace GPT-2 tokenizer.
        prompt: The text prompt to condition generation on.
        max_new: How many new tokens to generate.
        temperature: Temperature to divide logits by before softmax.
                     Lower = more conservative (peakier); higher = more random.
        top_k: Keep only the k most likely tokens before sampling.

    Returns:
        The prompt plus the generated text as a single string.
    """
    model.eval()
    token_ids = tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        for _ in range(max_new):
            logits = model(token_ids)
            next_logits = logits[:, -1, :]

            # TODO: Scale logits by temperature before top-k filtering.
            # HINT: temperature is applied before filtering; divide the logits by temperature.
            next_logits = None
            if next_logits is None:
                raise NotImplementedError("TODO: temperature scaling")

            # Use the provided helper to filter to top-k and sample one token.
            next_token = _sample_topk_token(next_logits, top_k)
            token_ids = torch.cat([token_ids, next_token], dim=1)

    return tokenizer.decode(token_ids[0])
