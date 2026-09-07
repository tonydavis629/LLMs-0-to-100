"""
Module 4 Solution: LLM Architectures — Assemble GPT-2 and Generate Text

Complete reference implementation of a decoder-only transformer that
loads real GPT-2 weights and generates text via greedy and sampling decoding.
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
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_pos, d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look up token and positional embeddings and add them."""
        b, t = token_ids.size()
        position = torch.arange(t, dtype=torch.long, device=token_ids.device)
        position = position.unsqueeze(0).expand(b, t)
        return self.token_embed(token_ids) + self.pos_embed(position)


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
        """Apply the two-layer FFN with GELU nonlinearity."""
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


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
        """Pre-norm transformer block with residual connections."""
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


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
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Run a full forward pass from token IDs to logits."""
        x = self.embed(token_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))


# ---------------------------------------------------------------------------
# Step 5: Greedy decoding
# ---------------------------------------------------------------------------


def greedy_decode(model: GPT2Model, tokenizer, prompt: str, max_new: int = 10) -> str:
    """Generate text by always picking the most likely next token."""
    model.eval()
    token_ids = tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        for _ in range(max_new):
            logits = model(token_ids)
            next_logits = logits[:, -1, :]
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
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
    """Generate text by sampling from a temperature-scaled, top-k truncated distribution."""
    model.eval()
    token_ids = tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        for _ in range(max_new):
            logits = model(token_ids)
            next_logits = logits[:, -1, :]

            # Scale logits by temperature before top-k filtering.
            next_logits = next_logits / temperature

            next_token = _sample_topk_token(next_logits, top_k)
            token_ids = torch.cat([token_ids, next_token], dim=1)

    return tokenizer.decode(token_ids[0])


# ---------------------------------------------------------------------------
# Extra credit: top-p (nucleus) sampling
# ---------------------------------------------------------------------------


def sample_topp(
    model: GPT2Model,
    tokenizer,
    prompt: str,
    max_new: int = 10,
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> str:
    """Generate text with nucleus (top-p) sampling."""
    model.eval()
    token_ids = tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        for _ in range(max_new):
            logits = model(token_ids)
            next_logits = logits[:, -1, :]
            next_logits = next_logits / temperature

            # Sort logits descending and compute cumulative probabilities
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumsum = torch.cumsum(sorted_probs, dim=-1)

            # Remove tokens where cumulative probability exceeds top_p
            mask = cumsum > top_p
            mask[:, 1:] = mask[:, :-1].clone()
            mask[:, 0] = False
            sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))

            # Scatter back to original index order
            full_logits = torch.empty_like(next_logits)
            full_logits.scatter_(-1, sorted_indices, sorted_logits)

            probs = F.softmax(full_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_ids = torch.cat([token_ids, next_token], dim=1)

    return tokenizer.decode(token_ids[0])
