"""
Module 3 Solution: Attention Mechanisms

Complete reference implementation of a tiny attention layer,
causal masking, sinusoidal positional encodings, and a KV cache for
autoregressive generation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Step 1: Create token vectors for a tiny sequence
# ---------------------------------------------------------------------------


def make_token_vectors(vocab_size: int = 10, d_model: int = 8, seq_len: int = 5) -> torch.Tensor:
    """Create a fixed random embedding matrix and look up token vectors."""
    torch.manual_seed(42)
    E = torch.randn(vocab_size, d_model) * 0.1
    token_ids = torch.tensor([2, 5, 1, 8, 3])
    X = E[token_ids[:seq_len]]
    return X


# ---------------------------------------------------------------------------
# Steps 2-7: A tiny single-head attention layer
# ---------------------------------------------------------------------------


class TinyAttentionLayer:
    """A minimal attention layer with randomly initialized projection weights."""

    def __init__(self, d_model: int = 8, d_k: int = 4, seed: int = 42) -> None:
        self.d_model = d_model
        self.d_k = d_k
        torch.manual_seed(seed)
        self.W_Q = torch.randn(d_model, d_k) * 0.1
        self.W_K = torch.randn(d_model, d_k) * 0.1
        self.W_V = torch.randn(d_model, d_k) * 0.1

    def compute_qkv(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project the token matrix X into query, key, and value matrices."""
        Q = X @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V
        return Q, K, V

    def raw_attention_scores(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """Compute the pairwise compatibility scores: Q @ K^T."""
        return Q @ K.T

    def scaled_softmax(self, scores: torch.Tensor) -> torch.Tensor:
        """Scale scores by 1/sqrt(d_k) and apply softmax along the key dimension."""
        return F.softmax(scores / (self.d_k ** 0.5), dim=-1)

    def attention_output(self, weights: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Compute the attention output as a weighted sum of value vectors."""
        return weights @ V

    def causal_mask(self, seq_len: int) -> torch.Tensor:
        """Create a causal mask for autoregressive attention."""
        allowed = torch.tril(torch.ones(seq_len, seq_len))
        mask = allowed.masked_fill(allowed == 0, float("-inf"))
        mask = mask.masked_fill(mask == 1, 0.0)
        return mask

    def masked_attention(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled dot-product attention with a causal mask."""
        Q, K, V = self.compute_qkv(X)
        scores = self.raw_attention_scores(Q, K)
        scaled = scores / (self.d_k ** 0.5)
        masked_scores = scaled + self.causal_mask(X.shape[0])
        weights = F.softmax(masked_scores, dim=-1)
        output = self.attention_output(weights, V)
        return output, weights


# ---------------------------------------------------------------------------
# Step 8: Add sinusoidal positional encodings
# ---------------------------------------------------------------------------


def add_positional_embeddings(X: torch.Tensor) -> torch.Tensor:
    """Add sinusoidal positional encodings to the token representations."""
    seq_len, d_model = X.shape
    position = torch.arange(seq_len, dtype=X.dtype, device=X.device).unsqueeze(1)
    dim_pair = torch.arange(0, d_model, 2, dtype=X.dtype, device=X.device)
    angle_rates = 1 / (10000 ** (dim_pair / d_model))
    angles = position * angle_rates

    P = torch.zeros_like(X)
    P[:, 0::2] = torch.sin(angles)
    P[:, 1::2] = torch.cos(angles)
    return X + P


# ---------------------------------------------------------------------------
# Extra credit: simulate a KV cache for one-token-at-a-time generation
# ---------------------------------------------------------------------------


def kv_cache_step(
    new_token: torch.Tensor,
    cached_keys: torch.Tensor,
    cached_values: torch.Tensor,
    W_Q: torch.Tensor,
    W_K: torch.Tensor,
    W_V: torch.Tensor,
    d_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute attention for one new token using a KV cache."""
    new_query = new_token @ W_Q
    new_key = new_token @ W_K
    new_value = new_token @ W_V

    updated_keys = torch.cat([cached_keys, new_key], dim=0)
    updated_values = torch.cat([cached_values, new_value], dim=0)

    scores = new_query @ updated_keys.T / (d_k ** 0.5)
    weights = F.softmax(scores, dim=-1)
    output = weights @ updated_values

    return output, updated_keys, updated_values
