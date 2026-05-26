"""
Module 3 Solution: Attention Mechanisms

Complete reference implementation of scaled dot-product attention,
causal masking, positional embeddings, and a KV cache for
autoregressive generation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Step 1: Create token vectors for a tiny sequence
# ---------------------------------------------------------------------------


def make_token_vectors(vocab_size: int = 10, d_model: int = 8, seq_len: int = 5) -> torch.Tensor:
    """Create a fixed random embedding matrix and look up token vectors.

    Args:
        vocab_size: Number of tokens in the vocabulary.
        d_model: Dimension of each embedding vector.
        seq_len: Number of tokens in the sequence.

    Returns:
        X: Tensor of shape (seq_len, d_model) with one row per token.
    """
    torch.manual_seed(42)
    E = torch.randn(vocab_size, d_model) * 0.1
    token_ids = torch.tensor([2, 5, 1, 8, 3])
    X = E[token_ids]
    return X


# ---------------------------------------------------------------------------
# Step 2: Compute Q, K, and V projections
# ---------------------------------------------------------------------------


def compute_qkv(X: torch.Tensor, d_k: int = 4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project the token matrix X into query, key, and value matrices.

    Args:
        X: Token embeddings, shape (seq_len, d_model).
        d_k: Dimension of queries, keys, and values.

    Returns:
        (Q, K, V): Each of shape (seq_len, d_k).
    """
    torch.manual_seed(42)
    seq_len, d_model = X.shape
    W_Q = torch.randn(d_model, d_k) * 0.1
    W_K = torch.randn(d_model, d_k) * 0.1
    W_V = torch.randn(d_model, d_k) * 0.1

    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    return Q, K, V


# ---------------------------------------------------------------------------
# Step 3: Compute raw attention scores with QK^T
# ---------------------------------------------------------------------------


def raw_attention_scores(Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Compute the pairwise compatibility scores: Q @ K^T.

    Args:
        Q: Query matrix, shape (seq_len, d_k).
        K: Key matrix, shape (seq_len, d_k).

    Returns:
        Scores matrix, shape (seq_len, seq_len).
    """
    return Q @ K.T


# ---------------------------------------------------------------------------
# Step 4: Apply scaled softmax to produce attention weights
# ---------------------------------------------------------------------------


def scaled_softmax(scores: torch.Tensor, d_k: int) -> torch.Tensor:
    """Scale scores by 1/sqrt(d_k) and apply softmax along the key dimension.

    Args:
        scores: Raw attention scores, shape (seq_len, seq_len).
        d_k: Dimension of keys (used for scaling).

    Returns:
        Attention weights, shape (seq_len, seq_len). Each row sums to 1.
    """
    return F.softmax(scores / (d_k ** 0.5), dim=-1)


# ---------------------------------------------------------------------------
# Step 5: Compute the weighted sum of value vectors
# ---------------------------------------------------------------------------


def attention_output(weights: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Compute the attention output as a weighted sum of value vectors.

    Args:
        weights: Attention weights, shape (seq_len, seq_len).
        V: Value matrix, shape (seq_len, d_k).

    Returns:
        Output tensor, shape (seq_len, d_k).
    """
    return weights @ V


# ---------------------------------------------------------------------------
# Step 6: Add a causal mask
# ---------------------------------------------------------------------------


def causal_mask(seq_len: int) -> torch.Tensor:
    """Create a causal (lower-triangular) mask for autoregressive attention.

    Args:
        seq_len: Length of the sequence.

    Returns:
        Mask tensor, shape (seq_len, seq_len), with 0s and -infs.
    """
    mask = torch.tril(torch.ones(seq_len, seq_len))
    mask = mask.masked_fill(mask == 0, float('-inf'))
    return mask


# ---------------------------------------------------------------------------
# Step 7: Compute masked attention weights
# ---------------------------------------------------------------------------


def masked_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, d_k: int) -> torch.Tensor:
    """Compute scaled dot-product attention with a causal mask.

    Args:
        Q: Query matrix, shape (seq_len, d_k).
        K: Key matrix, shape (seq_len, d_k).
        V: Value matrix, shape (seq_len, d_k).
        d_k: Key dimension for scaling.

    Returns:
        Output tensor, shape (seq_len, d_k).
    """
    seq_len = Q.shape[0]
    scores = raw_attention_scores(Q, K)
    scaled = scores / (d_k ** 0.5)
    mask = causal_mask(seq_len)
    scaled = scaled + mask
    weights = F.softmax(scaled, dim=-1)
    return attention_output(weights, V)


# ---------------------------------------------------------------------------
# Step 8: Add positional embeddings
# ---------------------------------------------------------------------------


def add_positional_embeddings(X: torch.Tensor) -> torch.Tensor:
    """Add learned positional embeddings to the token representations.

    Args:
        X: Token embeddings, shape (seq_len, d_model).

    Returns:
        X_pos: Token embeddings plus positional embeddings, shape (seq_len, d_model).
    """
    torch.manual_seed(42)
    seq_len, d_model = X.shape
    P = torch.randn(seq_len, d_model) * 0.1

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
    """Compute attention for one new token using a KV cache.

    Args:
        new_token: The raw new token embedding, shape (1, d_model).
        cached_keys: All previously computed keys, shape (n_cached, d_k).
        cached_values: All previously computed values, shape (n_cached, d_k).
        W_Q: Query projection matrix, shape (d_model, d_k).
        W_K: Key projection matrix, shape (d_model, d_k).
        W_V: Value projection matrix, shape (d_model, d_k).
        d_k: Key dimension for scaling.

    Returns:
        (output, updated_keys, updated_values):
            output: Attention output for the new token, shape (1, d_k).
            updated_keys: Keys with the new key appended, shape (n_cached+1, d_k).
            updated_values: Values with the new value appended, shape (n_cached+1, d_k).
    """
    new_key = new_token @ W_K
    new_value = new_token @ W_V

    updated_keys = torch.cat([cached_keys, new_key], dim=0)
    updated_values = torch.cat([cached_values, new_value], dim=0)

    new_query = new_token @ W_Q
    scores = new_query @ updated_keys.T / (d_k ** 0.5)
    weights = F.softmax(scores, dim=-1)
    output = weights @ updated_values

    return output, updated_keys, updated_values
