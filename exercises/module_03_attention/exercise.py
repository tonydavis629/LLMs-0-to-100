"""
Module 3 Exercise: Attention Mechanisms

Implement scaled dot-product attention step by step on a tiny token sequence.
You will compute Q, K, V projections, raw attention scores, scaled softmax
weights, the weighted output, and then add a causal mask and positional
embeddings to see how they change the attention pattern.

Fill in the lines marked with `raise NotImplementedError(...)`.
Each one needs only ONE line of code (or two at most).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Step 1: Create token vectors for a tiny sequence
# ---------------------------------------------------------------------------


def make_token_vectors(vocab_size: int = 10, d_model: int = 8, seq_len: int = 5) -> torch.Tensor:
    """Create a fixed random embedding matrix and look up token vectors.

    We use a small vocabulary of 10 tokens and an embedding dimension of 8.
    A fixed random seed makes the output reproducible.

    Args:
        vocab_size: Number of tokens in the vocabulary.
        d_model: Dimension of each embedding vector.
        seq_len: Number of tokens in the sequence.

    Returns:
        X: Tensor of shape (seq_len, d_model) with one row per token.
    """
    torch.manual_seed(42)
    # Embedding matrix: one row per vocabulary entry
    E = torch.randn(vocab_size, d_model) * 0.1
    # Pick seq_len token ids (just the first ones for reproducibility)
    token_ids = torch.tensor([2, 5, 1, 8, 3])
    # Look up each token's embedding vector
    X = E[token_ids]
    return X


# ---------------------------------------------------------------------------
# Step 2: Compute Q, K, and V projections
# ---------------------------------------------------------------------------


def compute_qkv(X: torch.Tensor, d_k: int = 4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project the token matrix X into query, key, and value matrices.

    Each projection uses its own learned weight matrix:
        Q = X @ W_Q,  K = X @ W_K,  V = X @ W_V

    Args:
        X: Token embeddings, shape (seq_len, d_model).
        d_k: Dimension of queries, keys, and values.

    Returns:
        (Q, K, V): Each of shape (seq_len, d_k).
    """
    torch.manual_seed(42)
    seq_len, d_model = X.shape
    # Random projection matrices (learned in practice)
    W_Q = torch.randn(d_model, d_k) * 0.1
    W_K = torch.randn(d_model, d_k) * 0.1
    W_V = torch.randn(d_model, d_k) * 0.1

    # TODO: Compute Q, K, V by projecting X through W_Q, W_K, W_V
    # HINT: use matrix multiplication X @ W_Q, X @ W_K, X @ W_V
    raise NotImplementedError("TODO: compute Q, K, V projections")


# ---------------------------------------------------------------------------
# Step 3: Compute raw attention scores with QK^T
# ---------------------------------------------------------------------------


def raw_attention_scores(Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Compute the pairwise compatibility scores: Q @ K^T.

    Each entry (i, j) measures how much query i aligns with key j.

    Args:
        Q: Query matrix, shape (seq_len, d_k).
        K: Key matrix, shape (seq_len, d_k).

    Returns:
        Scores matrix, shape (seq_len, seq_len).
    """
    # TODO: Compute the attention scores as the matrix product of Q and K^T
    # HINT: use Q @ K.T (the .T attribute transposes a 2D tensor)
    raise NotImplementedError("TODO: compute raw attention scores Q @ K^T")


# ---------------------------------------------------------------------------
# Step 4: Apply scaled softmax to produce attention weights
# ---------------------------------------------------------------------------


def scaled_softmax(scores: torch.Tensor, d_k: int) -> torch.Tensor:
    """Scale scores by 1/sqrt(d_k) and apply softmax along the key dimension.

    The scaling prevents dot products from growing large with dimension,
    which would push softmax into regions with tiny gradients.

    Args:
        scores: Raw attention scores, shape (seq_len, seq_len).
        d_k: Dimension of keys (used for scaling).

    Returns:
        Attention weights, shape (seq_len, seq_len). Each row sums to 1.
    """
    # TODO: Scale the scores by 1/sqrt(d_k), then apply softmax along dim=-1
    # HINT: divide scores by (d_k ** 0.5), then call F.softmax(..., dim=-1)
    raise NotImplementedError("TODO: apply scaled softmax to attention scores")


# ---------------------------------------------------------------------------
# Step 5: Compute the weighted sum of value vectors
# ---------------------------------------------------------------------------


def attention_output(weights: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Compute the attention output as a weighted sum of value vectors.

    Each output token is a weighted average of all value vectors,
    with weights given by the attention matrix.

    Args:
        weights: Attention weights, shape (seq_len, seq_len).
        V: Value matrix, shape (seq_len, d_k).

    Returns:
        Output tensor, shape (seq_len, d_k).
    """
    # TODO: Compute the weighted sum of values using the attention weights
    # HINT: use weights @ V (matrix multiply the weight matrix by the value matrix)
    raise NotImplementedError("TODO: compute attention output as weighted sum of values")


# ---------------------------------------------------------------------------
# Step 6: Add a causal mask
# ---------------------------------------------------------------------------


def causal_mask(seq_len: int) -> torch.Tensor:
    """Create a causal (lower-triangular) mask for autoregressive attention.

    A causal mask prevents each token from attending to future tokens.
    Entry (i, j) is 0 if j <= i (token i may attend to token j),
    and -inf if j > i (token i must NOT attend to token j).

    Args:
        seq_len: Length of the sequence.

    Returns:
        Mask tensor, shape (seq_len, seq_len), with 0s and -infs.
    """
    # TODO: Create a lower-triangular mask of 0s with -inf above the diagonal
    # HINT: use torch.tril(torch.ones(seq_len, seq_len)) for the lower triangle,
    #        then convert: 1 -> 0.0 and 0 -> float('-inf')
    raise NotImplementedError("TODO: create causal mask")


# ---------------------------------------------------------------------------
# Step 7: Compute masked attention weights
# ---------------------------------------------------------------------------


def masked_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, d_k: int) -> torch.Tensor:
    """Compute scaled dot-product attention with a causal mask.

    Steps: (1) raw scores QK^T, (2) scale by 1/sqrt(d_k),
    (3) add causal mask, (4) softmax, (5) weighted sum of V.

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
    # Add the mask: -inf entries become 0 after softmax
    scaled = scaled + mask
    weights = F.softmax(scaled, dim=-1)
    return attention_output(weights, V)


# ---------------------------------------------------------------------------
# Step 8: Add positional embeddings
# ---------------------------------------------------------------------------


def add_positional_embeddings(X: torch.Tensor) -> torch.Tensor:
    """Add learned positional embeddings to the token representations.

    Without position information, attention is permutation-equivariant:
    it treats "dog bites man" the same as "man bites dog".
    Positional embeddings give each position a unique vector that
    the model can use to distinguish token order.

    Args:
        X: Token embeddings, shape (seq_len, d_model).

    Returns:
        X_pos: Token embeddings plus positional embeddings, shape (seq_len, d_model).
    """
    torch.manual_seed(42)
    seq_len, d_model = X.shape
    # Learned positional embedding: one vector per position
    P = torch.randn(seq_len, d_model) * 0.1

    # TODO: Add the positional embeddings P to the token embeddings X
    # HINT: simply add X + P (elementwise addition of the two tensors)
    raise NotImplementedError("TODO: add positional embeddings to token embeddings")


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

    During autoregressive generation, we process one new token at a time.
    Instead of recomputing keys and values for all previous tokens, we
    cache them and only compute the new key and value for the latest token.

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
    # TODO: Compute new Q, K, V from the raw token using W_Q, W_K, W_V
    # HINT: project the new_token through all three weight matrices with @
    new_key = None
    new_value = None
    if new_key is None or new_value is None:
        raise NotImplementedError("Extra credit: project new_token through W_Q, W_K, W_V, then append K and V to cache")

    # Append new key and value to the cache
    updated_keys = torch.cat([cached_keys, new_key], dim=0)
    updated_values = torch.cat([cached_values, new_value], dim=0)

    # Compute the new query
    new_query = new_token @ W_Q
    # Compute attention: new query attends to ALL keys (no causal mask needed
    # because the new token is the last one, so it can see all previous tokens)
    scores = new_query @ updated_keys.T / (d_k ** 0.5)
    weights = F.softmax(scores, dim=-1)
    output = weights @ updated_values

    return output, updated_keys, updated_values
