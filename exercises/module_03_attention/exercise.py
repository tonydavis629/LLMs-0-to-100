"""
Module 3 Exercise: Attention Mechanisms

Implement a tiny attention layer step by step on a small token sequence.
You will create Q, K, and V from a layer's learned weight matrices, compute
attention scores, turn those scores into weights, produce the weighted sum,
add a causal mask, and add sinusoidal positional encodings.

Fill in the lines marked with `raise NotImplementedError(...)`.
Each blank needs only one expression or one short assignment.
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
    # Create the embedding table: one vector for each vocabulary item.
    E = torch.randn(vocab_size, d_model) * 0.1
    # Choose token ids for the toy sentence.
    token_ids = torch.tensor([2, 5, 1, 8, 3])
    # Slice the embedding table to get one vector per token in the sequence.
    X = E[token_ids[:seq_len]]
    return X


# ---------------------------------------------------------------------------
# Steps 2-7: A tiny single-head attention layer
# ---------------------------------------------------------------------------


class TinyAttentionLayer:
    """A minimal attention layer with randomly initialized projection weights."""

    def __init__(self, d_model: int = 8, d_k: int = 4, seed: int = 42) -> None:
        """Initialize the projection matrices used by one attention head.

        Args:
            d_model: Dimension of each token embedding.
            d_k: Dimension of the query, key, and value vectors.
            seed: Random seed so the exercise output is reproducible.
        """
        # Store dimensions so other methods can use them.
        self.d_model = d_model
        self.d_k = d_k
        # Fix the random seed so every run creates the same weights.
        torch.manual_seed(seed)
        # Random weights stand in for learned weights in a trained model.
        self.W_Q = torch.randn(d_model, d_k) * 0.1
        self.W_K = torch.randn(d_model, d_k) * 0.1
        self.W_V = torch.randn(d_model, d_k) * 0.1

    def compute_qkv(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project the token matrix X into query, key, and value matrices.

        Args:
            X: Token embeddings, shape (seq_len, d_model).

        Returns:
            (Q, K, V): Each tensor has shape (seq_len, d_k).
        """
        # TODO: Compute and return Q, K, V by multiplying X by this layer's W_Q, W_K, and W_V.
        # HINT: use X @ self.W_Q, X @ self.W_K, and X @ self.W_V.
        raise NotImplementedError("TODO: compute Q, K, V projections")

    def raw_attention_scores(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """Compute the pairwise compatibility scores: Q @ K^T.

        Args:
            Q: Query matrix, shape (seq_len, d_k).
            K: Key matrix, shape (seq_len, d_k).

        Returns:
            Scores matrix, shape (seq_len, seq_len).
        """
        # TODO: Compute the attention scores as the matrix product of Q and K^T.
        # HINT: use Q @ K.T (the .T attribute transposes a 2D tensor).
        raise NotImplementedError("TODO: compute raw attention scores Q @ K^T")

    def scaled_softmax(self, scores: torch.Tensor) -> torch.Tensor:
        """Scale scores by 1/sqrt(d_k) and apply softmax along the key dimension.

        Args:
            scores: Raw attention scores, shape (seq_len, seq_len).

        Returns:
            Attention weights, shape (seq_len, seq_len). Each row sums to 1.
        """
        # TODO: Scale the scores by 1/sqrt(d_k), then apply softmax along dim=-1.
        # HINT: divide scores by (self.d_k ** 0.5), then call F.softmax(..., dim=-1).
        raise NotImplementedError("TODO: apply scaled softmax to attention scores")

    def attention_output(self, weights: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Compute the attention output as a weighted sum of value vectors.

        Args:
            weights: Attention weights, shape (seq_len, seq_len).
            V: Value matrix, shape (seq_len, d_k).

        Returns:
            Output tensor, shape (seq_len, d_k).
        """
        # TODO: Compute the weighted sum of values using the attention weights.
        # HINT: use weights @ V (matrix multiply the weight matrix by the value matrix).
        raise NotImplementedError("TODO: compute attention output as weighted sum of values")

    def causal_mask(self, seq_len: int) -> torch.Tensor:
        """Create a causal mask for autoregressive attention.

        Entry (i, j) is 0 if j <= i, which means token i may attend to token j.
        Entry (i, j) is -inf if j > i, which blocks future tokens before softmax.

        Args:
            seq_len: Length of the sequence.

        Returns:
            Mask tensor, shape (seq_len, seq_len), with 0s and -infs.
        """
        # Create a lower-triangular matrix: 1 means allowed, 0 means blocked.
        allowed = torch.tril(torch.ones(seq_len, seq_len))
        # TODO: Convert allowed positions to 0.0 and blocked positions to -inf.
        # HINT: use masked_fill twice: first replace 0s with -inf, then replace 1s with 0.0.
        raise NotImplementedError("TODO: create causal mask")

    def masked_attention(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled dot-product attention with a causal mask.

        Args:
            X: Token embeddings, shape (seq_len, d_model).

        Returns:
            (output, weights): masked attention output and masked weights.
        """
        # Project the input tokens into Q, K, and V.
        Q, K, V = self.compute_qkv(X)
        # Compare every query to every key.
        scores = self.raw_attention_scores(Q, K)
        # Scale scores before masking and softmax.
        scaled = scores / (self.d_k ** 0.5)
        # Add a mask that blocks future positions.
        masked_scores = scaled + self.causal_mask(X.shape[0])
        # Normalize each row into attention weights.
        weights = F.softmax(masked_scores, dim=-1)
        # Retrieve values using the masked weights.
        output = self.attention_output(weights, V)
        return output, weights


# ---------------------------------------------------------------------------
# Step 8: Add sinusoidal positional encodings
# ---------------------------------------------------------------------------


def add_positional_embeddings(X: torch.Tensor) -> torch.Tensor:
    """Add sinusoidal positional encodings to the token representations.

    Args:
        X: Token embeddings, shape (seq_len, d_model).

    Returns:
        X_pos: Token embeddings plus positional encodings, shape (seq_len, d_model).
    """
    # Read the sequence length and embedding width from the input tensor.
    seq_len, d_model = X.shape
    # Create a column vector [0, 1, 2, ...] for token positions.
    position = torch.arange(seq_len, dtype=X.dtype, device=X.device).unsqueeze(1)
    # Create the even dimension indices: 0, 2, 4, ...
    dim_pair = torch.arange(0, d_model, 2, dtype=X.dtype, device=X.device)
    # Compute the denominator term from the sinusoidal encoding equation.
    angle_rates = 1 / (10000 ** (dim_pair / d_model))
    # Broadcast positions against dimension frequencies to get all angles.
    angles = position * angle_rates
    # Start with zeros, then fill even dimensions with sine values.
    P = torch.zeros_like(X)
    P[:, 0::2] = torch.sin(angles)

    # TODO: Fill the odd dimensions of P with cosine values from the sinusoidal equation.
    # HINT: use torch.cos(angles) and assign the result to P[:, 1::2].
    raise NotImplementedError("TODO: fill cosine positional dimensions")

    # Add position information to each token embedding.
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
    # TODO: Project new_token through W_Q, W_K, and W_V.
    # HINT: use new_token @ W_Q, new_token @ W_K, and new_token @ W_V.
    new_query = None
    new_key = None
    new_value = None
    if new_query is None or new_key is None or new_value is None:
        raise NotImplementedError("Extra credit: project new_token through W_Q, W_K, and W_V")

    # Append the new key and value to the existing cache.
    updated_keys = torch.cat([cached_keys, new_key], dim=0)
    updated_values = torch.cat([cached_values, new_value], dim=0)

    # Compare the new query with all cached keys.
    scores = new_query @ updated_keys.T / (d_k ** 0.5)
    # Convert the scores into weights over all cached values.
    weights = F.softmax(scores, dim=-1)
    # Retrieve the weighted sum of cached values.
    output = weights @ updated_values

    return output, updated_keys, updated_values
