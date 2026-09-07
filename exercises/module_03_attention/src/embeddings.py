"""Token embedding setup provided for you.

You do NOT need to edit this file. It builds the fixed random embedding
table and looks up the vectors for the toy sequence the exercise attends over.
"""

from __future__ import annotations

import torch


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
