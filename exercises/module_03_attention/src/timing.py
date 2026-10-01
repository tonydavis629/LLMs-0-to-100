"""Timing attention with and without a KV cache, provided for you.

You do NOT need to edit this file. The extra credit plots these timings to
show why caching keys and values makes generating each new token cheaper.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from src.prerequisites import earlier_qkv


def time_attention(layer, X: torch.Tensor) -> tuple[list[int], list[float], list[float]]:
    """Time one new token's attention with and without a KV cache.

    Args:
        layer: The attention layer whose weights project the tokens.
        X: Token embeddings, shape (seq_len, d_model).

    Returns:
        (lengths, ms_without_cache, ms_with_cache): average milliseconds per
        token for each sequence length 1..seq_len.
    """
    d_k = layer.d_k
    lengths = list(range(1, X.shape[0] + 1))
    no_cache, with_cache = [], []
    for n in lengths:
        Q, K, V = earlier_qkv(layer, X[:n], "time attention with and without the cache")

        # Without a cache: recompute attention for every token from scratch
        start = time.perf_counter()
        for _ in range(100):
            weights = F.softmax(Q @ K.T / (d_k ** 0.5), dim=-1)
            _ = weights @ V
        no_cache.append((time.perf_counter() - start) / 100 * 1000)

        # With a cache: only the newest token's query attends over the stored keys
        start = time.perf_counter()
        for _ in range(100):
            weights = F.softmax(Q[-1:] @ K.T / (d_k ** 0.5), dim=-1)
            _ = weights @ V
        with_cache.append((time.perf_counter() - start) / 100 * 1000)
    return lengths, no_cache, with_cache
