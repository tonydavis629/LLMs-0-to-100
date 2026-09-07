"""Token sampling helpers, provided for you.

You do NOT need to edit this file. `_sample_topk_token()` restricts the
next-token distribution to its k most likely entries and draws one token
from what remains; you call it from `sample_with_temperature_topk()`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _sample_topk_token(next_logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Sample one token from the top-k logits."""
    # Find the k largest logits for each batch item.
    topk_vals, _ = torch.topk(next_logits, top_k, dim=-1)
    # The last value in topk_vals is the cutoff for staying in the top k.
    threshold = topk_vals[:, -1].unsqueeze(-1)
    # Set all logits below the cutoff to -infinity, so softmax gives them 0 probability.
    next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))
    # Convert the filtered logits to probabilities.
    probs = F.softmax(next_logits, dim=-1)
    # Draw one token ID from that probability distribution.
    return torch.multinomial(probs, num_samples=1)
