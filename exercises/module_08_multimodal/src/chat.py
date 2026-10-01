"""Building the chat-format training examples, provided for you.

You do NOT need to edit this file. The language model was instruction-tuned
in Module 6 on conversations laid out as

    <|user|> prompt <|end|> <|assistant|> response <|end|>

These helpers turn a prompt (and, for training, its response) into that
token layout, line up the next-token targets behind the visual prefix, and
turn each dataset scene into its five training examples.
"""

from __future__ import annotations

import torch

from src.data import questions_for
from src.tokenizer import encode


def prompt_ids(prompt: str, special: dict, stoi: dict) -> list[int]:
    """The prompt as the model sees it: <|user|> prompt <|end|> <|assistant|>."""
    return [special["<|user|>"]] + encode(prompt, stoi) + [special["<|end|>"]] + [special["<|assistant|>"]]


def chat_ids(prompt: str, response: str, special: dict, stoi: dict):
    """Build the full prompt + response sequence, and where the response starts."""
    prefix = prompt_ids(prompt, special, stoi)
    seq = prefix + encode(response, stoi) + [special["<|end|>"]]
    return torch.tensor(seq, dtype=torch.long), len(prefix)


def targets_and_mask(seq: torch.Tensor, resp_start: int, k: int):
    """Align next-token targets to a sequence that has k visual prefix slots in front.

    Logits position t predicts input position t+1; input position t+1 is text token
    seq[t+1-k]. We train only where that predicted token is part of the response.
    """
    L = seq.shape[0]
    T = k + L
    pos = torch.arange(T)
    seq_idx = pos + 1 - k
    valid = (seq_idx >= 0) & (seq_idx <= L - 1)
    targets = torch.zeros(T, dtype=torch.long)
    targets[valid] = seq[seq_idx[valid].clamp(min=0)]
    mask = valid & (seq_idx >= resp_start)
    return targets, mask


def bridge_examples(split: dict, describe: str) -> list:
    """Flatten each scene into (image, prompt, response) tasks: describe + 4 VQAs."""
    out = []
    images = split["images"]
    for i in range(images.shape[0]):
        img = images[i]
        out.append((img, describe, split["captions"][i]))
        for q, a in questions_for(tuple(split["top"][i]), tuple(split["bottom"][i])):
            out.append((img, q, a))
    return out
