"""
Tokenizer with four atomic special tokens appended to the base 65-char vocab.

Specials are single IDs, not character runs:
  <|user|>       - starts a user turn
  <|assistant|>  - starts an assistant turn
  <|end|>        - ends a turn
  <|pad|>        - padding token

Base vocab = the 65 characters from the Module 5 checkpoint.
Total vocab = 69.
"""

from __future__ import annotations

SPECIAL_TOKENS = ["<|user|>", "<|assistant|>", "<|end|>", "<|pad|>"]


def build_vocab(base_stoi: dict[str, int]) -> tuple[dict[str, int], dict[int, str]]:
    """Return expanded stoi/itos with special tokens at the end."""
    stoi = dict(base_stoi)
    itos = {i: c for c, i in base_stoi.items()}
    for tok in SPECIAL_TOKENS:
        idx = len(stoi)
        stoi[tok] = idx
        itos[idx] = tok
    return stoi, itos


def encode(text: str, stoi: dict[str, int]) -> list[int]:
    """Encode a plain string into token IDs (character by character)."""
    return [stoi[c] for c in text]


def decode(ids: list[int] | torch.Tensor, itos: dict[int, str]) -> str:
    """Decode token IDs back to a string, preserving special tokens as literals."""
    import torch
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return "".join(itos.get(int(i), "?") for i in ids)
