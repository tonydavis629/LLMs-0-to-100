"""The chat format: building prompts and reading answers, provided for you.

You do NOT need to edit this file. The instruct model from Module 6 expects
every prompt wrapped in special tokens:

    <|user|> reverse: abcde <|end|> <|assistant|>

and it signals the end of its answer with another <|end|>. These helpers
build that prefix and cut the model's output back down to the answer text.
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.tokenizer import decode, encode

END_TOKEN = "<|end|>"

_MODULE_DIR = Path(__file__).resolve().parent.parent


def data_file(name: str) -> Path:
    """Return the path to a bundled file in this module's data/ folder."""
    return _MODULE_DIR / "data" / name


def prompt_ids(prompt: str, stoi: dict[str, int]) -> torch.Tensor:
    """Build the generation prefix [user] prompt [end] [assistant] as a (1, P) tensor."""
    ids = [stoi["<|user|>"]] + encode(prompt, stoi) + [stoi[END_TOKEN]] + [stoi["<|assistant|>"]]
    return torch.tensor([ids], dtype=torch.long)


def response_text(seq: torch.Tensor, prompt_len: int, itos: dict[int, str]) -> str:
    """Decode the generated portion of a sequence, cut at the first <|end|>."""
    text = decode(seq[prompt_len:], itos)
    return text.split(END_TOKEN, 1)[0]


def truncate_at_end(seq: torch.Tensor, prompt_len: int, end_id: int) -> torch.Tensor:
    """Keep prompt + generated tokens up to and including the first <|end|>."""
    gen_ids = seq[prompt_len:].tolist()
    if end_id in gen_ids:
        cut = gen_ids.index(end_id) + 1
        return seq[: prompt_len + cut]
    return seq
