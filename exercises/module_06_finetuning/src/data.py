"""
Toy instruction dataset: deterministically generated, highly learnable.

All characters used here must exist in the base Module-5 vocabulary (65 chars).
Tasks weighted toward simple memorization and transformation so a tiny model
shows a crisp flip in a few hundred CPU steps.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


def _make_uppercase(n: int, rng: random.Random) -> list[dict[str, str]]:
    words = ["hello", "world", "python", "machine", "learning", "model",
             "token", "embed", "attention", "neuron", "layer", "batch",
             "train", "eval", "loss", "gradient", "optimizer"]
    out = []
    for _ in range(n):
        w = rng.choice(words)
        out.append({"prompt": f"uppercase: {w}", "response": w.upper()})
    return out


def _make_qa(n: int, rng: random.Random) -> list[dict[str, str]]:
    pairs = [
        ("capital of France?", "Paris."),
        ("capital of Japan?", "Tokyo."),
        ("capital of Germany?", "Berlin."),
        ("color of sky?", "blue."),
        ("color of grass?", "green."),
        ("opposite of hot?", "cold."),
        ("opposite of up?", "down."),
    ]
    out = []
    for _ in range(n):
        q, a = rng.choice(pairs)
        out.append({"prompt": q, "response": a})
    return out


def _make_repeat(n: int, rng: random.Random) -> list[dict[str, str]]:
    words = ["cat", "dog", "sun", "moon", "star", "tree", "code", "data",
             "text", "word", "chat", "bot", "run", "fun"]
    out = []
    for _ in range(n):
        w = rng.choice(words)
        out.append({"prompt": f"repeat: {w}", "response": w})
    return out


def _make_reverse(n: int, rng: random.Random) -> list[dict[str, str]]:
    words = ["cat", "dog", "red", "top", "pot", "live", "draw", "part", "step"]
    out = []
    for _ in range(n):
        w = rng.choice(words)
        out.append({"prompt": f"reverse: {w}", "response": w[::-1]})
    return out


def generate_dataset(seed: int = 42) -> list[dict[str, str]]:
    """Generate prompt-response pairs for toy SFT."""
    rng = random.Random(seed)
    pairs: list[dict[str, str]] = []
    pairs += _make_uppercase(120, rng)
    pairs += _make_qa(100, rng)
    pairs += _make_repeat(80, rng)
    pairs += _make_reverse(50, rng)
    rng.shuffle(pairs)
    return pairs


def save_dataset(pairs: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in pairs:
            f.write(json.dumps(item) + "\n")


def load_dataset(path: Path) -> list[dict[str, str]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
