"""
Module 1 Solution: N-gram Language Models

Character-level and word-level n-gram models for text generation,
recreating Shannon's 1948 experiment on statistical language modeling.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from pathlib import Path


def load_text(filepath: str) -> str:
    """Load and return the contents of a text file.

    Strip the Project Gutenberg header/footer by finding the lines:
      '*** START OF THE PROJECT GUTENBERG EBOOK ...'
      '*** END OF THE PROJECT GUTENBERG EBOOK ...'
    and returning only the text between them.
    """
    raw = Path(filepath).read_text(encoding="utf-8")

    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"

    lines = raw.split("\n")
    start_idx = 0
    end_idx = len(lines)

    for i, line in enumerate(lines):
        if start_marker in line.upper():
            start_idx = i + 1
        if end_marker in line.upper():
            end_idx = i
            break

    return "\n".join(lines[start_idx:end_idx])


# ---------------------------------------------------------------------------
# Character-level models
# ---------------------------------------------------------------------------


def char_uniform(length: int = 500) -> str:
    """Generate random text by sampling uniformly from a-z and space (0th order)."""
    alphabet = list("abcdefghijklmnopqrstuvwxyz ")
    return "".join(random.choices(alphabet, k=length))


def char_unigram(text: str, length: int = 500) -> str:
    """Generate text using character unigram frequencies (1st order)."""
    text = text.lower()
    counts = Counter(text)
    chars = list(counts.keys())
    weights = list(counts.values())
    return "".join(random.choices(chars, weights=weights, k=length))


def build_char_ngram_model(text: str, n: int) -> dict[str, Counter]:
    """Build a character-level n-gram model.

    Args:
        text: The source text (will be lowercased).
        n: The order of the model (2 = bigram, 3 = trigram, etc.)

    Returns:
        A dict mapping each (n-1)-character context to a Counter of
        next-character frequencies.
    """
    text = text.lower()
    model: dict[str, Counter] = {}

    for i in range(len(text) - n + 1):
        context = text[i : i + n - 1]
        next_char = text[i + n - 1]
        if context not in model:
            model[context] = Counter()
        model[context][next_char] += 1

    return model


def generate_from_char_model(
    model: dict[str, Counter], length: int = 500, seed: str | None = None
) -> str:
    """Generate text from a character n-gram model."""
    if seed is None:
        seed = random.choice(list(model.keys()))

    n = len(seed) + 1
    result = list(seed)

    while len(result) < length:
        context = "".join(result[-(n - 1) :])
        if context in model:
            counter = model[context]
            chars = list(counter.keys())
            weights = list(counter.values())
            next_char = random.choices(chars, weights=weights, k=1)[0]
        else:
            context = random.choice(list(model.keys()))
            counter = model[context]
            chars = list(counter.keys())
            weights = list(counter.values())
            next_char = random.choices(chars, weights=weights, k=1)[0]
        result.append(next_char)

    return "".join(result[:length])


# ---------------------------------------------------------------------------
# Word-level models
# ---------------------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer. Lowercase and split on whitespace."""
    return text.lower().split()


def word_unigram(text: str, length: int = 100) -> str:
    """Generate text using word unigram frequencies."""
    words = tokenize(text)
    counts = Counter(words)
    word_list = list(counts.keys())
    weights = list(counts.values())
    return " ".join(random.choices(word_list, weights=weights, k=length))


def build_word_ngram_model(
    text: str, n: int
) -> dict[tuple[str, ...], Counter]:
    """Build a word-level n-gram model."""
    words = tokenize(text)
    model: dict[tuple[str, ...], Counter] = {}

    for i in range(len(words) - n + 1):
        context = tuple(words[i : i + n - 1])
        next_word = words[i + n - 1]
        if context not in model:
            model[context] = Counter()
        model[context][next_word] += 1

    return model


def generate_from_word_model(
    model: dict[tuple[str, ...], Counter],
    length: int = 100,
    seed: tuple[str, ...] | None = None,
) -> str:
    """Generate text from a word n-gram model."""
    if seed is None:
        seed = random.choice(list(model.keys()))

    n = len(seed) + 1
    result = list(seed)

    while len(result) < length:
        context = tuple(result[-(n - 1) :])
        if context in model:
            counter = model[context]
            words = list(counter.keys())
            weights = list(counter.values())
            next_word = random.choices(words, weights=weights, k=1)[0]
        else:
            context = random.choice(list(model.keys()))
            counter = model[context]
            words = list(counter.keys())
            weights = list(counter.values())
            next_word = random.choices(words, weights=weights, k=1)[0]
        result.append(next_word)

    return " ".join(result[:length])


# ---------------------------------------------------------------------------
# Stretch goal: Perplexity
# ---------------------------------------------------------------------------


def cross_entropy(text: str, model: dict[str, Counter]) -> float:
    """Compute the cross-entropy of text under a character n-gram model.

    Cross-entropy = -(1/N) * sum(log2(P(c_i | context_i)))

    For unseen contexts or characters, use a small smoothing probability
    of 1e-6 to avoid log(0).
    """
    text = text.lower()
    # Determine context length from the model
    sample_key = next(iter(model))
    context_len = len(sample_key)
    n = context_len + 1

    total_log_prob = 0.0
    count = 0

    for i in range(len(text) - n + 1):
        context = text[i : i + context_len]
        next_char = text[i + context_len]

        if context in model:
            counter = model[context]
            total = sum(counter.values())
            char_count = counter.get(next_char, 0)
            if char_count > 0:
                prob = char_count / total
            else:
                prob = 1e-6
        else:
            prob = 1e-6

        total_log_prob += math.log2(prob)
        count += 1

    if count == 0:
        return 0.0

    return -total_log_prob / count


def perplexity(text: str, model: dict[str, Counter]) -> float:
    """Compute perplexity: 2^(cross_entropy)."""
    return 2 ** cross_entropy(text, model)
