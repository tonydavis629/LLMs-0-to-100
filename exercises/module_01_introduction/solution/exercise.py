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

# Provided helpers live in src/ - you do not need to edit them.
from src.text import tokenize


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


# ---------------------------------------------------------------------------
# Word-level models
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Extra credit: Perplexity
# ---------------------------------------------------------------------------


def cross_entropy(text: str, model: dict[str, Counter]) -> float:
    """Compute the cross-entropy of text under a character n-gram model.

    Cross-entropy = -(1/N) * sum(log2(P(c_i | context_i)))

    For unseen contexts or characters, use a small smoothing probability
    of 1e-6 to avoid log(0).
    """
    text = text.lower()
    # Every key in the model has the same length: the context size (n-1)
    context_len = len(next(iter(model)))

    total_bits = 0.0
    count = 0
    for i in range(len(text) - context_len):
        context = text[i : i + context_len]
        next_char = text[i + context_len]
        counter = model.get(context)
        if counter and counter[next_char] > 0:
            prob = counter[next_char] / sum(counter.values())
        else:
            prob = 1e-6  # never seen: tiny probability instead of log(0)
        total_bits += -math.log2(prob)
        count += 1

    return total_bits / count if count else 0.0
