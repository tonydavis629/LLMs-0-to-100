"""
Module 1 Exercise: N-gram Language Models

Build character-level and word-level n-gram models to generate text.
This recreates Shannon's 1948 experiment on statistical language modeling.

Complete the TODO sections below. Each TODO asks you to fill in ONE line
(replace ___ with your answer).
"""

from __future__ import annotations

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

    # TODO: Return the text between start_idx and end_idx, joined by newlines
    return ___


# ---------------------------------------------------------------------------
# Character-level models
# ---------------------------------------------------------------------------


def char_uniform(length: int = 500) -> str:
    """Generate random text by sampling uniformly from a-z and space (0th order).

    Returns a string of the given length.
    """
    alphabet = list("abcdefghijklmnopqrstuvwxyz ")

    # TODO: Return a string of `length` characters sampled uniformly at random
    # Hint: use random.choices() and "".join()
    return ___


def char_unigram(text: str, length: int = 500) -> str:
    """Generate text using character unigram frequencies (1st order).

    Count how often each character appears in `text`, then sample characters
    proportional to those frequencies.
    """
    text = text.lower()
    counts = Counter(text)
    chars = list(counts.keys())
    weights = list(counts.values())

    # TODO: Return a string of `length` characters sampled using the weights
    # Hint: use random.choices() with the weights parameter, then "".join()
    return ___


def build_char_ngram_model(text: str, n: int) -> dict[str, Counter]:
    """Build a character-level n-gram model.

    Args:
        text: The source text (will be lowercased).
        n: The order of the model (2 = bigram, 3 = trigram, etc.)

    Returns:
        A dict mapping each (n-1)-character context to a Counter of
        next-character frequencies.

    Example for n=2 (bigram):
        model["th"] -> Counter({"e": 450, "a": 230, "i": 180, ...})
    """
    text = text.lower()
    model: dict[str, Counter] = {}

    for i in range(len(text) - n + 1):
        # TODO: Extract the (n-1)-char context and the next character from text[i:]
        # Hint: context = text[i : i + n - 1], next_char = text[i + n - 1]
        context, next_char = ___
        if context not in model:
            model[context] = Counter()
        model[context][next_char] += 1

    return model


def generate_from_char_model(model: dict[str, Counter], length: int = 500, seed: str | None = None) -> str:
    """Generate text from a character n-gram model.

    Args:
        model: A dict from build_char_ngram_model().
        length: Number of characters to generate.
        seed: Starting context. If None, pick a random context from the model.

    Returns:
        A generated string of the given length.
    """
    if seed is None:
        seed = random.choice(list(model.keys()))

    n = len(seed) + 1  # context length + 1
    result = list(seed)

    while len(result) < length:
        # TODO: Get the current context (the last n-1 characters of result)
        # Hint: join the last (n-1) elements of result into a string
        context = ___
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
    """Generate text using word unigram frequencies.

    Args:
        text: Source text.
        length: Number of words to generate.

    Returns:
        A string of space-separated generated words.
    """
    words = tokenize(text)
    counts = Counter(words)
    word_list = list(counts.keys())
    weights = list(counts.values())

    # TODO: Return `length` words sampled by frequency, joined by spaces
    # Hint: same pattern as char_unigram but with " ".join() and word_list
    return ___


def build_word_ngram_model(text: str, n: int) -> dict[tuple[str, ...], Counter]:
    """Build a word-level n-gram model.

    Args:
        text: Source text.
        n: Order of the model (2 = bigram, 3 = trigram).

    Returns:
        A dict mapping (n-1)-word tuple contexts to Counter of next words.
    """
    words = tokenize(text)
    model: dict[tuple[str, ...], Counter] = {}

    for i in range(len(words) - n + 1):
        # TODO: Extract the (n-1)-word context tuple and the next word
        # Hint: context = tuple(words[i : i + n - 1]), next_word = words[i + n - 1]
        context, next_word = ___
        if context not in model:
            model[context] = Counter()
        model[context][next_word] += 1

    return model


def generate_from_word_model(model: dict[tuple[str, ...], Counter], length: int = 100, seed: tuple[str, ...] | None = None) -> str:
    """Generate text from a word n-gram model.

    Args:
        model: A dict from build_word_ngram_model().
        length: Number of words to generate.
        seed: Starting context tuple. If None, pick random.

    Returns:
        A string of space-separated generated words.
    """
    if seed is None:
        seed = random.choice(list(model.keys()))

    n = len(seed) + 1
    result = list(seed)

    while len(result) < length:
        # TODO: Get the current context as a tuple of the last (n-1) words
        # Hint: similar to the char model, but use tuple() instead of "".join()
        context = ___
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

    Where N is the number of characters evaluated and P is the probability
    from the model.

    For unseen contexts or characters, use a small smoothing probability
    of 1e-6 to avoid log(0).

    Args:
        text: The text to evaluate.
        model: A character n-gram model from build_char_ngram_model().

    Returns:
        The cross-entropy in bits per character.
    """
    # TODO: This is the stretch goal! Try it on your own.
    # Hint: perplexity = 2^(cross_entropy)
    raise NotImplementedError("Complete cross_entropy() -- stretch goal!")


def perplexity(text: str, model: dict[str, Counter]) -> float:
    """Compute perplexity: 2^(cross_entropy)."""
    return 2 ** cross_entropy(text, model)
