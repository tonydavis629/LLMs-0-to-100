"""
Module 1 Exercise: N-gram Language Models

Build character-level and word-level n-gram models to generate text.
This recreates Shannon's 1948 experiment on statistical language modeling.

Fill in the lines marked with `raise NotImplementedError(...)`.
Each one needs only ONE line of code.
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path


def load_text(filepath: str) -> str:
    """Load a text file and strip the Project Gutenberg header/footer."""

    # Read the entire file into a string
    raw = Path(filepath).read_text(encoding="utf-8")

    # Markers that surround the actual book text in Gutenberg files
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"

    # Split the file into a list of lines
    lines = raw.split("\n")

    # Find the line numbers where the book starts and ends
    start_idx = 0
    end_idx = len(lines)
    for i, line in enumerate(lines):
        if start_marker in line.upper():
            start_idx = i + 1       # book starts on the NEXT line
        if end_marker in line.upper():
            end_idx = i             # book ends BEFORE this line
            break

    # TODO: Join lines[start_idx:end_idx] with newlines and return the result
    raise NotImplementedError("TODO: return the joined lines")


# ---------------------------------------------------------------------------
# Character-level models
# ---------------------------------------------------------------------------


def char_uniform(length: int = 500) -> str:
    """Generate random text by sampling uniformly from a-z and space.

    This is 0th order -- every character is equally likely.
    """
    # The 27 characters we can pick from (26 letters + space)
    alphabet = list("abcdefghijklmnopqrstuvwxyz ")

    # TODO: Sample `length` random characters from alphabet and join into a string
    # Use random.choices(population, k=...) to pick, then "".join() to combine
    raise NotImplementedError("TODO: return a random string of the given length")


def char_unigram(text: str, length: int = 500) -> str:
    """Generate text using character unigram frequencies (1st order).

    Count how often each character appears in `text`, then sample
    characters proportional to those frequencies.
    """
    # Lowercase the text so 'A' and 'a' are treated the same
    text = text.lower()

    # Counter counts how many times each character appears
    # e.g. Counter("aab") -> {"a": 2, "b": 1}
    counts = Counter(text)

    # Separate the characters and their counts into two parallel lists
    chars = list(counts.keys())       # e.g. ["a", "b", " ", "e", ...]
    weights = list(counts.values())   # e.g. [2, 1, 5, 3, ...]

    # TODO: Sample `length` characters using the weights, join into a string
    # Same as char_uniform but pass weights=weights to random.choices()
    raise NotImplementedError("TODO: return a frequency-weighted random string")


def build_char_ngram_model(text: str, n: int) -> dict[str, Counter]:
    """Build a character-level n-gram model.

    Slide over the text with a window of size n. The first n-1 characters
    are the "context" and the nth character is what we're predicting.

    Args:
        text: The source text (will be lowercased).
        n: The order of the model (2 = bigram, 3 = trigram, etc.)

    Returns:
        A dict mapping each (n-1)-character context to a Counter of
        next-character frequencies.
        Example for n=2: model["t"] -> Counter({"h": 450, "o": 230, ...})
    """
    text = text.lower()
    model: dict[str, Counter] = {}

    # Slide a window of size n across the text
    for i in range(len(text) - n + 1):
        # TODO: Extract the context (first n-1 chars) and next_char (the nth char)
        # context = text[i : i + n - 1], next_char = text[i + n - 1]
        raise NotImplementedError("TODO: set context and next_char from text[i:]")

        # Create a Counter for this context if we haven't seen it before
        if context not in model:
            model[context] = Counter()
        # Increment the count for this (context -> next_char) pair
        model[context][next_char] += 1

    return model


def generate_from_char_model(
    model: dict[str, Counter], length: int = 500, seed: str | None = None
) -> str:
    """Generate text from a character n-gram model.

    Start with a seed context, look up what characters follow it,
    sample one proportionally, append it, and repeat.
    """
    # If no seed given, pick a random context from the model
    if seed is None:
        seed = random.choice(list(model.keys()))

    # n is the full window size (context length + 1)
    n = len(seed) + 1
    # Start with the seed characters in a list
    result = list(seed)

    while len(result) < length:
        # TODO: Get the current context -- the last (n-1) characters joined together
        # e.g. if n=3 and result ends with ['a','t'], context = "at"
        raise NotImplementedError("TODO: set context from the last n-1 chars of result")

        if context in model:
            # Look up what characters follow this context
            counter = model[context]
            chars = list(counter.keys())
            weights = list(counter.values())
            # Pick one character, weighted by how often it appeared
            next_char = random.choices(chars, weights=weights, k=1)[0]
        else:
            # Context not in model -- fall back to a random context
            context = random.choice(list(model.keys()))
            counter = model[context]
            chars = list(counter.keys())
            weights = list(counter.values())
            next_char = random.choices(chars, weights=weights, k=1)[0]
        result.append(next_char)

    # Join all characters into a single string and return
    return "".join(result[:length])


# ---------------------------------------------------------------------------
# Word-level models
# ---------------------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer. Lowercase and split on whitespace."""
    return text.lower().split()


def word_unigram(text: str, length: int = 100) -> str:
    """Generate text using word unigram frequencies.

    Same idea as char_unigram, but with whole words instead of characters.
    """
    # Split text into a list of words
    words = tokenize(text)

    # Count how often each word appears
    counts = Counter(words)
    word_list = list(counts.keys())
    weights = list(counts.values())

    # TODO: Sample `length` words using the weights, join with spaces
    # Same pattern as char_unigram but use " ".join() instead of "".join()
    raise NotImplementedError("TODO: return frequency-weighted random words")


def build_word_ngram_model(text: str, n: int) -> dict[tuple[str, ...], Counter]:
    """Build a word-level n-gram model.

    Same structure as build_char_ngram_model, but contexts are tuples
    of words instead of strings of characters.
    """
    words = tokenize(text)
    model: dict[tuple[str, ...], Counter] = {}

    for i in range(len(words) - n + 1):
        # TODO: Extract context tuple and next_word
        # context = tuple(words[i : i + n - 1]), next_word = words[i + n - 1]
        raise NotImplementedError("TODO: set context and next_word from words[i:]")

        if context not in model:
            model[context] = Counter()
        model[context][next_word] += 1

    return model


def generate_from_word_model(
    model: dict[tuple[str, ...], Counter],
    length: int = 100,
    seed: tuple[str, ...] | None = None,
) -> str:
    """Generate text from a word n-gram model.

    Same as generate_from_char_model, but with word tuples for context.
    """
    if seed is None:
        seed = random.choice(list(model.keys()))

    n = len(seed) + 1
    result = list(seed)

    while len(result) < length:
        # TODO: Get the current context as a tuple of the last (n-1) words
        # Same as the char version but use tuple() instead of "".join()
        raise NotImplementedError("TODO: set context from the last n-1 words of result")

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
# Extra credit: Perplexity
# ---------------------------------------------------------------------------


def cross_entropy(text: str, model: dict[str, Counter]) -> float:
    """Compute the cross-entropy of text under a character n-gram model.

    Cross-entropy = -(1/N) * sum(log2(P(c_i | context_i)))

    For unseen contexts or characters, use a small smoothing probability
    of 1e-6 to avoid log(0).

    Returns:
        The cross-entropy in bits per character.
    """
    raise NotImplementedError("Extra credit: implement cross_entropy()")


def perplexity(text: str, model: dict[str, Counter]) -> float:
    """Compute perplexity: 2^(cross_entropy)."""
    return 2 ** cross_entropy(text, model)
