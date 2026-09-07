"""The sampling loops that turn an n-gram count table into text.

You do NOT need to edit this file. The loop is always the same: look at the
current context, look up which tokens followed that context in the training
data, and pick one at random weighted by how often it followed. Modern
language models run this exact loop, with a learned probability distribution
in place of the counts.
"""

from __future__ import annotations

import random
from collections import Counter


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
        # The current context is the last n-1 characters we have generated
        context = "".join(result[-(n - 1):])

        if context in model:
            # Look up what characters follow this context
            counter = model[context]
            chars = list(counter.keys())
            weights = list(counter.values())
            # Pick one character, weighted by how often it appeared
            next_char = random.choices(chars, weights=weights, k=1)[0]
        else:
            # Context not in model: fall back to a random context
            context = random.choice(list(model.keys()))
            counter = model[context]
            chars = list(counter.keys())
            weights = list(counter.values())
            next_char = random.choices(chars, weights=weights, k=1)[0]
        result.append(next_char)

    # Join all characters into a single string and return
    return "".join(result[:length])


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
        # The current context is the last n-1 words we have generated
        context = tuple(result[-(n - 1):])

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
