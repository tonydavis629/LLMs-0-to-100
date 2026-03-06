"""
Module 1 Exercise: N-gram Text Generator

Run with:
    uv run python -m exercises.module_01_introduction.src.main \
        exercises/module_01_introduction/data/alice.txt
"""

from __future__ import annotations

import argparse
import sys

from .ngrams import (
    build_char_ngram_model,
    build_word_ngram_model,
    char_uniform,
    char_unigram,
    generate_from_char_model,
    generate_from_word_model,
    load_text,
    word_unigram,
)


def _try_run(label: str, fn, *args) -> None:
    """Run fn(*args) and print the result, or skip if not yet implemented."""
    print(f"=== {label} ===")
    try:
        result = fn(*args)
        print(result)
    except NotImplementedError as e:
        print(f"  [skipped: {e}]")
    print()


def main():
    parser = argparse.ArgumentParser(description="N-gram text generator")
    parser.add_argument("corpus", help="Path to text file (e.g., data/alice.txt)")
    parser.add_argument(
        "--model",
        choices=["uniform", "char1", "char2", "char3", "word1", "word2", "word3", "all"],
        default="all",
        help="Which model to run (default: all)",
    )
    parser.add_argument("--length", type=int, default=500, help="Length of generated text")
    args = parser.parse_args()

    # Step 1: Load the corpus
    try:
        text = load_text(args.corpus)
        print(f"Loaded {len(text)} characters from {args.corpus}\n")
    except NotImplementedError as e:
        print(f"Cannot load text: {e}")
        print("Implement load_text() first, then re-run.")
        sys.exit(1)

    models = (
        ["uniform", "char1", "char2", "char3", "word1", "word2", "word3"]
        if args.model == "all"
        else [args.model]
    )

    for name in models:
        match name:
            case "uniform":
                _try_run("0th Order: Uniform Random Characters", char_uniform, args.length)
            case "char1":
                _try_run("1st Order: Character Unigrams", char_unigram, text, args.length)
            case "char2":
                _try_run("2nd Order: Character Bigrams", lambda t, l: generate_from_char_model(build_char_ngram_model(t, 2), l), text, args.length)
            case "char3":
                _try_run("3rd Order: Character Trigrams", lambda t, l: generate_from_char_model(build_char_ngram_model(t, 3), l), text, args.length)
            case "word1":
                _try_run("Word Unigrams", word_unigram, text, min(args.length, 100))
            case "word2":
                _try_run("Word Bigrams", lambda t, l: generate_from_word_model(build_word_ngram_model(t, 2), l), text, min(args.length, 100))
            case "word3":
                _try_run("Word Trigrams", lambda t, l: generate_from_word_model(build_word_ngram_model(t, 3), l), text, min(args.length, 100))


if __name__ == "__main__":
    main()
