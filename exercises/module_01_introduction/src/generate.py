"""
Module 1 Exercise: Text Generation CLI

Run with:
    uv run python -m exercises.module_01_introduction.src.generate \
        exercises/module_01_introduction/data/alice.txt

Options:
    --model uniform|char1|char2|char3|word1|word2|word3|all
    --length N
"""

from __future__ import annotations

import argparse

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


def _run_model(model_name: str, text: str, length: int) -> None:
    match model_name:
        case "uniform":
            print("=== 0th Order: Uniform Random Characters ===")
            print(char_uniform(length))

        case "char1":
            print("=== 1st Order: Character Unigrams ===")
            print(char_unigram(text, length))

        case "char2":
            print("=== 2nd Order: Character Bigrams ===")
            model = build_char_ngram_model(text, 2)
            print(generate_from_char_model(model, length))

        case "char3":
            print("=== 3rd Order: Character Trigrams ===")
            model = build_char_ngram_model(text, 3)
            print(generate_from_char_model(model, length))

        case "word1":
            wlen = min(length, 100)
            print("=== Word Unigrams ===")
            print(word_unigram(text, wlen))

        case "word2":
            wlen = min(length, 100)
            print("=== Word Bigrams ===")
            model = build_word_ngram_model(text, 2)
            print(generate_from_word_model(model, wlen))

        case "word3":
            wlen = min(length, 100)
            print("=== Word Trigrams ===")
            model = build_word_ngram_model(text, 3)
            print(generate_from_word_model(model, wlen))

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

    text = load_text(args.corpus)
    print(f"Loaded {len(text)} characters from {args.corpus}\n")

    if args.model == "all":
        for name in ["uniform", "char1", "char2", "char3", "word1", "word2", "word3"]:
            _run_model(name, text, args.length)
    else:
        _run_model(args.model, text, args.length)


if __name__ == "__main__":
    main()
