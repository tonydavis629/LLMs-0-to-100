"""
Module 1 Solution: Text Generation CLI

Run with: uv run generate
"""

from __future__ import annotations

import argparse

from .ngrams import (
    build_char_ngram_model,
    build_word_ngram_model,
    char_uniform,
    char_unigram,
    cross_entropy,
    generate_from_char_model,
    generate_from_word_model,
    load_text,
    perplexity,
    word_unigram,
)


def main():
    parser = argparse.ArgumentParser(description="N-gram text generator")
    parser.add_argument("corpus", help="Path to text file (e.g., data/alice.txt)")
    parser.add_argument(
        "--model",
        choices=["uniform", "char1", "char2", "char3", "word1", "word2", "word3", "all"],
        default="all",
        help="Which model to use (default: all)",
    )
    parser.add_argument("--length", type=int, default=500, help="Length of generated text")
    parser.add_argument("--perplexity", action="store_true", help="Show cross-entropy and perplexity")
    args = parser.parse_args()

    text = load_text(args.corpus)
    print(f"Loaded {len(text)} characters from {args.corpus}\n")

    models_to_run = (
        ["uniform", "char1", "char2", "char3", "word1", "word2", "word3"]
        if args.model == "all"
        else [args.model]
    )

    for model_name in models_to_run:
        match model_name:
            case "uniform":
                print("=== 0th Order: Uniform Random Characters ===")
                print(char_uniform(args.length))

            case "char1":
                print("\n=== 1st Order: Character Unigrams ===")
                print(char_unigram(text, args.length))

            case "char2":
                print("\n=== 2nd Order: Character Bigrams ===")
                model = build_char_ngram_model(text, 2)
                print(generate_from_char_model(model, args.length))
                if args.perplexity:
                    h = cross_entropy(text, model)
                    print(f"  Cross-entropy: {h:.2f} bits/char | Perplexity: {perplexity(text, model):.2f}")

            case "char3":
                print("\n=== 3rd Order: Character Trigrams ===")
                model = build_char_ngram_model(text, 3)
                print(generate_from_char_model(model, args.length))
                if args.perplexity:
                    h = cross_entropy(text, model)
                    print(f"  Cross-entropy: {h:.2f} bits/char | Perplexity: {perplexity(text, model):.2f}")

            case "word1":
                length = min(args.length, 100)
                print("\n=== Word Unigrams ===")
                print(word_unigram(text, length))

            case "word2":
                length = min(args.length, 100)
                print("\n=== Word Bigrams ===")
                model = build_word_ngram_model(text, 2)
                print(generate_from_word_model(model, length))

            case "word3":
                length = min(args.length, 100)
                print("\n=== Word Trigrams ===")
                model = build_word_ngram_model(text, 3)
                print(generate_from_word_model(model, length))

        print()


if __name__ == "__main__":
    main()
