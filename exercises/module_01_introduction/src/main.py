"""
Module 1 Exercise: N-gram Text Generator

Run with:
    uv run python module_01_introduction/src/main.py

Add --solution to run the finished answers from solution/exercise.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the module root (parent of src/) importable so we can `import exercise`
_MODULE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_MODULE_DIR))

# `--solution` swaps in the finished answers from solution/exercise.py.
# Registering it as "exercise" before the imports below means every
# `from exercise import ...` in this file picks it up with no other change.
if "--solution" in sys.argv:
    import importlib.util

    sys.argv.remove("--solution")
    _sol = Path(__file__).resolve().parent.parent / "solution" / "exercise.py"
    _spec = importlib.util.spec_from_file_location("exercise", _sol)
    _exercise = importlib.util.module_from_spec(_spec)
    sys.modules["exercise"] = _exercise
    _spec.loader.exec_module(_exercise)

from exercise import (
    build_char_ngram_model,
    build_word_ngram_model,
    char_uniform,
    char_unigram,
    load_text,
    word_unigram,
)

# Provided helpers: the sampling loops that turn a count table into text
from src.sampling import (
    generate_from_char_model,
    generate_from_word_model,
)

DEFAULT_CORPUS = _MODULE_DIR / "data" / "alice.txt"
DEFAULT_CORPUS_LABEL = Path("module_01_introduction/data/alice.txt")


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
        text = load_text(str(DEFAULT_CORPUS))
        print(f"Loaded {len(text)} characters from {DEFAULT_CORPUS_LABEL}\n")
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
