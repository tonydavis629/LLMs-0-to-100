"""
Module 1 Exercise runner: N-gram Text Generator

Run with:
    uv run python module_01_introduction/src/main.py

Each step below runs one function you write in exercise.py, prints the text
it generates, then tests it. Read top to bottom, the steps go from random
noise to text that starts to look like Alice in Wonderland:

    1. load the book
    2. random characters, all equally likely
    3. random characters, drawn as often as they appear in the book
    4. characters that depend on the previous 1 or 2 characters (bigram, trigram)
    5. random words, drawn as often as they appear in the book
    6. words that depend on the previous 1 or 2 words
    Extra credit: score each model on text it never saw

Add --step N to run one step (1-6, or "ec" for extra credit).
Add --solution to run the finished answers from solution/exercise.py.
"""

from __future__ import annotations

import argparse
import sys
from functools import cache
from pathlib import Path

# Make the module root (parent of src/) importable so we can `from exercise import ...`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# With --solution, swap in solution/exercise.py before anything imports `exercise`
from src.solution import use_solution_if_requested

use_solution_if_requested()

from exercise import (
    build_char_ngram_model,
    build_word_ngram_model,
    char_uniform,
    char_unigram,
    cross_entropy,
    load_text,
    word_unigram,
)
from src.metrics import perplexity
from src.reporting import run_step
from src.sampling import generate_from_char_model, generate_from_word_model

# One test file per step lives in tests/
from tests.test_extra_credit import check_cross_entropy
from tests.test_step1_load_text import check_load_text
from tests.test_step2_char_uniform import check_char_uniform
from tests.test_step3_char_unigram import check_char_unigram
from tests.test_step4_char_ngram_model import check_build_char_ngram_model
from tests.test_step5_word_unigram import check_word_unigram
from tests.test_step6_word_ngram_model import check_build_word_ngram_model

CORPUS = Path(__file__).resolve().parent.parent / "data" / "alice.txt"
CORPUS_LABEL = "module_01_introduction/data/alice.txt"  # the path as printed


@cache  # read the book once, then reuse the same text in every step
def corpus() -> str:
    """Return the book as one string, loaded with your load_text() from Step 1."""
    try:
        return load_text(str(CORPUS))
    except NotImplementedError:
        raise NotImplementedError("needs the corpus: finish Step 1 (load_text) first") from None


# ---------------------------------------------------------------------------
# Step 1: load the book
# ---------------------------------------------------------------------------


def step_1(args) -> str:
    def show():
        text = load_text(str(CORPUS))  # called directly, so an unfinished Step 1 shows its own TODO
        print(f"Loaded {len(text)} characters from {CORPUS_LABEL}")

    return run_step("Step 1: load_text()", show, lambda: check_load_text(corpus()))


# ---------------------------------------------------------------------------
# Step 2: random characters, all equally likely
# ---------------------------------------------------------------------------


def step_2(args) -> str:
    return run_step("Step 2: char_uniform()",
                    lambda: print(char_uniform(args.length)),
                    lambda: check_char_uniform(char_uniform))


# ---------------------------------------------------------------------------
# Step 3: random characters, as often as they appear in the book
# ---------------------------------------------------------------------------


def step_3(args) -> str:
    return run_step("Step 3: char_unigram()",
                    lambda: print(char_unigram(corpus(), args.length)),
                    lambda: check_char_unigram(char_unigram))


# ---------------------------------------------------------------------------
# Step 4: characters that depend on the characters before them
# ---------------------------------------------------------------------------


def step_4(args) -> str:
    def show():
        text = corpus()
        # Your function builds the count table; generate_from_char_model() samples from it
        for n, name in ((2, "bigram"), (3, "trigram")):
            model = build_char_ngram_model(text, n)
            print(f"--- {name} (n={n}) ---")
            print(generate_from_char_model(model, args.length))

    return run_step("Step 4: build_char_ngram_model()", show,
                    lambda: check_build_char_ngram_model(build_char_ngram_model, corpus()))


# ---------------------------------------------------------------------------
# Step 5: random words, as often as they appear in the book
# ---------------------------------------------------------------------------


def step_5(args) -> str:
    return run_step("Step 5: word_unigram()",
                    lambda: print(word_unigram(corpus(), args.words)),
                    lambda: check_word_unigram(word_unigram))


# ---------------------------------------------------------------------------
# Step 6: words that depend on the words before them
# ---------------------------------------------------------------------------


def step_6(args) -> str:
    def show():
        text = corpus()
        for n, name in ((2, "bigram"), (3, "trigram")):
            model = build_word_ngram_model(text, n)
            print(f"--- {name} (n={n}) ---")
            print(generate_from_word_model(model, args.words))

    return run_step("Step 6: build_word_ngram_model()", show,
                    lambda: check_build_word_ngram_model(build_word_ngram_model, corpus()))


# ---------------------------------------------------------------------------
# Extra credit: score each model on text it never saw
# ---------------------------------------------------------------------------


def step_extra(args) -> str:
    def show():
        # Hold out the last 10% of the book so the model is scored on text
        # it never saw during training
        text = corpus().lower()
        split = int(len(text) * 0.9)
        train, held = text[:split], text[split:]

        # The models to score come from Step 4, so that has to be done first
        try:
            models = [build_char_ngram_model(train, n) for n in range(1, 6)]
        except NotImplementedError:
            raise NotImplementedError("needs Step 4 (build_char_ngram_model) to build the models it scores") from None

        # Score every model before printing, so an unfinished cross_entropy()
        # shows INCOMPLETE instead of a half-printed table
        rows = [(n, cross_entropy(held, m), perplexity(held, m)) for n, m in enumerate(models, 1)]

        print("Train on the first 90% of Alice, score the held-out last 10%:")
        print()
        print("  order   bits/char   perplexity")
        for n, bits, ppl in rows:
            print(f"  n={n}     {bits:8.3f}   {ppl:10.2f}")
        print()

    return run_step("Extra Credit: cross_entropy()", show,
                    lambda: check_cross_entropy(cross_entropy, corpus()))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

STEPS = {"1": step_1, "2": step_2, "3": step_3, "4": step_4, "5": step_5, "6": step_6, "ec": step_extra}


def main():
    parser = argparse.ArgumentParser(description="N-gram text generator")
    parser.add_argument("--step", choices=[*STEPS, "all"], default="all",
                        help="Which step to run (default: all). Step 1 always runs, since every other step needs the corpus.")
    parser.add_argument("--length", type=int, default=500, help="Characters of text to generate per character model")
    parser.add_argument("--words", type=int, default=100, help="Words of text to generate per word model")
    args = parser.parse_args()

    for number, step in STEPS.items():
        # Step 1 always runs: every other step needs the book it loads
        if args.step in ("all", number) or number == "1":
            step(args)


if __name__ == "__main__":
    main()
