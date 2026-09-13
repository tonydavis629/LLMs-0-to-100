"""
Module 1 Exercise: N-gram Text Generator

Run with:
    uv run python module_01_introduction/src/main.py

Every step is tagged on its header line, then its output follows: the text
your code generated and the result of each test in tests/. The tags are:

    CORRECT     every test for the step passed
    INCORRECT   your code ran but at least one test failed (details follow)
    INCOMPLETE  the function still raises NotImplementedError

Add --step N to run one step (1-6, or "ec" for extra credit).
Add --solution to run the finished answers from solution/exercise.py.
"""

from __future__ import annotations

import argparse
import io
import sys
from contextlib import redirect_stdout
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
    cross_entropy,
    load_text,
    word_unigram,
)

# Provided helpers: the sampling loops that turn a count table into text,
# and the perplexity metric for the extra credit
from src.metrics import perplexity
from src.sampling import generate_from_char_model, generate_from_word_model

# One test file per step lives in tests/
from tests.test_extra_credit import check_cross_entropy
from tests.test_step1_load_text import check_load_text
from tests.test_step2_char_uniform import check_char_uniform
from tests.test_step3_char_unigram import check_char_unigram
from tests.test_step4_char_ngram_model import check_build_char_ngram_model
from tests.test_step5_word_unigram import check_word_unigram
from tests.test_step6_word_ngram_model import check_build_word_ngram_model

DEFAULT_CORPUS = _MODULE_DIR / "data" / "alice.txt"
DEFAULT_CORPUS_LABEL = Path("module_01_introduction/data/alice.txt")

# The three possible outcomes for a step
CORRECT = "CORRECT"
INCORRECT = "INCORRECT"
INCOMPLETE = "INCOMPLETE"

# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

# ANSI color codes, used only when printing to a real terminal
_COLORS = {CORRECT: "\033[32m", INCORRECT: "\033[31m", INCOMPLETE: "\033[90m"}
_RESET = "\033[0m"


def _tag(status: str) -> str:
    """Format a status label in a fixed-width column, colored on a terminal."""
    label = f"{status:<10}"
    if sys.stdout.isatty():
        return f"{_COLORS[status]}{label}{_RESET}"
    return label


def _print_checks(checks) -> None:
    """Print one line per test, with details under any that failed."""
    for check in checks:
        print(f"  {_tag(CORRECT if check.passed else INCORRECT)} {check.name}")
        if not check.passed and check.detail:
            for line in check.detail.split("\n"):
                print(f"             {line.strip()}")


def run_step(title: str, show, check) -> str:
    """Run one step and print its header, tag, output, and test results.

    `show()` prints whatever the student's code generates.
    `check()` returns the list of Check results for the step.

    The tag goes on the header line, so the output is captured first and
    printed after the tag is known. Returns CORRECT, INCORRECT, or INCOMPLETE.
    """
    buffer = io.StringIO()
    checks = []
    note = ""
    try:
        with redirect_stdout(buffer):
            show()
        checks = check()
        status = CORRECT if all(c.passed for c in checks) else INCORRECT
    except NotImplementedError as e:
        # The student has not filled in this blank yet
        status, note = INCOMPLETE, str(e)
    except Exception as e:  # noqa: BLE001 - show students any crash, whatever its type
        status, note = INCORRECT, f"your code crashed: {type(e).__name__}: {e}"

    print(f"=== {title} === {_tag(status).rstrip()}")
    if note:
        print(f"  {note}")
    output = buffer.getvalue()
    if output:
        print(output, end="" if output.endswith("\n") else "\n")
    _print_checks(checks)
    print()
    return status


def _needs_text(title: str) -> str:
    """Report a step that cannot run because load_text() is unfinished."""
    print(f"=== {title} === {_tag(INCOMPLETE).rstrip()}")
    print("  needs the corpus: finish Step 1 (load_text) first")
    print()
    return INCOMPLETE


# ---------------------------------------------------------------------------
# The steps
# ---------------------------------------------------------------------------


def step_1(text_holder: dict) -> str:
    """Load the corpus. Later steps read it from text_holder["text"]."""

    def show():
        text = load_text(str(DEFAULT_CORPUS))
        text_holder["text"] = text
        print(f"Loaded {len(text)} characters from {DEFAULT_CORPUS_LABEL}")

    return run_step("Step 1: load_text()", show, lambda: check_load_text(text_holder["text"]))


def step_2(length: int) -> str:
    return run_step(
        "Step 2: char_uniform()",
        lambda: print(char_uniform(length)),
        lambda: check_char_uniform(char_uniform),
    )


def step_3(text: str, length: int) -> str:
    return run_step(
        "Step 3: char_unigram()",
        lambda: print(char_unigram(text, length)),
        lambda: check_char_unigram(char_unigram),
    )


def step_4(text: str, length: int) -> str:
    def show():
        # One fill-in gives both models: the sampling loop is provided
        for n, name in ((2, "bigram"), (3, "trigram")):
            model = build_char_ngram_model(text, n)
            print(f"--- {name} (n={n}) ---")
            print(generate_from_char_model(model, length))

    return run_step(
        "Step 4: build_char_ngram_model()",
        show,
        lambda: check_build_char_ngram_model(build_char_ngram_model, text),
    )


def step_5(text: str, words: int) -> str:
    return run_step(
        "Step 5: word_unigram()",
        lambda: print(word_unigram(text, words)),
        lambda: check_word_unigram(word_unigram),
    )


def step_6(text: str, words: int) -> str:
    def show():
        for n, name in ((2, "bigram"), (3, "trigram")):
            model = build_word_ngram_model(text, n)
            print(f"--- {name} (n={n}) ---")
            print(generate_from_word_model(model, words))

    return run_step(
        "Step 6: build_word_ngram_model()",
        show,
        lambda: check_build_word_ngram_model(build_word_ngram_model, text),
    )


def step_extra(text: str) -> str:
    """Extra credit: score each n-gram order on held-out text."""

    def show():
        # Hold out the last 10% of the book so the model is scored on text
        # it never saw during training
        lower = text.lower()
        split = int(len(lower) * 0.9)
        train, held = lower[:split], lower[split:]

        # The models to score come from Step 4, so that has to be done first
        try:
            models = [build_char_ngram_model(train, n) for n in range(1, 6)]
        except NotImplementedError:
            raise NotImplementedError("needs Step 4 (build_char_ngram_model) to build the models it scores")

        # Score every model before printing, so an unfinished cross_entropy()
        # shows INCOMPLETE instead of a half-printed table
        rows = [(n, cross_entropy(held, m), perplexity(held, m)) for n, m in enumerate(models, 1)]

        print("Train on the first 90% of Alice, score the held-out last 10%:")
        print()
        print("  order   bits/char   perplexity")
        for n, bits, ppl in rows:
            print(f"  n={n}     {bits:8.3f}   {ppl:10.2f}")
        print()

    return run_step(
        "Extra Credit: cross_entropy()",
        show,
        lambda: check_cross_entropy(cross_entropy, text),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

STEP_NAMES = {
    "1": "load_text()",
    "2": "char_uniform()",
    "3": "char_unigram()",
    "4": "build_char_ngram_model()",
    "5": "word_unigram()",
    "6": "build_word_ngram_model()",
    "ec": "cross_entropy()",
}


def main():
    parser = argparse.ArgumentParser(description="N-gram text generator")
    parser.add_argument(
        "--step",
        choices=[*STEP_NAMES, "all"],
        default="all",
        help="Which step to run (default: all). Step 1 always runs, since every other step needs the corpus.",
    )
    parser.add_argument("--length", type=int, default=500, help="Characters of text to generate per character model")
    parser.add_argument("--words", type=int, default=100, help="Words of text to generate per word model")
    args = parser.parse_args()

    steps = list(STEP_NAMES) if args.step == "all" else ["1", args.step]
    results: dict[str, str] = {}

    # Step 1 always runs first: every other step needs the corpus it loads
    holder: dict = {}
    results["1"] = step_1(holder)
    text = holder.get("text")

    for step in steps[1:]:
        title = f"Step {step}: {STEP_NAMES[step]}" if step != "ec" else f"Extra Credit: {STEP_NAMES[step]}"
        if step == "2":
            results[step] = step_2(args.length)
        elif text is None:
            # Everything past Step 2 reads the corpus, so it cannot run yet
            results[step] = _needs_text(title)
        elif step == "3":
            results[step] = step_3(text, args.length)
        elif step == "4":
            results[step] = step_4(text, args.length)
        elif step == "5":
            results[step] = step_5(text, args.words)
        elif step == "6":
            results[step] = step_6(text, args.words)
        elif step == "ec":
            results[step] = step_extra(text)



if __name__ == "__main__":
    main()

