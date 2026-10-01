"""
Module 9 Exercise runner: build a small benchmark suite and score two models

Run with:
    uv run python module_09_evaluation/src/main.py

Loads two finished checkpoints (the Module 6 instruct model and the Module 7 GRPO
model) and scores both with the metrics you write in exercise.py. The protocol is
printed first, then each step below runs one of your metrics and tests it:

    1. perplexity: how surprised each model is by held-out text
    2. normalize_answer: make "It is down." and "it is down" compare equal
    3. exact_match: is the greedy answer exactly right?
    4. token_f1: partial credit for overlapping words
    5. task_accuracy: average the scores within each task
    6. suite_score: one headline number, and what it hides
    7. score_multiple_choice: pick an option by likelihood, no generation
    8. pass_at_k: is the right answer anywhere in k sampled tries?

Add --step N to run one step (1-8); the protocol is printed only for a full run.
Add --solution to run the finished answers from solution/exercise.py.
"""

from __future__ import annotations

import argparse
import sys
from functools import cache
from pathlib import Path

import torch

# Make the module root (parent of src/) importable so we can `from exercise import ...`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# With --solution, swap in solution/exercise.py before anything imports `exercise`
from src.solution import use_solution_if_requested

use_solution_if_requested()

from exercise import (
    perplexity,
    normalize_answer,
    exact_match,
    token_f1,
    task_accuracy,
    suite_score,
    score_multiple_choice,
    pass_at_k,
)
from src.display import print_metric_table, print_protocol, sample_pair, shorten
from src.harness import mean_token_loss, option_log_probs, run_cases
from src.prerequisites import probe_own, require
from src.reporting import run_step
from src.suite import load_suite
from src.visualization import save_task_comparison

# One test file per step lives in tests/
from tests.test_step1_perplexity import check_perplexity
from tests.test_step2_normalize import check_normalize_answer
from tests.test_step3_exact_match import check_exact_match
from tests.test_step4_token_f1 import check_token_f1
from tests.test_step5_task_accuracy import check_task_accuracy
from tests.test_step6_suite_score import check_suite_score
from tests.test_step7_multiple_choice import check_score_multiple_choice
from tests.test_step8_pass_at_k import check_pass_at_k

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

# ---------------------------------------------------------------------------
# The protocol. Every one of these knobs moves the scores, which is why a
# reproducible report prints them before it prints any number.
# ---------------------------------------------------------------------------
BLOCK_SIZE = 128         # context window used for the loss and for generation
MAX_NEW_TOKENS = 14      # generation budget per case (longest answer is 12 chars)
N_SAMPLES = 5            # sampled completions per case, for pass@k
TEMPERATURE = 0.8        # sampling temperature for those completions
PASS_K = 5               # the k in the reported pass@k
SEED = 1337

TASK_ORDER = ["uppercase", "repeat", "reverse", "qa"]
# Both checkpoints ship with the repo and were pretrained for this module by
# src/make_checkpoints.py: instruct_model.pt is the Module 5 base model after
# multi-task SFT (the Module 6 story), and rl_model.pt is that same checkpoint after GRPO
# on `reverse` only (the Module 7 story).
MODELS = [("instruct", "instruct_model.pt"), ("rl", "rl_model.pt")]



@cache  # load once, then every step shares the same models and data
def suite():
    """Both checkpoints, the tokenizer maps, and every evaluation file (see src/suite.py)."""
    return load_suite(MODELS, TASK_ORDER)


@cache  # generation is the slow part, so it runs once and later steps reuse it
def answers() -> dict[str, list[dict]]:
    """Both models' greedy and sampled answers for every case.

    Returns {"instruct": [...], "rl": [...]}, one record per case with the
    case's acceptable "answers", the "greedy" answer, and the list of "samples".
    """
    s = suite()
    print(f"  Generating {len(s.cases)} greedy + {len(s.cases) * N_SAMPLES}"
          f" sampled answers per model...")
    return {
        name: run_cases(model, s.cases, s.special, s.enc, s.itos,
                        max_new_tokens=MAX_NEW_TOKENS, block_size=BLOCK_SIZE,
                        n_samples=N_SAMPLES, temperature=TEMPERATURE, seed=SEED)
        for name, model in s.models.items()
    }


# ---------------------------------------------------------------------------
# Scoring one answer with your metrics, then averaging per task
# ---------------------------------------------------------------------------


def greedy_em(rec: dict) -> float:
    """Exact match of the greedy answer (your Step 3)."""
    return exact_match(rec["greedy"], rec["answers"])


def greedy_f1(rec: dict) -> float:
    """Token F1 of the greedy answer against its closest acceptable answer (your Step 4)."""
    return max(token_f1(rec["greedy"], answer) for answer in rec["answers"])


def samples_correct(rec: dict) -> int:
    """How many of the N_SAMPLES sampled answers are exactly right."""
    return sum(int(exact_match(s, rec["answers"])) for s in rec["samples"])


def per_task_scores(records: list[dict], score) -> dict[str, float]:
    """Apply `score(record)` to every case, group by task, and average with your task_accuracy()."""
    grouped: dict[str, list[float]] = {task: [] for task in TASK_ORDER}
    for rec in records:
        grouped[rec["task"]].append(score(rec))
    per_task = task_accuracy(grouped)
    if not isinstance(per_task, dict):
        raise TypeError("task_accuracy() should return a dict of task -> mean score,"
                        f" got a {type(per_task).__name__}")
    return per_task


# ---------------------------------------------------------------------------
# Step 1: perplexity on held-out text (no labels, no generation)
# ---------------------------------------------------------------------------


def step_1() -> str:
    def show():
        perplexity(1.0)  # an unfinished perplexity() stops here with its TODO
        s = suite()
        print(f"    {'model':<12}{'loss (nats)':>14}{'perplexity':>14}")
        for name, model in s.models.items():
            loss = mean_token_loss(model, s.held_out, s.stoi, BLOCK_SIZE)
            print(f"    {name:<12}{loss:>14.4f}{perplexity(loss):>14.2f}")
        print("  Lower is better. This says nothing about whether either model")
        print("  follows instructions, which is exactly its limitation.")

    return run_step("Step 1: perplexity()", show, lambda: check_perplexity(perplexity))


# ---------------------------------------------------------------------------
# Step 2: the same answer, formatted four ways, after normalization
# ---------------------------------------------------------------------------


def step_2() -> str:
    def show():
        print("  One answer formatted four ways, after normalize_answer():")
        for raw in ("It is down.", "  it is   down ", "IT IS DOWN!", "it is\tdown\n"):
            print(f"    {raw!r:<20} -> {normalize_answer(raw)!r}")

    return run_step("Step 2: normalize_answer()", show,
                    lambda: check_normalize_answer(normalize_answer))


# ---------------------------------------------------------------------------
# Step 3: exact match on the greedy answers
# ---------------------------------------------------------------------------


def step_3() -> str:
    def show():
        probe_own(exact_match, "blue", ["blue"], why="compare answers")
        records = answers()
        total = len(suite().cases)
        # How many of the 50 greedy answers each model gets exactly right
        hits = {name: sum(greedy_em(rec) for rec in recs) for name, recs in records.items()}
        print(f"  Greedy answers that match exactly: instruct {hits['instruct']:.0f}/{total},"
              f" rl {hits['rl']:.0f}/{total}")
        print("  One case per task, with its exact-match score:")
        for task in TASK_ORDER:
            left, right = sample_pair(records, task)
            print(f"    [{task}] {left['prompt']!r}   want {left['answers'][0]!r}")
            for name, rec in (("instruct", left), ("rl", right)):
                shown = repr(shorten(rec["greedy"]))
                print(f"        {name:<8} {shown:<28} {greedy_em(rec):.1f}")

    return run_step("Step 3: exact_match()", show, lambda: check_exact_match(exact_match))


# ---------------------------------------------------------------------------
# Step 4: token F1, and where it gives partial credit
# ---------------------------------------------------------------------------


def step_4() -> str:
    def show():
        probe_own(token_f1, "it is blue", "it is blue", why="split answers into tokens")
        records = answers()
        # Every (model, case, F1) where F1 is neither a full miss nor a full hit
        partial = [(name, rec, greedy_f1(rec))
                   for name, recs in records.items() for rec in recs
                   if 0.0 < greedy_f1(rec) < 1.0]
        print("  Greedy answers that earn partial credit (F1 between 0 and 1):")
        for name, rec, f1 in partial[:3]:  # a few examples are enough
            print(f"    [{rec['task']}] {rec['prompt']!r}   want {rec['answers'][0]!r}")
            shown = repr(shorten(rec["greedy"]))
            print(f"        {name:<8} {shown:<28} {f1:.2f}")
        if len(partial) > 3:
            print(f"    ... and {len(partial) - 3} more")
        if not partial:
            print("    none: every greedy answer scores exactly 0 or 1")

    return run_step("Step 4: token_f1()", show, lambda: check_token_f1(token_f1))


# ---------------------------------------------------------------------------
# Step 5: per-task tables for exact match and F1
# ---------------------------------------------------------------------------


def step_5() -> str:
    def show():
        task_accuracy({"qa": [1.0, 0.0]})  # an unfinished task_accuracy() stops here
        require("Step 3 (exact_match)", lambda: exact_match("blue", ["blue"]),
                "score the answers it averages")
        require("Step 4 (token_f1)", lambda: token_f1("blue", "blue"),
                "score the answers it averages")
        records = answers()
        tables = [
            (greedy_em, "EXACT MATCH (greedy decoding)",
             "The strictest metric: the normalized answer must equal an acceptable answer."),
            (greedy_f1, "TOKEN F1 (greedy decoding)",
             "Partial credit for overlapping tokens; equals exact match on one-word answers."),
        ]
        for i, (score, title, note) in enumerate(tables):
            if i > 0:
                print()
            per_task = {name: per_task_scores(recs, score) for name, recs in records.items()}
            print_metric_table(title, note, per_task, suite().counts)

    return run_step("Step 5: task_accuracy()", show, lambda: check_task_accuracy(task_accuracy))


# ---------------------------------------------------------------------------
# Step 6: the headline number, what it hides, and the bar chart
# ---------------------------------------------------------------------------


def step_6() -> str:
    def show():
        suite_score({"qa": 0.5})  # an unfinished suite_score() stops here with its TODO
        require("Step 3 (exact_match)", lambda: exact_match("blue", ["blue"]),
                "score the answers")
        require("Step 5 (task_accuracy)", lambda: task_accuracy({"qa": [1.0]}),
                "average the scores within each task")
        records = answers()
        per_task_em, overall, pooled = {}, {}, {}
        for name, recs in records.items():
            per_task_em[name] = per_task_scores(recs, greedy_em)
            overall[name] = suite_score(per_task_em[name])
            # The other protocol: one average over all 50 cases, ignoring tasks
            pooled[name] = sum(greedy_em(rec) for rec in recs) / len(recs)
            print(f"    {name:<12}{overall[name]:>8.1%}   (mean of the four task scores)")
        print(f"  Overall difference: {overall['rl'] - overall['instruct']:+.1%}")
        print(f"  Averaging all {len(suite().cases)} cases instead gives instruct {pooled['instruct']:.1%},"
              f" rl {pooled['rl']:.1%} ({pooled['rl'] - pooled['instruct']:+.1%}).")
        print("  Read the per-task table before believing either number.")

        save_task_comparison(
            TASK_ORDER,
            [per_task_em["instruct"][t] for t in TASK_ORDER],
            [per_task_em["rl"][t] for t in TASK_ORDER],
            OUTPUT_DIR / "task_comparison.png",
            overall["instruct"],
            overall["rl"],
        )
        print("  Saved chart to output/task_comparison.png")

    return run_step("Step 6: suite_score()", show, lambda: check_suite_score(suite_score))


# ---------------------------------------------------------------------------
# Step 7: multiple choice, scored by likelihood (nothing is generated)
# ---------------------------------------------------------------------------


def step_7() -> str:
    def show():
        score_multiple_choice([-1.0, -2.0], [2, 2])  # an unfinished function stops here
        s = suite()
        for name, model in s.models.items():
            correct = 0
            for case in s.mc_cases:
                # The log-probability of each option, appended to the question
                totals, lengths = option_log_probs(model, case["question"], case["options"],
                                                   s.special, s.enc)
                correct += int(score_multiple_choice(totals, lengths) == case["answer_index"])
            accuracy = correct / len(s.mc_cases)
            print(f"    {name:<12}{correct:>3}/{len(s.mc_cases)}   {accuracy:>6.1%}")
        print("  Chance is 25%. Nothing was generated: each option was scored under")
        print("  the model and the highest per-token log-probability won.")

    return run_step("Step 7: score_multiple_choice()", show,
                    lambda: check_score_multiple_choice(score_multiple_choice))


# ---------------------------------------------------------------------------
# Step 8: pass@1 and pass@k from the sampled answers
# ---------------------------------------------------------------------------


def step_8() -> str:
    def show():
        pass_at_k(5, 2, 3)  # an unfinished pass_at_k() stops here with its TODO
        require("Step 3 (exact_match)", lambda: exact_match("blue", ["blue"]),
                "check each sampled answer")
        require("Step 5 (task_accuracy)", lambda: task_accuracy({"qa": [1.0]}),
                "average the estimates within each task")
        records = answers()
        tables = [
            (1, f"pass@1 (sampled at temperature {TEMPERATURE})",
             f"Accuracy of a single sampled answer, estimated from N={N_SAMPLES} samples per case."),
            (PASS_K, f"pass@{PASS_K} (sampled at temperature {TEMPERATURE})",
             f"Is the right answer anywhere in the model's distribution across {PASS_K} tries?"),
        ]
        for i, (k, title, note) in enumerate(tables):
            if i > 0:
                print()

            def estimate(rec: dict) -> float:
                """pass@k for one case, from how many of its samples were right."""
                return pass_at_k(N_SAMPLES, samples_correct(rec), k)

            per_task = {name: per_task_scores(recs, estimate) for name, recs in records.items()}
            print_metric_table(title, note, per_task, suite().counts)

    return run_step("Step 8: pass_at_k()", show, lambda: check_pass_at_k(pass_at_k))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

STEPS = {"1": step_1, "2": step_2, "3": step_3, "4": step_4,
         "5": step_5, "6": step_6, "7": step_7, "8": step_8}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a small benchmark suite and score two models")
    parser.add_argument("--step", choices=[*STEPS, "all"], default="all",
                        help="Which step to run (default: all)")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    if args.step == "all":
        # A full report states its protocol before any number
        print_protocol(suite(), block_size=BLOCK_SIZE, max_new_tokens=MAX_NEW_TOKENS,
                       n_samples=N_SAMPLES, temperature=TEMPERATURE, seed=SEED)
    for number, step in STEPS.items():
        if args.step in ("all", number):
            step()


if __name__ == "__main__":
    main()
