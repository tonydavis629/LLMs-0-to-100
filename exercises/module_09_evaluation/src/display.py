"""Printing the protocol and the score tables, provided for you.

You do NOT need to edit this file. These functions only format numbers that
src/main.py has already computed with your metrics.
"""

from __future__ import annotations


def print_protocol(suite, *, block_size: int, max_new_tokens: int, n_samples: int,
                   temperature: float, seed: int) -> None:
    """The protocol comes first. A benchmark number belongs to a model AND a
    protocol, so a report that does not state one cannot be reproduced."""
    print("MODULE 9: evaluating two finished checkpoints")
    print()
    print("Models under test")
    print("  instruct  data/instruct_model.pt   Module 6: multi-task SFT")
    print("  rl        data/rl_model.pt         Module 7: GRPO on `reverse` only")
    print("  Both are the same TinyGPT architecture, tokenizer, and chat template.")
    print()
    print("Protocol")
    print("  chat template     <|user|> PROMPT <|end|> <|assistant|> ANSWER <|end|>")
    print("  normalization     lowercase, strip punctuation, collapse whitespace")
    print(f"  generation budget {max_new_tokens} tokens    context {block_size}")
    print("  decoding          greedy for exact match and F1")
    print(f"                    {n_samples} samples at temperature {temperature} for pass@k")
    print(f"  seed              {seed} (per case, so both models see the same draws)")
    print(f"  suite             {len(suite.cases)} generated cases across {len(suite.counts)} tasks,"
          f" {len(suite.mc_cases)} multiple-choice questions")
    print(f"  held-out text     {len(suite.held_out):,} characters, never seen in training")
    print()


def sample_pair(records: dict[str, list[dict]], task: str) -> tuple[dict, dict]:
    """One case from `task`, as answered by each model.

    Prefers a case where the two models disagree, since that is the informative
    one; falls back to the first case of the task.
    """
    pairs = [(a, b) for a, b in zip(records["instruct"], records["rl"]) if a["task"] == task]
    return next(((a, b) for a, b in pairs if a["greedy"] != b["greedy"]), pairs[0])


def shorten(text: str, limit: int = 22) -> str:
    """Trim a generation for display. A model that never emits <|end|> runs long."""
    return text if len(text) <= limit else text[:limit] + "..."


def _pct(value: float | None) -> str:
    """Format a score as a percentage, or a placeholder when it is missing."""
    return "   --" if value is None else f"{value:6.1%}"


def print_metric_table(title: str, note: str, per_task: dict[str, dict[str, float]],
                       counts: dict[str, int]) -> None:
    """Print one metric's per-task numbers for both models, plus the difference."""
    print(f"  {title}")
    print(f"  {note}")
    print(f"    {'task':<12}{'cases':>7}{'instruct':>12}{'rl':>10}{'diff':>10}")
    for task in counts:
        left = per_task["instruct"].get(task)
        right = per_task["rl"].get(task)
        diff = "" if left is None or right is None else f"{right - left:+9.1%}"
        print(f"    {task:<12}{counts[task]:>7}{_pct(left):>12}{_pct(right):>10}{diff:>10}")
