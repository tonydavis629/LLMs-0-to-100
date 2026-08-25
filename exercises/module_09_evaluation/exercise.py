"""
Module 9 Exercise: Build a small benchmark suite and score two models

You implement the eight metrics below. Everything else (the two checkpoints, the
tokenizer, the sampler, the evaluation data, the plotting, and the runner) is
provided. Each blank is one line or one short expression.

Nothing here trains a model. Both checkpoints are finished: `instruct_model.pt` is
the Module 6 instruction-tuned model, and `rl_model.pt` is that same model after
Module 7's GRPO run on the reverse task. The question this exercise answers is the
one every model release has to answer: which of these two is better, and better at
what? The per-task table at the end is the whole point &mdash; the RL model wins the
task it was trained on, and the average alone would hide what that cost.

Run after each step; unfinished steps are skipped automatically:
    uv run python module_09_evaluation/src/main.py
"""

from __future__ import annotations

import math
import string
from collections import Counter


# ---------------------------------------------------------------------------
# Step 1: Perplexity
# ---------------------------------------------------------------------------


def perplexity(mean_token_loss: float) -> float:
    """Convert an average per-token cross-entropy loss (in nats) into perplexity.

    This is the base-model metric from Module 5, and the only number in this suite
    that needs no labels at all: just held-out text and the model's own loss on it.
    Perplexity is the model's average branching factor &mdash; roughly, how many
    equally likely tokens it is choosing among at each position. Lower is better,
    and a perplexity of 1.0 would mean the model was never surprised.

    Args:
        mean_token_loss: Average -log p(token) over the held-out tokens, in nats.

    Returns:
        The perplexity as a plain float.
    """
    # TODO: Return the perplexity that corresponds to this average loss.
    # HINT: perplexity is the exponential of the mean loss; math.exp does this.
    raise NotImplementedError("TODO: convert average token loss into perplexity")


# ---------------------------------------------------------------------------
# Step 2: Answer normalization
# ---------------------------------------------------------------------------


def normalize_answer(text: str) -> str:
    """Put a generated answer into the canonical form that scoring compares.

    "4", " 4 ", and "4." are the same answer and three different strings, so every
    benchmark ships a normalization step before it compares anything. Without one,
    exact match measures formatting instead of correctness. The first two lines are
    provided; you write the whitespace rule.

    Returns:
        The normalized string.
    """
    lowered = text.lower()
    stripped = "".join(ch for ch in lowered if ch not in string.punctuation)
    # TODO: Return `stripped` with leading and trailing whitespace removed and every
    #       run of internal whitespace collapsed to a single space.
    # HINT: .split() with no argument splits on any run of whitespace and drops the
    #       empties; " ".join(...) puts the pieces back together.
    raise NotImplementedError("TODO: collapse the whitespace in the normalized answer")


# ---------------------------------------------------------------------------
# Step 3: Exact match
# ---------------------------------------------------------------------------


def exact_match(prediction: str, answers: list[str]) -> float:
    """Score 1.0 if the normalized prediction equals any acceptable answer, else 0.0.

    Benchmarks carry a *list* of acceptable answers because more than one string can
    be right ("it is blue" and "blue"). Exact match is the most transparent metric
    there is, and the most brittle: one extra word and a correct answer scores zero.

    Returns:
        1.0 or 0.0.
    """
    # TODO: Return 1.0 when the normalized prediction matches any normalized
    #       acceptable answer, otherwise 0.0.
    # HINT: normalize both sides; `any(...)` over a generator, wrapped in float().
    raise NotImplementedError("TODO: score the prediction against the acceptable answers")


# ---------------------------------------------------------------------------
# Step 4: Token-level F1
# ---------------------------------------------------------------------------


def token_f1(prediction: str, reference: str) -> float:
    """Token-level F1 between a prediction and one reference answer.

    F1 gives partial credit where exact match gives none: "it is bluu" shares two of
    its three tokens with "it is blue", which is worth more than zero. Precision is
    the fraction of predicted tokens that are correct, recall the fraction of
    reference tokens that were produced, and F1 is their harmonic mean &mdash; the
    information-retrieval metric that entered question answering through SQuAD.

    Everything up to the harmonic mean is provided.

    Returns:
        A score in [0.0, 1.0].
    """
    predicted_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not predicted_tokens or not reference_tokens:
        return float(predicted_tokens == reference_tokens)

    # Counter intersection counts each shared token no more times than it appears
    # in both sides, so repeating a word cannot inflate the overlap.
    shared = sum((Counter(predicted_tokens) & Counter(reference_tokens)).values())
    if shared == 0:
        return 0.0
    precision = shared / len(predicted_tokens)
    recall = shared / len(reference_tokens)
    # TODO: Return the F1 score: the harmonic mean of precision and recall.
    # HINT: 2 * precision * recall, divided by their sum.
    raise NotImplementedError("TODO: combine precision and recall into F1")


# ---------------------------------------------------------------------------
# Step 5: Per-task accuracy
# ---------------------------------------------------------------------------


def task_accuracy(scores_by_task: dict[str, list[float]]) -> dict[str, float]:
    """Average each task's per-case scores into one number per task.

    Keeping the breakdown is the habit this whole exercise is arguing for. An
    aggregate can hide a total regression on one task inside a small overall gain,
    and only the per-task view shows it.

    Args:
        scores_by_task: Task name -> the list of per-case scores for that task.

    Returns:
        Task name -> mean score.
    """
    # TODO: Return a dict mapping each task to the mean of its scores.
    # HINT: a dict comprehension over .items(); sum(scores) / len(scores).
    raise NotImplementedError("TODO: average the per-case scores within each task")


# ---------------------------------------------------------------------------
# Step 6: The overall suite score
# ---------------------------------------------------------------------------


def suite_score(per_task: dict[str, float]) -> float:
    """Average the per-task scores into the single headline number.

    Note what this choice does: averaging the four *task* scores weights every task
    equally, while averaging all 50 *cases* would let the task with the most cases
    dominate. Neither is more correct, but they give different numbers on the same
    model, which is the whole reason a benchmark has to publish its protocol.

    Returns:
        The mean of the per-task scores.
    """
    # TODO: Return the mean of the per-task scores.
    # HINT: per_task.values() gives the scores; sum(...) / len(...).
    raise NotImplementedError("TODO: average the per-task scores into the suite score")


# ---------------------------------------------------------------------------
# Step 7: Multiple choice, scored by likelihood
# ---------------------------------------------------------------------------


def score_multiple_choice(option_log_probs: list[float], option_lengths: list[int]) -> int:
    """Pick the option the model finds most likely, per token.

    This is how MMLU, HellaSwag, and ARC are actually run: nothing is generated. The
    runner scores each candidate answer under the model and hands you its total
    log-probability and its length in tokens. Dividing by the length is what makes
    the comparison fair &mdash; every extra token adds another negative number, so
    total log-probability systematically prefers the shortest option.

    Args:
        option_log_probs: Total log-probability of each option, one per option.
        option_lengths: Number of scored tokens in each option, same order.

    Returns:
        The index of the chosen option.
    """
    # TODO: Return the index of the option with the highest AVERAGE log-probability
    #       per token (total log-probability divided by number of tokens).
    # HINT: range(len(option_log_probs)) gives the indices; max(..., key=...) picks
    #       the best one, and the key is a lambda dividing one list by the other.
    raise NotImplementedError("TODO: choose the option with the best per-token log-probability")


# ---------------------------------------------------------------------------
# Step 8: pass@k
# ---------------------------------------------------------------------------


def pass_at_k(n: int, c: int, k: int) -> float:
    """Probability that at least one of k draws from n samples is correct.

    From the HumanEval paper. Sampling n completions and observing c correct ones,
    the unbiased estimate of pass@k is one minus the probability that a random
    k-subset misses every correct sample:

        pass@k = 1 - C(n - c, k) / C(n, k)

    pass@1 is the ordinary accuracy of a single sample. Large-k pass@k asks a
    different question: is the right answer anywhere in the model's distribution?
    Module 7's headline claim &mdash; that RL sharpens the distribution rather than
    expanding it &mdash; is exactly a claim about the gap between these two.

    Args:
        n: Total samples drawn for this case.
        c: How many of them were correct.
        k: Budget of attempts to score.

    Returns:
        A probability in [0.0, 1.0].
    """
    if n - c < k:
        return 1.0  # too few wrong samples to fill a k-subset: some draw must hit
    # TODO: Return the pass@k estimate from the formula above.
    # HINT: math.comb(a, b) is the binomial coefficient C(a, b).
    raise NotImplementedError("TODO: compute the pass@k estimate")
