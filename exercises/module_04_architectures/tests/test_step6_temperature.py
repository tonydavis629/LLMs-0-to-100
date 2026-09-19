"""Step 6: sample_with_temperature_topk()

Run by src/main.py. You do NOT need to edit this file. The checks swap
GPT-2 for tiny fake models with fixed logits (see fakes.py).
"""

from __future__ import annotations

import torch

from tests.check import Check, bad, ok
from tests.fakes import FixedModel, NumberTokenizer


def _logits_after_scaling(sample_fn, logits: list[float], temperature: float) -> list[float]:
    """Run one decoding step and return the logits handed to the top-k helper.

    The provided helper `_sample_topk_token` is swapped (for this one call)
    for a stand-in that records what it receives and returns token 0.
    """
    seen = []

    def record(next_logits, top_k):
        seen.append(next_logits.detach().flatten().tolist())
        return torch.zeros(1, 1, dtype=torch.long)

    helpers = sample_fn.__globals__  # the namespace exercise.py runs in
    original = helpers["_sample_topk_token"]
    helpers["_sample_topk_token"] = record
    try:
        sample_fn(FixedModel(logits), NumberTokenizer(), "0", max_new=1,
                  temperature=temperature, top_k=len(logits))
    finally:
        helpers["_sample_topk_token"] = original
    return seen[0]


def _share_of_favorite(sample_fn, temperature: float, draws: int = 1000) -> float:
    """How often token 1 is drawn when the logits are [0, 1] (73% at T=1)."""
    torch.manual_seed(0)
    text = sample_fn(FixedModel([0.0, 1.0]), NumberTokenizer(), "0", max_new=draws,
                     temperature=temperature, top_k=2)
    tokens = text.split()[1:]  # drop the prompt
    return tokens.count("1") / len(tokens)


def check_temperature(sample_with_temperature_topk) -> list[Check]:
    """Divide by T before sampling: T < 1 sharpens the distribution, T > 1 flattens it."""
    checks = []

    got = _logits_after_scaling(sample_with_temperature_topk, [2.0, 4.0, 6.0], temperature=2.0)
    name = "divides the logits by the temperature: [2, 4, 6] at T=2 become [1, 2, 3]"
    if torch.allclose(torch.tensor(got), torch.tensor([1.0, 2.0, 3.0])):
        checks.append(ok(name))
    else:
        nudges = {
            (4.0, 8.0, 12.0): " (multiplying by T flips its effect; divide instead)",
            (2.0, 4.0, 6.0): " (the logits were not scaled at all)",
        }
        nudge = nudges.get(tuple(round(v, 4) for v in got), "")
        checks.append(bad(name, f"expected [1.0, 2.0, 3.0]\ngot      {got}{nudge}"))

    # softmax([0, 1] / T): T=0.5 gives the favorite 88%, T=2 gives it 62%.
    # Multiplying by T instead would swap those two numbers.
    for temperature, expected, swapped, word in ((0.5, 0.881, 0.622, "sharpens"),
                                                 (2.0, 0.622, 0.881, "flattens")):
        share = _share_of_favorite(sample_with_temperature_topk, temperature)
        name = (f"T={temperature:g} {word}: logits [0, 1] pick the favorite "
                f"~{expected:.0%} of the time (73% at T=1)")
        if abs(share - expected) < 0.05:
            checks.append(ok(name))
        else:
            nudge = " (did you multiply by the temperature?)" if abs(share - swapped) < 0.05 else ""
            checks.append(bad(name, f"expected about {expected:.1%} over 1000 draws, got {share:.1%}{nudge}"))
    return checks
