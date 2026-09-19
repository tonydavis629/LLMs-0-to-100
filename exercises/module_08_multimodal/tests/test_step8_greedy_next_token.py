"""Step 8: greedy_next_token()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import torch

from tests.check import Check, bad, ok


def check_greedy_next_token(greedy_next_token) -> list[Check]:
    """The argmax over the vocabulary, taken at the final position only."""
    checks = []

    # One sequence, two positions, vocabulary of 3. The first position favors
    # token 1, the last position favors token 2: only the last one counts.
    logits = torch.tensor([[[0.0, 5.0, 1.0],
                            [2.0, 0.0, 9.0]]])
    got = greedy_next_token(logits)
    name = "uses the last position: logits [0,5,1] then [2,0,9] give token 2"
    if got.numel() == 1 and int(got.flatten()[0]) == 2:
        checks.append(ok(name))
    elif got.numel() == 1 and int(got.flatten()[0]) == 1:
        checks.append(bad(name, "expected 2, got 1 (token 1 wins at the first position; use the final one)"))
    else:
        checks.append(bad(name, f"expected tensor([2]), got {got.tolist()}"))

    # A batch of 2 sequences: one token id per sequence
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 5)
    got = greedy_next_token(logits)
    expected = logits[:, -1, :].argmax(dim=-1)
    name = "returns one token id per sequence: a (2, 3, 5) batch gives shape (2,)"
    checks.append(
        ok(name)
        if tuple(got.shape) == (2,) and torch.equal(got, expected)
        else bad(name, f"expected {expected.tolist()} with shape (2,), got {got.tolist()} with shape "
                       f"{tuple(got.shape)} (take the argmax over the vocabulary, dim=-1)")
    )

    # Token ids index the vocabulary, so they must be integers, not probabilities
    checks.append(
        ok("returns integer token ids (torch.long), not logit values")
        if got.dtype == torch.long
        else bad("returns integer token ids (torch.long), not logit values",
                 f"got dtype {got.dtype} (use argmax, which returns positions, not max, which returns values)")
    )
    return checks


def check_grounded_captions(results: dict) -> list[Check]:
    """With the trained bridge, the caption should depend on the image."""
    checks = []
    correct, total = results["caption_correct"], results["caption_total"]
    name = "held-out caption exact-match is at least 50% (the scenes were never trained on)"
    checks.append(
        ok(name)
        if correct / total >= 0.5
        else bad(name, f"got {correct}/{total} ({correct / total:.1%})")
    )

    # The grounding test: one prompt, three different scenes, three different answers
    captions = results["demo_captions"]
    name = "the same prompt returns a different caption for each of the 3 demo images"
    checks.append(
        ok(name)
        if len(set(captions)) == len(captions)
        else bad(name, f"got {captions} (the model is not reading the image)")
    )
    return checks
