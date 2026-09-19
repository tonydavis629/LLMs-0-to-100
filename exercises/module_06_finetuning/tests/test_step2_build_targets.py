"""Step 2: build_targets()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

from tests.check import Check, bad, expect_equal, ok


def check_build_targets(build_targets) -> list[Check]:
    """Next-token targets, with every prediction made inside the prompt set to -100."""
    checks = []

    # Hand example: 7 ids, the first 4 are the prompt (user, text, end, assistant)
    ids = [10, 11, 12, 13, 14, 15, 16]
    targets = build_targets(ids, 4)
    checks.append(
        expect_equal("ids [10..16] with prompt_span=4 gives [-100, -100, -100, 14, 15, 16, -100]",
                     targets, [-100, -100, -100, 14, 15, 16, -100])
    )

    # One target per input position, or the batch tensors will not line up
    ids = list(range(20, 30))
    targets = build_targets(ids, 3)
    checks.append(
        ok("returns one target per input position (10 ids give 10 targets)")
        if len(targets) == len(ids)
        else bad("returns one target per input position (10 ids give 10 targets)",
                 f"got {len(targets)} targets for {len(ids)} ids")
    )

    # The assistant marker (last prompt token) is where the answer starts
    first = targets[2] if len(targets) > 2 else None
    checks.append(
        ok("prompt_span=3: the assistant marker's position predicts ids[3], the first response token")
        if first == ids[3]
        else bad("prompt_span=3: the assistant marker's position predicts ids[3], the first response token",
                 f"expected {ids[3]}, got {first!r}"
                 + (" (mask prompt_span - 1 positions, not prompt_span)" if first == -100 else "")
                 + (" (each target is the NEXT token, ids[t + 1])" if first == ids[2] else ""))
    )

    # Each unmasked target is the NEXT token, never the token at the same position
    shifted = all(t == -100 or (i + 1 < len(ids) and t == ids[i + 1]) for i, t in enumerate(targets))
    checks.append(
        ok("every unmasked target is the next token: targets[t] == ids[t + 1]")
        if shifted
        else bad("every unmasked target is the next token: targets[t] == ids[t + 1]",
                 f"ids     {ids}\ngot     {targets}")
    )
    return checks
