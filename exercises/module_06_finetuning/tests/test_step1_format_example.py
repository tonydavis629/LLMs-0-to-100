"""Step 1: format_example()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

from tests.check import Check, bad, expect_equal, ok

# Toy vocabulary for the hand examples: a=1, b=2, c=3, ... and three marker ids
TOY_SPECIAL = {"<|user|>": 90, "<|assistant|>": 91, "<|end|>": 92, "<|pad|>": 93}


def toy_encode(text: str) -> list[int]:
    """Map each lowercase letter to its position in the alphabet (a=1, b=2, ...)."""
    return [ord(c) - ord("a") + 1 for c in text]


def check_format_example(format_example, special: dict[str, int], encode_fn, decode_fn) -> list[Check]:
    """The chat template: user turn, end, assistant turn, response, end."""
    checks = []

    # Hand example with toy ids: a=1, b=2, c=3, and user=90, assistant=91, end=92
    ids = format_example("ab", "c", TOY_SPECIAL, toy_encode)
    checks.append(
        expect_equal("prompt 'ab', response 'c' gives [user, a, b, end, assistant, c, end]",
                     ids, [90, 1, 2, 92, 91, 3, 92])
    )

    # One flat list of ints, not a list of lists
    ids = format_example("hi", "ok", TOY_SPECIAL, toy_encode)
    flat = isinstance(ids, list) and all(isinstance(i, int) for i in ids)
    checks.append(
        ok("returns one flat list of ints ('hi' -> 'ok' gives 2 + 2 + 4 markers = 8 ids)")
        if flat and len(ids) == 8
        else bad("returns one flat list of ints ('hi' -> 'ok' gives 2 + 2 + 4 markers = 8 ids)",
                 f"got {ids!r} (join the pieces with +, and wrap each marker id in [ ])")
    )

    # The real 69-token vocabulary: decoding the ids gives back the template text
    ids = format_example("hi", "HO", special, encode_fn)
    name = "real vocabulary: 'hi' -> 'HO' decodes to '<|user|>hi<|end|><|assistant|>HO<|end|>'"
    if isinstance(ids, list) and all(isinstance(i, int) for i in ids):
        checks.append(expect_equal(name, decode_fn(ids), "<|user|>hi<|end|><|assistant|>HO<|end|>"))
    else:
        checks.append(bad(name, f"got {ids!r}, which is not a flat list of token ids"))
    return checks
