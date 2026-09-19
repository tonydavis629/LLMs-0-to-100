"""Step 9: build_generation_prompt()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

from tests.check import Check, bad, expect_equal, ok
from tests.test_step1_format_example import TOY_SPECIAL, toy_encode


def check_build_generation_prompt(build_generation_prompt, special: dict[str, int], encode_fn,
                                  decode_fn) -> list[Check]:
    """The user turn, then the assistant marker, and nothing more: the model writes the rest."""
    checks = []

    # Hand example with toy ids: a=1, b=2, and user=90, assistant=91, end=92
    ids = build_generation_prompt("ab", TOY_SPECIAL, toy_encode)
    checks.append(
        expect_equal("prompt 'ab' gives [user, a, b, end, assistant], with no response yet",
                     ids, [90, 1, 2, 92, 91])
    )

    # The last token must be the assistant marker: the next token generated is the answer
    ids = build_generation_prompt("hi", TOY_SPECIAL, toy_encode)
    last = ids[-1] if isinstance(ids, list) and ids else None
    extra_end = isinstance(ids, list) and ids[-2:] == [TOY_SPECIAL["<|assistant|>"], TOY_SPECIAL["<|end|>"]]
    checks.append(
        ok("ends on the assistant marker, so the next generated token starts the answer")
        if last == TOY_SPECIAL["<|assistant|>"]
        else bad("ends on the assistant marker, so the next generated token starts the answer",
                 f"the last id is {last!r}, expected {TOY_SPECIAL['<|assistant|>']} (<|assistant|>)"
                 + (" (do not add a closing <|end|>: the answer has not been written yet)"
                    if extra_end else "")
                 + (" (the <|assistant|> marker is missing)"
                    if isinstance(ids, list) and TOY_SPECIAL["<|assistant|>"] not in ids else ""))
    )

    # The real 69-token vocabulary: the exact prefix the runner feeds the model
    text = decode_fn(build_generation_prompt("uppercase: hello", special, encode_fn))
    checks.append(
        expect_equal("real vocabulary: 'uppercase: hello' decodes to '<|user|>uppercase: hello<|end|><|assistant|>'",
                     text, "<|user|>uppercase: hello<|end|><|assistant|>")
    )
    return checks
