"""Step 5: greedy_decode()

Run by src/main.py. You do NOT need to edit this file. The first two
checks swap GPT-2 for tiny fake models whose best next token is known
(see fakes.py). The last one runs your decoding loop on Hugging Face's
GPT-2, so it does not depend on the model you built in Steps 1-4.
"""

from __future__ import annotations

import torch

from tests.check import Check, bad, ok
from tests.fakes import CountingModel, FixedModel, NumberTokenizer


def check_greedy_decode(greedy_decode) -> list[Check]:
    """Always take the highest-scoring token, then feed it back in."""
    checks = []
    tokenizer = NumberTokenizer()

    # The counting model always prefers "last token + 1"
    got = greedy_decode(CountingModel(), tokenizer, "1", max_new=4)
    checks.append(
        ok('follows the argmax: a counting model turns "1" into "1 2 3 4 5"')
        if got == "1 2 3 4 5"
        else bad('follows the argmax: a counting model turns "1" into "1 2 3 4 5"',
                 f'expected "1 2 3 4 5"\ngot      "{got}"')
    )

    # Close logits: sampling would sometimes pick another token; argmax never does
    model = FixedModel([1.0, 1.2, 0.9, 1.1])
    outputs = set()
    for seed in range(3):
        torch.manual_seed(seed)
        outputs.add(greedy_decode(model, tokenizer, "0", max_new=5))
    checks.append(
        ok("picks the largest of close logits [1.0, 1.2, 0.9, 1.1]: token 1, every time")
        if outputs == {"0 1 1 1 1 1"}
        else bad("picks the largest of close logits [1.0, 1.2, 0.9, 1.1]: token 1, every time",
                 f'expected "0 1 1 1 1 1" on every run\ngot      {sorted(outputs)}\n'
                 "(use torch.argmax, not sampling or argmin)")
    )
    return checks


def check_greedy_on_gpt2(results: dict) -> list[Check]:
    """Your loop driving Hugging Face's GPT-2, against Hugging Face's own generate()."""
    yours, theirs = results["greedy_on_hf"], results["hf_greedy"]
    name = "real GPT-2: your loop matches Hugging Face's generate(do_sample=False)"
    if yours == theirs:
        return [ok(name)]
    return [bad(name, f"expected {theirs!r}\ngot      {yours!r}")]
