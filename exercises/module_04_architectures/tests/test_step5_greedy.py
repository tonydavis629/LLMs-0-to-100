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


class _LogitsOnly(torch.nn.Module):
    """Wrap Hugging Face's GPT-2 so that calling it returns plain logits, like yours."""

    def __init__(self, hf_model: torch.nn.Module) -> None:
        super().__init__()
        self.hf_model = hf_model

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.hf_model(token_ids).logits


def check_greedy_on_gpt2(greedy_decode, hf_model, tokenizer, prompt: str) -> list[Check]:
    """Your loop driving Hugging Face's GPT-2, against Hugging Face's own generate()."""
    yours = greedy_decode(_LogitsOnly(hf_model), tokenizer, prompt, max_new=10)
    ids = tokenizer.encode(prompt, return_tensors="pt")
    out = hf_model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=10,
                            do_sample=False, pad_token_id=tokenizer.eos_token_id)
    theirs = tokenizer.decode(out[0])
    name = "real GPT-2: your loop matches Hugging Face's generate(do_sample=False)"
    if yours == theirs:
        return [ok(name)]
    return [bad(name, f"expected {theirs!r}\ngot      {yours!r}")]
