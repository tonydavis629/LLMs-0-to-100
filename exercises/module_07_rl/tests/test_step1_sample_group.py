"""Step 1: sample_group()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations

import torch

from tests.check import Check, bad, ok


class _FakeSampler:
    """Stands in for model.generate so the test needs no real model.

    It records how it was called and returns a (B, L) tensor: each row of the
    prompt batch followed by `max_new_tokens` copies of a number unique to that
    row and call, so every row of every call is a different sample.
    """

    def __init__(self):
        self.calls = []

    def __call__(self, policy, prompt_ids, max_new_tokens, block_size,
                 temperature=1.0, greedy=False, generator=None):
        self.calls.append({"temperature": temperature, "greedy": greedy, "generator": generator,
                           "max_new_tokens": max_new_tokens, "block_size": block_size,
                           "rows": prompt_ids.shape[0]})
        rows = prompt_ids.shape[0]
        first = 10 * len(self.calls)  # call 1 uses 10, 11, ...; call 2 uses 20, 21, ...
        new_tokens = torch.arange(first, first + rows).unsqueeze(1).repeat(1, max_new_tokens)
        return torch.cat([prompt_ids, new_tokens], dim=1)


def check_sample_group(sample_group) -> list[Check]:
    """One batched generate_fn call with one prompt row per group member."""
    checks = []
    fake = _FakeSampler()
    gen = torch.Generator().manual_seed(0)
    prompt_ids = torch.tensor([[7, 8]])
    group = sample_group(None, prompt_ids, 3, 2, 16, 0.7, fake, gen)

    # Three members from one call whose batch holds three copies of the prompt
    n_samples = len(group) if isinstance(group, (list, tuple, torch.Tensor)) else None
    distinct = n_samples == 3 and len({tuple(torch.as_tensor(s).flatten().tolist()) for s in group}) == 3
    name = "group_size=3 gives a list of 3 different samples from one generate_fn call on a 3-row batch"
    if isinstance(group, list) and distinct and [c["rows"] for c in fake.calls] == [3]:
        checks.append(ok(name))
    else:
        kind = type(group).__name__
        batches = [c["rows"] for c in fake.calls]
        if isinstance(group, torch.Tensor):
            tip = "return a list: list() splits the 2-D result into its rows"
        else:
            tip = ("repeat the prompt group_size times and call generate_fn once; "
                   "copies of one sample all get advantage 0")
        checks.append(bad(name, f"got a {kind} of {n_samples} samples from {len(fake.calls)} call(s) "
                                f"with batch rows {batches} ({tip})"))

    # Each member is one row of the (3, L) result: prompt [7, 8] plus 2 new tokens
    name = "each sample is one row of the (3, L) result: prompt [7, 8] + 2 tokens gives shape (4,)"
    shapes = [tuple(s.shape) if isinstance(s, torch.Tensor) else type(s).__name__ for s in group]
    if shapes and all(shape == (4,) for shape in shapes):
        checks.append(ok(name))
    else:
        checks.append(bad(name, f"expected shape (4,) for every sample, got {shapes} "
                                "(list() of a 2-D tensor gives its rows)"))

    # temperature and generator reach the call; greedy stays off
    name = "passes temperature=0.7 and the generator to generate_fn (sampling, not greedy)"
    expected = {"temperature": 0.7, "greedy": False, "generator": gen, "max_new_tokens": 2, "block_size": 16}
    # Every argument that arrived with the wrong value, on any call
    wrong = {k: v for c in fake.calls for k, v in c.items()
             if k in expected and v is not expected[k] and v != expected[k]}
    if fake.calls and not wrong:
        checks.append(ok(name))
    else:
        shown = ", ".join(f"{k}={'<the generator>' if v is gen else repr(v)}" for k, v in wrong.items())
        checks.append(bad(name, f"generate_fn received {shown or 'no calls'} "
                                "(pass temperature=temperature and generator=generator by keyword)"))
    return checks
