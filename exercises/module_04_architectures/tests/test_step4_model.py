"""Step 4: GPT2Model.forward()

Run by src/main.py. You do NOT need to edit this file. The first three
checks use a tiny model whose embedding and blocks are simple stand-ins
(see fakes.py), so they only test the wiring you write in this step. The
last check runs your full GPT-2 against Hugging Face's.
"""

from __future__ import annotations

import torch

from tests.check import Check, bad, ok
from tests.fakes import first_feature, tiny_gpt2


def check_gpt2_forward(GPT2Model) -> list[Check]:
    """embed -> every block in order -> ln_f -> lm_head, on a tiny model."""
    checks = []
    model, log = tiny_gpt2(GPT2Model, n_layers=3, vocab_size=7)
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    with torch.no_grad():
        out = model(ids)

    checks.append(
        ok("returns logits over the vocabulary: ids (2, 3) give shape (2, 3, vocab_size=7)")
        if tuple(out.shape) == (2, 3, 7)
        else bad("returns logits over the vocabulary: ids (2, 3) give shape (2, 3, vocab_size=7)",
                 f"expected (2, 3, 7), got {tuple(out.shape)} (did you finish with self.lm_head?)")
    )
    checks.append(
        ok("runs all 3 blocks once each, in order: block calls were [0, 1, 2]")
        if log == [0, 1, 2]
        else bad("runs all 3 blocks once each, in order: block calls were [0, 1, 2]",
                 f"got block calls {log} (loop over self.blocks)")
    )
    if tuple(out.shape) != (2, 3, 7):
        return checks

    # The stand-in blocks add 1, 2 and 3 to the first feature: 6 in total
    with torch.no_grad():
        x = model.embed(ids)
        want = model.lm_head(model.ln_f(x + first_feature(6, 4)))
        no_norm = model.lm_head(x + first_feature(6, 4))
        last_only = model.lm_head(model.ln_f(x + first_feature(3, 4)))
    name = "feeds each block's output to the next, then applies ln_f and lm_head"
    if torch.allclose(out, want, atol=1e-5):
        checks.append(ok(name))
    elif torch.allclose(out, no_norm, atol=1e-5):
        checks.append(bad(name, "the final layer norm is missing (apply self.ln_f before self.lm_head)"))
    elif torch.allclose(out, last_only, atol=1e-5):
        checks.append(bad(name, "only the last block's change survived (write x = block(x) inside the loop)"))
    else:
        diff = float((out - want).abs().max())
        checks.append(bad(name, f"largest difference from the reference is {diff:.4f}"))
    return checks


def check_matches_hugging_face(results: dict) -> list[Check]:
    """Your GPT-2 with the real weights against Hugging Face's GPT2LMHeadModel."""
    diff = results["hf_logit_diff"]
    name = "real GPT-2: your logits match Hugging Face's GPT2LMHeadModel to within 0.001"
    if diff < 1e-3:
        return [ok(name)]
    return [bad(name, f"largest difference is {diff:.4f} over the prompt's 5 x 50257 logits\n"
                      "(every layer feeds this; if an earlier step is INCORRECT, fix that first)")]
