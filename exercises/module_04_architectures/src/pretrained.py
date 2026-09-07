"""Loading the real GPT-2 checkpoint, provided for you.

You do NOT need to edit this file. It downloads the official GPT-2 small
weights from HuggingFace and copies each tensor into the matching layer of
the model you assembled in `exercise.py`.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def load_gpt2_weights(model: nn.Module) -> None:
    """Load pretrained GPT-2 small weights into the custom model.

    This function downloads the official GPT-2 weights from HuggingFace,
    then copies each tensor into the matching layer of the custom model.

    Args:
        model: Your GPT2Model instance from exercise.py (already initialized).
    """
    from transformers import GPT2LMHeadModel

    print("Downloading GPT-2 weights from HuggingFace ...")
    pretrained = GPT2LMHeadModel.from_pretrained("gpt2")
    pretrained_sd = pretrained.state_dict()

    # Build a flat name mapping from custom -> pretrained.
    custom_names = [n for n, _ in model.named_parameters()]
    pretrained_names = list(pretrained_sd.keys())

    # Map custom names to pretrained names.
    # Custom names look like: embed.token_embed.weight, blocks.0.attn.c_attn.weight, etc.
    # Pretrained names look like: transformer.wte.weight, transformer.h.0.attn.c_attn.weight, etc.
    mapping = {}
    for cn in custom_names:
        if cn == "embed.token_embed.weight":
            pn = "transformer.wte.weight"
        elif cn == "embed.pos_embed.weight":
            pn = "transformer.wpe.weight"
        elif cn.startswith("blocks."):
            # blocks.0.attn.c_attn.weight  ->  transformer.h.0.attn.c_attn.weight
            rest = cn[len("blocks."):]
            pn = "transformer.h." + rest
        elif cn == "ln_f.weight":
            pn = "transformer.ln_f.weight"
        elif cn == "ln_f.bias":
            pn = "transformer.ln_f.bias"
        elif cn == "lm_head.weight":
            pn = "lm_head.weight"
        else:
            raise ValueError(f"Unmapped custom parameter: {cn}")
        mapping[cn] = pn

    # Copy weights.
    custom_sd = model.state_dict()
    new_sd = {}
    for cn, pn in mapping.items():
        pt = pretrained_sd[pn]
        ct = custom_sd[cn]
        if pt.shape != ct.shape:
            raise ValueError(f"Shape mismatch for {cn}: custom {ct.shape} vs pretrained {pt.shape}")
        new_sd[cn] = pt

    model.load_state_dict(new_sd)
    print(f"Loaded {len(new_sd)} parameter tensors from pretrained GPT-2.")
