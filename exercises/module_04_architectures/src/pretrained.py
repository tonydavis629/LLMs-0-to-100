"""Loading the real GPT-2 checkpoint and tokenizer, provided for you.

You do NOT need to edit this file. It fetches the official GPT-2 small
weights from HuggingFace (downloaded on the first run, then read from the
local cache) and copies each tensor into the matching layer of the model
you assembled in `exercise.py`.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _quiet_huggingface() -> None:
    """Hide HuggingFace's progress bars and notices so the runner output stays readable."""
    from huggingface_hub.utils import logging as hub_logging
    from transformers.utils import logging as hf_logging

    hub_logging.set_verbosity_error()
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()


def load_tokenizer():
    """Return the GPT-2 byte-pair-encoding tokenizer from HuggingFace."""
    from transformers import GPT2Tokenizer

    _quiet_huggingface()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_gpt2_weights(model: nn.Module) -> nn.Module:
    """Load pretrained GPT-2 small weights into the custom model.

    This function fetches the official GPT-2 weights from HuggingFace,
    then copies each tensor into the matching layer of the custom model.

    Args:
        model: Your GPT2Model instance from exercise.py (already initialized).

    Returns:
        HuggingFace's own GPT2LMHeadModel with the same weights, which the
        runner uses as a reference to check your model against.
    """
    from transformers import GPT2LMHeadModel

    print("Loading GPT-2 small weights from HuggingFace (downloaded on the first run) ...")
    _quiet_huggingface()
    pretrained = GPT2LMHeadModel.from_pretrained("gpt2")
    pretrained_sd = pretrained.state_dict()

    # Build a flat name mapping from our parameter names to HuggingFace's.
    #
    # Two things differ between the two models:
    #
    # 1) Names. We call the layer norms `ln1`/`ln2` and the feed-forward layers
    #    `ffn.fc1`/`ffn.fc2`; HuggingFace calls them `ln_1`/`ln_2` and
    #    `mlp.c_fc`/`mlp.c_proj`.
    # 2) Weight layout. HuggingFace's GPT-2 uses its own `Conv1D` layer, which
    #    stores weights as (in_features, out_features). `nn.Linear` stores them
    #    the other way round, as (out_features, in_features), so those four
    #    tensors per block have to be transposed on the way in. Biases match.
    custom_names = [n for n, _ in model.named_parameters()]

    # Our block-level suffix -> HuggingFace's, and whether the tensor is a
    # transposed Conv1D weight.
    BLOCK_PARTS = {
        "ln1": ("ln_1", False),
        "ln2": ("ln_2", False),
        "attn.c_attn": ("attn.c_attn", True),
        "attn.c_proj": ("attn.c_proj", True),
        "ffn.fc1": ("mlp.c_fc", True),
        "ffn.fc2": ("mlp.c_proj", True),
    }

    mapping = {}          # custom name -> pretrained name
    needs_transpose = set()  # custom names whose weight must be transposed

    for cn in custom_names:
        if cn == "embed.token_embed.weight":
            pn = "transformer.wte.weight"
        elif cn == "embed.pos_embed.weight":
            pn = "transformer.wpe.weight"
        elif cn.startswith("blocks."):
            # blocks.0.attn.c_attn.weight -> layer "0", part "attn.c_attn", kind "weight"
            rest = cn[len("blocks."):]
            layer, remainder = rest.split(".", 1)
            part, kind = remainder.rsplit(".", 1)
            if part not in BLOCK_PARTS:
                raise ValueError(f"Unmapped block parameter: {cn}")
            hf_part, is_conv1d = BLOCK_PARTS[part]
            pn = f"transformer.h.{layer}.{hf_part}.{kind}"
            if is_conv1d and kind == "weight":
                needs_transpose.add(cn)
        elif cn in ("ln_f.weight", "ln_f.bias"):
            pn = "transformer." + cn
        elif cn == "lm_head.weight":
            # GPT-2 ties the output head to the token embedding table.
            pn = "lm_head.weight"
        else:
            raise ValueError(f"Unmapped custom parameter: {cn}")
        mapping[cn] = pn

    # Copy weights, transposing the Conv1D ones.
    custom_sd = model.state_dict()
    new_sd = {}
    for cn, pn in mapping.items():
        pt = pretrained_sd[pn]
        if cn in needs_transpose:
            pt = pt.t()
        ct = custom_sd[cn]
        if pt.shape != ct.shape:
            raise ValueError(f"Shape mismatch for {cn}: custom {ct.shape} vs pretrained {pt.shape}")
        new_sd[cn] = pt

    # The causal mask is a registered buffer, not a parameter, so it is absent
    # from new_sd; keep the one the model built for itself.
    model.load_state_dict(new_sd, strict=False)
    print(f"Loaded {len(new_sd)} parameter tensors from pretrained GPT-2.")
    return pretrained.eval()
