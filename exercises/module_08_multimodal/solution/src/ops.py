"""Provided vision-language plumbing (NOT part of the exercise).

These are the small, mechanical tensor operations that sit between the eight steps
you implement in `exercise.py`. They are given to you so the exercise stays focused
on the conceptually important pieces: patchifying and pooling an image, the CLIP
contrastive objective, and bridging the image into the language model.

You do not need to edit this file.
"""

from __future__ import annotations

import torch


def flatten_patches(patches: torch.Tensor) -> torch.Tensor:
    """Flatten each patch (C, P, P) into one vector, so a patch becomes one token.

    Args:
        patches: shape (B, N, C, P, P).

    Returns:
        Flattened patches of shape (B, N, C*P*P).
    """
    return patches.flatten(2)


def project_patches(patches_flat: torch.Tensor, patch_proj: torch.nn.Linear) -> torch.Tensor:
    """Linearly project each flattened patch into the vision embedding width.

    The visual analog of a token embedding: it maps a patch's raw pixel vector into
    the model's representation space.
    """
    return patch_proj(patches_flat)


def add_position_embeddings(patch_embeds: torch.Tensor, pos_embed: torch.Tensor) -> torch.Tensor:
    """Add learned position embeddings so the model can recover 2-D layout.

    Flattening threw away where each patch sat; a learned embedding per position puts
    it back, letting the model tell "top" from "bottom". Broadcasting handles the batch.
    """
    return patch_embeds + pos_embed


def encode_text(token_ids: torch.Tensor, text_encoder: torch.nn.Module) -> torch.Tensor:
    """Encode caption token ids into pooled text embeddings with the provided tower.

    The text encoder returns one embedding per caption at the SAME width as the image
    embedding, so the two live in a shared space and can be compared directly.
    """
    return text_encoder(token_ids)


def retrieval_accuracy(logits: torch.Tensor) -> float:
    """Image-to-text retrieval accuracy: how often the top caption is the right one.

    The matched caption for image i is the diagonal entry i, so this is the fraction of
    rows whose argmax column equals the row index.
    """
    labels = torch.arange(logits.shape[0], device=logits.device)
    return (logits.argmax(dim=1) == labels).float().mean().item()


def concat_visual_prefix(prefix_embeds: torch.Tensor, token_embeds: torch.Tensor) -> torch.Tensor:
    """Concatenate visual prefix vectors in front of the text-token embeddings.

    Attention does not care whether a vector came from a pixel patch or a word; putting
    the visual prefix first makes the image a prefix the text can attend back to.

    Args:
        prefix_embeds: visual prefix, shape (B, K, d_llm).
        token_embeds: text-token embeddings, shape (B, L, d_llm).

    Returns:
        Combined sequence embeddings, shape (B, K + L, d_llm).
    """
    return torch.cat([prefix_embeds, token_embeds], dim=1)
