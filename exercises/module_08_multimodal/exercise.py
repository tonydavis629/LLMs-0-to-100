"""
Module 8 Exercise: Align image embeddings with NanoGPT

You implement the eight one-line steps below that build a tiny vision-language model.
Everything else is provided: the synthetic shapes dataset, the vision tower and text
encoder parameters, the NanoGPT language model, the projector, the small mechanical
tensor ops in `src/ops.py`, and the runner.

The goal is not a useful visual assistant; it is to make the multimodal *bridge*
visible in three stages:

  PART 1  Turn a 32x32 image into ONE embedding: split it into patches (step 1) and,
          after the provided tower mixes them, pool to one vector (step 2).
  PART 2  Align images and captions with a CLIP-style contrastive loss (steps 3-5),
          and watch retrieval accuracy climb from chance toward 1.0.
  PART 3  Project the image embedding into NanoGPT's hidden width as visual prefix
          tokens and train the model to caption and answer (steps 6-8). The payoff:
          the answer changes when the image changes.

Run after each step; unfinished steps are skipped automatically:
    uv run python module_08_multimodal/src/main.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ===========================================================================
# PART 1: Turn an image into one embedding (the vision tower)
# ===========================================================================


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Split each image into a grid of non-overlapping square patches.

    An image is (C, H, W); a transformer wants a *sequence*. We cut the image into
    (H/P) x (W/P) patches and lay them out in row-major order, so the 2-D picture
    becomes a 1-D list of patches — the visual analog of splitting text into tokens.
    (The provided `src/ops.py` then flattens, projects, and adds positions for you.)

    Args:
        images: batch of images, shape (B, C, H, W).
        patch_size: side length P of each square patch.

    Returns:
        Patches of shape (B, N, C, P, P) where N = (H/P) * (W/P).
    """
    B, C, H, W = images.shape
    # TODO: Reshape (B, C, H, W) into (B, N, C, P, P) patches with N = (H/P)*(W/P).
    # HINT: reshape to (B, C, H/P, P, W/P, P), permute the two grid axes next to the
    #       batch, then reshape to (B, N, C, P, P).
    raise NotImplementedError("TODO: split each image into a sequence of patches")


def pool_patches(patch_embeds: torch.Tensor) -> torch.Tensor:
    """Pool the patch sequence into a single image embedding.

    After the provided transformer has mixed information across patches, we collapse
    the sequence to one vector that summarizes the whole image (used for retrieval).

    Args:
        patch_embeds: shape (B, N, D_EMBED).

    Returns:
        One image embedding per image, shape (B, D_EMBED).
    """
    # TODO: Average the patch embeddings over the patch (sequence) dimension.
    # HINT: mean over dim=1.
    raise NotImplementedError("TODO: pool the patch sequence into one image embedding")


# ===========================================================================
# PART 2: Align images and captions (the CLIP objective)
# ===========================================================================


def l2_normalize(embeddings: torch.Tensor) -> torch.Tensor:
    """L2-normalize each embedding to unit length.

    On the unit sphere, a dot product IS the cosine similarity, so normalizing before
    comparing makes the similarity depend on direction (meaning) rather than length.

    Args:
        embeddings: shape (B, D).

    Returns:
        Unit-length embeddings, shape (B, D).
    """
    # TODO: Return the embeddings scaled to unit L2 norm along the last dimension.
    # HINT: F.normalize(embeddings, dim=-1).
    raise NotImplementedError("TODO: L2-normalize the embeddings")


def similarity_matrix(
    image_embeds: torch.Tensor, text_embeds: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Build the batch image-text similarity matrix, scaled by temperature.

    Entry (i, j) is the similarity between image i and caption j. Dividing by a small
    temperature sharpens the distribution before the softmax in the loss.

    Args:
        image_embeds: unit image embeddings, shape (B, D).
        text_embeds: unit text embeddings, shape (B, D).
        temperature: scalar > 0.

    Returns:
        Similarity logits of shape (B, B); row i indexes images, column j captions.
    """
    # TODO: Return the matrix of image-text dot products, divided by temperature.
    # HINT: image_embeds @ text_embeds.t(), then divide by temperature.
    raise NotImplementedError("TODO: build the image-text similarity matrix")


def clip_loss(logits: torch.Tensor) -> torch.Tensor:
    """Symmetric CLIP contrastive loss: cross-entropy in both directions.

    For a batch of B matched pairs, the correct caption for image i is caption i (the
    diagonal), and the correct image for caption j is image j. So the targets are just
    0..B-1: cross-entropy over rows is image->text retrieval, over columns is
    text->image retrieval, and CLIP averages the two.

    Args:
        logits: similarity matrix, shape (B, B).

    Returns:
        A scalar loss.
    """
    # TODO: Average row-wise (image->text) and column-wise (text->image) cross-entropy
    #       against the diagonal targets 0..B-1.
    # HINT: labels = torch.arange(B); F.cross_entropy(logits, labels) and the same on
    #       logits.t(); average the two.
    raise NotImplementedError("TODO: build the symmetric CLIP contrastive loss")


# ===========================================================================
# PART 3: Bridge the image into the language model (the projector)
# ===========================================================================


def image_to_prefix(
    image_embeds: torch.Tensor, to_prefix: torch.nn.Linear, prefix_len: int
) -> torch.Tensor:
    """Project one image embedding into `prefix_len` visual prefix vectors.

    The language model's tokens live at a different (wider) width than the image
    embedding. The projector maps the pooled image vector to prefix_len * d_llm
    numbers, which we reshape into prefix_len vectors — the "visual tokens" that will
    sit in front of the text. (The provided `concat_visual_prefix` then prepends them.)

    Args:
        image_embeds: pooled image embeddings, shape (B, D_EMBED).
        to_prefix: the provided Linear(D_EMBED -> prefix_len * d_llm).
        prefix_len: number of visual prefix vectors, K.

    Returns:
        Visual prefix embeddings of shape (B, prefix_len, d_llm).
    """
    # TODO: Apply to_prefix, then reshape the output into (B, prefix_len, d_llm).
    # HINT: to_prefix(image_embeds).view(B, prefix_len, -1).
    raise NotImplementedError("TODO: project the image embedding into visual prefix tokens")


def captioning_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked next-token loss over the response tokens only.

    This is exactly Module 6's masked SFT loss, now with an image condition in the
    prefix. We score next-token prediction only where the target is a response token
    (the mask), so the model is trained to *answer*, not to echo the prompt or the
    visual prefix.

    Args:
        logits: next-token logits, shape (T, vocab_size).
        targets: the token id to predict at each position, shape (T,).
        mask: True at response positions, shape (T,).

    Returns:
        A scalar cross-entropy loss over the masked positions.
    """
    # TODO: Cross-entropy between the logits and targets at the masked positions only.
    # HINT: index both logits and targets with the boolean mask, then F.cross_entropy.
    raise NotImplementedError("TODO: build the masked captioning loss")


def greedy_next_token(logits: torch.Tensor) -> torch.Tensor:
    """Pick the most likely next token from the last position's logits.

    Used to generate a caption or answer one token at a time from the image-conditioned
    prompt, so we can check whether the answer changes when the image changes.

    Args:
        logits: model logits, shape (B, T, vocab_size).

    Returns:
        The argmax token id at the final position, shape (B,).
    """
    # TODO: Return the argmax over the vocabulary at the final sequence position.
    # HINT: logits[:, -1, :].argmax(dim=-1).
    raise NotImplementedError("TODO: greedily pick the next token")
