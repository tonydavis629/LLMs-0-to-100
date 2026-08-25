# Module 8: Align Image Embeddings with NanoGPT

## Overview

Build a tiny **vision-language model** on a synthetic image-caption dataset. The goal
is not a useful visual assistant &mdash; it is to make the **multimodal bridge**
visible: how non-text data (an image) becomes something a language model can condition
on. You do it in three stages.

1. **Vision tower.** Turn a `32 x 32` image into one embedding: split it into patches
   (step 1) and, after the provided tower flattens, projects, adds positions, and mixes
   them, pool the patch sequence into a single vector (step 2). This is the Vision
   Transformer recipe in miniature.
2. **CLIP-style alignment.** Train an image encoder and a text encoder so that each
   image sits close to its own caption and far from the others, using a symmetric
   contrastive loss (steps 3&ndash;5). Retrieval accuracy climbs from chance (`~1/64`)
   toward `1.0`.
3. **The bridge.** Project the image embedding into the language model's hidden width
   as a few **visual prefix tokens** (step 6), prepend them to the text-token
   embeddings, and finetune so the model **captions** the image and **answers
   questions** about it (steps 7&ndash;8). The payoff: the answer changes when the
   image changes.

The images, the vision/text encoder parameters, the NanoGPT language model (the
Module 6/7 instruct checkpoint), the projector, the small mechanical tensor ops in
`src/ops.py`, and the runner are all provided. Your job is the eight one-line steps in
`exercise.py`.

## Setup

There is one shared environment for the whole repo. From the repo root:

```bash
uv sync
```

## Running

```bash
uv run python exercises/module_08_multimodal/src/main.py
```

The runner detects which steps you have implemented and skips the rest, so you can fill
in one step at a time and re-run immediately. It prints the dataset size, image
resolution, patch size, number of visual tokens per image, model parameter counts, the
contrastive loss and retrieval accuracy during CLIP training, the held-out retrieval
accuracy before and after, the captioning loss during bridge training, and a set of
image-conditioned generations. It also saves two figures to `output/`:

- `sample_scenes.png` &mdash; a grid of synthetic scenes with their captions.
- `retrieval_heatmap.png` &mdash; the image-text similarity matrix (a bright diagonal
  means retrieval works).

## What to implement

Open `exercise.py` and fill in each `raise NotImplementedError(...)` line. Each needs
only one expression or one short line.

| Step | Function | What it does |
|------|----------|--------------|
| 1 | `patchify()` | Split each image into a sequence of `P x P` patches |
| 2 | `pool_patches()` | Pool the patch sequence into one image embedding |
| 3 | `l2_normalize()` | L2-normalize image and text embeddings |
| 4 | `similarity_matrix()` | Build the batch image-text similarity matrix |
| 5 | `clip_loss()` | Symmetric image&rarr;text and text&rarr;image cross-entropy |
| 6 | `image_to_prefix()` | Project the image embedding into `K` visual prefix tokens |
| 7 | `captioning_loss()` | Masked next-token loss over the response tokens only |
| 8 | `greedy_next_token()` | Argmax the last position to decode one token |

Steps 1&ndash;5 unlock the CLIP alignment phase; steps 6&ndash;8 unlock the bridge.

The provided `src/ops.py` supplies the smaller mechanical steps around these
(`flatten_patches`, `project_patches`, `add_position_embeddings`, `encode_text`,
`retrieval_accuracy`, `concat_visual_prefix`). The model (`src/model.py`), tokenizer
(`src/tokenizer.py`), dataset (`src/data.py`), vision/text encoders (`src/vision.py`),
plotting (`src/visualization.py`), and runner (`src/main.py`) are all provided too. You
only edit `exercise.py`.

## Data

- `data/shapes_dataset.pt` &mdash; the bundled synthetic dataset: `340` train and `60`
  held-out `32 x 32` scenes, each with two colored shapes (one above the other), a
  caption (`"red square above blue circle"`), and derived visual questions
  (`"what color is on top?" -> "red"`). Held-out scenes are never trained on.
  (Regenerate with `src/data.py`.)
- `data/instruct_model.pt` &mdash; the language model: the Module 6/7 TinyGPT instruct
  checkpoint (character-level, vocabulary 69). It plays the role of NanoGPT at the end
  of the bridge; we finetune it alongside the projector so it learns to read the visual
  prefix. (Regenerate it with `../module_07_rl/solution/src/make_instruct_checkpoint.py`.)

The tokenizer is the Module 6 vocabulary: 65 characters plus four atomic special
tokens (`<|user|>`, `<|assistant|>`, `<|end|>`, `<|pad|>`). Captions and questions use
only lowercase letters, spaces, and `?`, all in the base vocabulary.

## Expected result

With all eight steps implemented, a full run (about a minute on a laptop CPU)
reaches roughly:

- **Held-out retrieval accuracy** `~77%` (from `~1.7%` at chance).
- **Held-out caption exact-match** `~87%`.
- **Grounded visual questions** answered correctly, and the same `"describe the image"`
  prompt returning a **different, correct** caption for each image &mdash; the model is
  using the image, not language priors.

## Extra credit

- **Patch-size sweep.** Change `PATCH_SIZE` in `src/vision.py` to `4`, `8`, and `16` and
  measure the detail-versus-cost tradeoff (more patches = more visual tokens = more
  attention cost). Note that the patch projection input width changes with it.
- **Image ablation.** Zero out the visual prefix before the LM (`prefix = prefix * 0`)
  and confirm the captioner falls back to language priors &mdash; it can no longer tell
  which colors or shapes are present.
- **Held-out composition.** Remove one color-shape pairing (e.g. `green triangle`) from
  the training scenes and check whether the model can still describe it in held-out
  images: does visual grounding compose, or did it memorize captions?
- **Hard negatives.** Build a retrieval batch where captions differ by a single
  attribute (`red square` vs `blue square`) and watch whether retrieval accuracy drops.
- **Visual hallucination probe.** Ask `"what color is the triangle?"` about an image
  with no triangle and see whether the model admits absence or guesses anyway.
- **Projector capacity sweep.** Replace the single-linear projector with a two-layer
  MLP and compare bridge convergence and caption accuracy.
