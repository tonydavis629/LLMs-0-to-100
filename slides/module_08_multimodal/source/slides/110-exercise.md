:::divider id="divider-exercise" title="Exercise" sub="Align image embeddings with NanoGPT"
:::

---

<!-- .slide: id="exercise-run" -->

## Running the Exercise

- Open `module_08_multimodal/exercise.py` and fill in the eight `NotImplementedError` lines
- Everything else is provided: dataset, encoders, NanoGPT, projector, ops, runner
- Run after each step; unfinished steps are skipped automatically

```bash
# Build a tiny vision-language model on synthetic shapes
cd exercises
uv run python module_08_multimodal/src/main.py
```

Model weights live in `data/instruct_model.pt`. <!-- .element: class="text-md" style="margin-top: 22px;" -->

---

<!-- .slide: id="exercise-overview" -->

## Exercise: Make the Bridge Visible

<div class="card-grid cols-3">
<div class="card"><h4>Part 1 &middot; Vision tower</h4><p>Steps 1&ndash;2: patchify a 32&times;32 image, then pool the mixed patches into one embedding. Flatten, project, and positions are provided.</p></div>
<div class="card"><h4>Part 2 &middot; CLIP</h4><p>Steps 3&ndash;5: normalize, build the similarity matrix, and the contrastive loss; retrieval climbs from ~1.7% to ~77%.</p></div>
<div class="card"><h4>Part 3 &middot; Bridge</h4><p>Steps 6&ndash;8: project the image into NanoGPT as visual prefix tokens, mask the captioning loss, and greedily decode.</p></div>
</div>

The payoff: the **same** prompt "describe the image" returns a **different, correct** caption per image. Proof the model uses the image, not language priors. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="exercise-scenes" -->

## The Synthetic Dataset

<div class="img-figure">
  <img src="images/sample_scenes.png" alt="Grid of synthetic 32x32 scenes: two colored shapes each, with captions like 'red square above blue circle'">
</div>

Two colored shapes per scene (one above the other), a caption, and derived questions. Balanced by construction: language priors alone cannot answer, so the image must be read. (Actual output of the exercise.) <!-- .element: class="text-lg" style="margin-top: 8px;" -->

---

:::step id="exercise-step1" title="Step 1: patchify()"
```python
def patchify(images, patch_size):
    """Split each image into a grid of non-overlapping square patches."""
    B, C, H, W = images.shape
    # TODO: Reshape (B, C, H, W) into (B, N, C, P, P) patches with N = (H/P)*(W/P).
    raise NotImplementedError("TODO: split each image into a sequence of patches")
```
+++
**Hint:** reshape to `(B, C, H/P, P, W/P, P)`, permute the two grid axes next to the batch, then reshape to `(B, N, C, P, P)`.
+++
**Answer:**

```python
return (
    images.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    .permute(0, 2, 4, 1, 3, 5)
    .reshape(B, -1, C, patch_size, patch_size)
)
```
:::

---

:::step id="exercise-step2" title="Step 2: pool_patches()"
```python
def pool_patches(patch_embeds):
    """Pool the patch sequence into a single image embedding."""
    # TODO: Average the patch embeddings over the patch (sequence) dimension.
    raise NotImplementedError("TODO: pool the patch sequence into one image embedding")
```
+++
**Hint:** mean over `dim=1`.
+++
**Answer:**

```python
return patch_embeds.mean(dim=1)
```
:::

---

:::step id="exercise-step3" title="Step 3: l2_normalize()"
```python
def l2_normalize(embeddings):
    """L2-normalize each embedding to unit length (dot product becomes cosine)."""
    # TODO: Return the embeddings scaled to unit L2 norm along the last dimension.
    raise NotImplementedError("TODO: L2-normalize the embeddings")
```
+++
**Hint:** `F.normalize(embeddings, dim=-1)`.
+++
**Answer:**

```python
return F.normalize(embeddings, dim=-1)
```
:::

---

:::step id="exercise-step4" title="Step 4: similarity_matrix()"
```python
def similarity_matrix(image_embeds, text_embeds, temperature):
    """Build the batch image-text similarity matrix, scaled by temperature."""
    # TODO: Return the matrix of image-text dot products, divided by temperature.
    raise NotImplementedError("TODO: build the image-text similarity matrix")
```
+++
**Hint:** `image_embeds @ text_embeds.t()`, then divide by `temperature`.
+++
**Answer:**

```python
return image_embeds @ text_embeds.t() / temperature
```
:::

---

:::step id="exercise-step5" title="Step 5: clip_loss()"
```python
def clip_loss(logits):
    """Symmetric CLIP contrastive loss: cross-entropy in both directions."""
    # TODO: Average row-wise (image->text) and column-wise (text->image) cross-entropy
    #       against the diagonal targets 0..B-1.
    raise NotImplementedError("TODO: build the symmetric CLIP contrastive loss")
```
+++
**Hint:** `labels = torch.arange(B)`; `F.cross_entropy(logits, labels)` and the same on `logits.t()`; average the two.
+++
**Answer:**

```python
labels = torch.arange(logits.shape[0], device=logits.device)
return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
```
:::

---

:::terminal id="exercise-output-clip" title="Part 2: The Contrastive Loss Falls, Retrieval Climbs" cmd="uv run python module_08_multimodal/src/main.py" caption="From 1.7% (chance is 1/64) to 76.7% held-out retrieval. The loss is the same cross-entropy as Module 5, but the classes are the other captions in the batch."
<span class="header">MODULE 8: Align image embeddings with NanoGPT</span>
Dataset: 340 train + 60 held-out scenes
Image: 3 x 32 x 32   Patch: 8 x 8   Visual tokens per image: 16
Vision encoder params: 113,472   Text encoder params: 107,072
Language model params: 818,560   Projector params: 33,280

<span class="header">PHASE 1: CLIP-style image-text alignment</span>
  Held-out retrieval accuracy before training: 1.7%  (chance is ~1/64)
  step    1   contrastive loss  3.468   batch retrieval acc  3.1%
  step  100   contrastive loss  0.371   batch retrieval acc 78.1%
  step  250   contrastive loss  0.143   batch retrieval acc 90.6%
  step  400   contrastive loss  0.477   batch retrieval acc 68.8%
  <span class="success">Held-out retrieval accuracy after training:  76.7%</span>
:::

---

<!-- .slide: id="exercise-heatmap" -->

## The Retrieval Matrix After Training

<div class="img-figure">
  <img src="images/retrieval_heatmap.png" alt="Image-text similarity heatmap with a bright yellow diagonal, showing correct retrieval">
</div>

Rows are images, columns are captions. The **bright diagonal** is each image matching its own caption, exactly what the contrastive loss maximized. (Actual output of the exercise.) <!-- .element: class="text-lg" style="margin-top: 8px;" -->

---

:::step id="exercise-step6" title="Step 6: image_to_prefix()"
```python
def image_to_prefix(image_embeds, to_prefix, prefix_len):
    """Project one image embedding into prefix_len visual prefix vectors."""
    # TODO: Apply to_prefix, then reshape the output into (B, prefix_len, d_llm).
    raise NotImplementedError("TODO: project the image embedding into visual prefix tokens")
```
+++
**Hint:** `to_prefix(image_embeds).view(B, prefix_len, -1)`.
+++
**Answer:**

```python
return to_prefix(image_embeds).view(image_embeds.shape[0], prefix_len, -1)
```
:::

---

:::step id="exercise-step7" title="Step 7: captioning_loss()"
```python
def captioning_loss(logits, targets, mask):
    """Masked next-token loss over the response tokens only (Module 6, with an image)."""
    # TODO: Cross-entropy between the logits and targets at the masked positions only.
    raise NotImplementedError("TODO: build the masked captioning loss")
```
+++
**Hint:** index both `logits` and `targets` with the boolean `mask`, then `F.cross_entropy`.
+++
**Answer:**

```python
return F.cross_entropy(logits[mask], targets[mask])
```
:::

---

:::step id="exercise-step8" title="Step 8: greedy_next_token()"
```python
def greedy_next_token(logits):
    """Pick the most likely next token from the last position's logits."""
    # TODO: Return the argmax over the vocabulary at the final sequence position.
    raise NotImplementedError("TODO: greedily pick the next token")
```
+++
**Hint:** `logits[:, -1, :].argmax(dim=-1)`.
+++
**Answer:**

```python
return logits[:, -1, :].argmax(dim=-1)
```
:::

---

:::terminal id="exercise-output-bridge" title="Part 3: The Answer Follows the Image" cmd="uv run python module_08_multimodal/src/main.py" caption="Held-out captions are 87% exact-match, and the SAME prompt returns a different, correct caption per image. The model is reading the image, not guessing from priors."
<span class="header">PHASE 2: bridge the image into the language model</span>
  Before bridge training (projector is random), 'describe the image' gives:
    image['red triangle above blue triangle'] -&gt; 'tookn'
    image['blue triangle above green triangle'] -&gt; 'toomme'
    image['red square above red circle'] -&gt; 'toomb'

  step    1   captioning loss  8.901
  step  200   captioning loss  0.142
  step  400   captioning loss  0.009

<span class="header">AFTER BRIDGE TRAINING: does the answer follow the image?</span>
  <span class="success">Held-out caption exact-match accuracy: 52/60 = 86.7%</span>

  Same prompt, different images (the grounding test):
    describe -&gt; 'red triangle above blue triangle'   (correct)
    describe -&gt; 'blue triangle above green triangle'   (correct)
    describe -&gt; 'red square above red circle'   (correct)

  Grounded visual questions on one held-out image:
    what color is on top?      -&gt; 'red'        (correct)
    what shape is on bottom?   -&gt; 'triangle'   (correct)
:::

---

<!-- .slide: id="exercise-extra-credit" -->

## Extra Credit

- **Patch-size sweep.** Set `PATCH_SIZE` to `4`, `8`, `16` in `src/vision.py` and measure the detail-versus-cost tradeoff.
- **Image ablation.** Zero the visual prefix (`prefix = prefix * 0`) and confirm the captioner falls back to **language priors**.
- **Held-out composition.** Remove one color-shape pairing from training and test whether grounding **composes** to it, or the model just memorized captions.
- **Visual hallucination probe.** Ask "what color is the triangle?" about an image with **no** triangle. Does the model admit absence or guess? <!-- .element: class="text-lg" style="margin-top: 8px;" -->
