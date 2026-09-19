:::divider id="divider-exercise" title="Exercise" sub="Align image embeddings with NanoGPT"
:::

---

<!-- .slide: id="exercise-run" -->

## Running the Exercise

- Open `module_08_multimodal/exercise.py` and fill in the eight `NotImplementedError` lines
- Everything else is provided: dataset, encoders, NanoGPT, projector, ops, runner
- Run after each step to see which steps pass

```bash
# Run every step; each is tagged CORRECT, INCORRECT, or INCOMPLETE
cd exercises
uv run python module_08_multimodal/src/main.py

# Run a single step (1-8)
uv run python module_08_multimodal/src/main.py --step 4
```

Model weights live in `data/instruct_model.pt`. <!-- .element: class="text-md" style="margin-top: 22px;" -->

---

<!-- .slide: id="exercise-results" -->

## Reading the Results

For each step the runner prints what your code produced (tensor shapes, training progress, generated captions), then runs the step's **tests** from `tests/`. Each test calls your function on small tensors whose correct answer is known. <!-- .element: class="text-lg" -->

- **CORRECT**: every test for the step passed
- **INCORRECT**: your code ran but a test failed; the expected and actual values are printed beneath
- **INCOMPLETE**: the function still raises `NotImplementedError`, or the step needs an earlier one that does

The tag sits on the step's header line. A tensor can have the right shape and still hold the wrong numbers, so the tests check values as well as shapes. <!-- .element: class="text-lg" style="margin-top: 15px;" -->

---

:::terminal id="exercise-results-example" title="What an INCORRECT Step Looks Like" cmd="uv run python module_08_multimodal/src/main.py --step 1" maxw="1000px" caption="Here <code>patchify()</code> skipped the permute and reshaped straight to <code>(B, N, C, P, P)</code>. The shape is right, so the runner's shape line looks fine, but each patch is a strip of whole pixel rows. In the first test the first patch came back as the top row of the image, and the message points at the missing permute."
<span class="header">=== Step 1: patchify() ===</span> <span class="t-fail">INCORRECT</span>
Dataset: 340 training + 60 held-out scenes; sample grid in output/sample_scenes.png
Images (340, 3, 32, 32) -&gt; patches (340, 16, 3, 8, 8)
  each 32 x 32 image is now a sequence of 16 patches of 8 x 8 pixels
  <span class="t-fail">INCORRECT</span>  patches come in row-major order: 4 x 4 image 0-15 gives [[0,1],[4,5]], [[2,3],[6,7]], ...
             expected the first two patches [[[0, 1], [4, 5]], [[2, 3], [6, 7]]]
             got [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
             (did you permute the two grid axes next to the batch before the last reshape?)
  <span class="success">CORRECT</span>    returns (B, N, C, P, P): a 2 x 3 x 8 x 8 batch with P=4 gives (2, 4, 3, 4, 4)
  <span class="t-fail">INCORRECT</span>  un-patchifying a random 2 x 3 x 4 x 6 batch gives back the original images
             the patches do not reassemble into the input images
             (check the permute: the grid rows H/P come before the grid columns W/P)
:::

---

<!-- .slide: id="exercise-overview" -->

## Exercise: Make the Bridge Visible

<div class="card-grid cols-3">
<div class="card"><h4>Part 1 &middot; Vision tower</h4><p>Steps 1&ndash;2: patchify a 32&times;32 image, then pool the mixed patches into one embedding. Flatten, project, and positions are provided.</p></div>
<div class="card"><h4>Part 2 &middot; CLIP</h4><p>Steps 3&ndash;5: normalize, build the similarity matrix, and the contrastive loss; retrieval climbs from ~1.7% to ~77%.</p></div>
<div class="card"><h4>Part 3 &middot; Bridge</h4><p>Steps 6&ndash;8: project the image into NanoGPT as visual prefix tokens, mask the captioning loss, and greedily decode.</p></div>
</div>

The payoff: the **same** prompt "describe the image" returns a **different, correct** caption per image. Proof the model uses the image, not language priors. <!-- .element: class="text-lg" -->

Each step is one line to fill in, and every step has its own tests in `tests/`. <!-- .element: class="text-lg" style="margin-top: 10px;" -->

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

:::terminal id="exercise-output-vision" title="Part 1: One Embedding per Image" cmd="uv run python module_08_multimodal/src/main.py" maxw="1000px" caption="Each 32&times;32 image becomes 16 patch tokens, and pooling turns the mixed patches into one 64-number embedding per image. The tower is still untrained; Step 5 trains it."
<span class="header">=== Step 1: patchify() ===</span> <span class="success">CORRECT</span>
Dataset: 340 training + 60 held-out scenes; sample grid in output/sample_scenes.png
Images (340, 3, 32, 32) -&gt; patches (340, 16, 3, 8, 8)
  each 32 x 32 image is now a sequence of 16 patches of 8 x 8 pixels
  <span class="success">CORRECT</span>    patches come in row-major order: 4 x 4 image 0-15 gives [[0,1],[4,5]], [[2,3],[6,7]], ...
  <span class="success">CORRECT</span>    returns (B, N, C, P, P): a 2 x 3 x 8 x 8 batch with P=4 gives (2, 4, 3, 4, 4)
  <span class="success">CORRECT</span>    un-patchifying a random 2 x 3 x 4 x 6 batch gives back the original images

<span class="header">=== Step 2: pool_patches() ===</span> <span class="success">CORRECT</span>
Vision tower (113,472 parameters, not trained yet) on 340 images:
  patchify (Step 1)             (340, 16, 3, 8, 8)
  flatten each patch            (340, 16, 192)
  project, add positions, mix   (340, 16, 64)
  pool (Step 2)                 <span class="success">(340, 64)</span>: one embedding per image
  <span class="success">CORRECT</span>    averages the patches: [1,2,3] and [3,4,5] pool to [2, 3, 4]
  <span class="success">CORRECT</span>    returns one vector per image: (B, N, D) = (2, 16, 64) gives (2, 64)
  <span class="success">CORRECT</span>    pools each image separately: patches [0,0],[2,2] -&gt; [1,1] and [10,10],[20,20] -&gt; [15,15]

<span class="header">=== Step 3: l2_normalize() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">TODO: L2-normalize the embeddings</span>

<span class="skipped">...</span>
:::

---

:::step id="exercise-step3" title="Step 3: l2_normalize()"
```python
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
    raise NotImplementedError("TODO: L2-normalize the embeddings")
```
+++
**Hint:** `F.normalize` does this in one call; normalize along the last dimension.
+++
**Answer:**

```python
return F.normalize(embeddings, dim=-1)
```
:::

---

:::step id="exercise-step4" title="Step 4: similarity_matrix()"
```python
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
    raise NotImplementedError("TODO: build the image-text similarity matrix")
```
+++
**Hint:** matrix-multiply the image embeddings by the transposed text embeddings, then divide by `temperature`.
+++
**Answer:**

```python
return image_embeds @ text_embeds.t() / temperature
```
:::

---

:::step id="exercise-step5" title="Step 5: clip_loss()"
```python
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
    raise NotImplementedError("TODO: build the symmetric CLIP contrastive loss")
```
+++
**Hint:** row i's correct label is i, so `torch.arange` builds the labels; take `F.cross_entropy` on `logits` and again on its transpose, then average the two.
+++
**Answer:**

```python
labels = torch.arange(logits.shape[0], device=logits.device)
return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
```
:::

---

:::terminal id="exercise-output-clip" title="Part 2: The Contrastive Loss Falls, Retrieval Climbs" cmd="uv run python module_08_multimodal/src/main.py" maxw="1000px" caption="From 1.7% (chance is 1/60) to 76.7% held-out retrieval. The loss is Module 5's cross-entropy, with the other captions in the batch as the classes. The last test requires retrieval above 50%."
<span class="header">=== Step 1: patchify() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 2: pool_patches() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 3: l2_normalize() ===</span> <span class="success">CORRECT</span>
<span class="t-gray">  after normalizing:  1.000 to 1.000</span>
<span class="header">=== Step 4: similarity_matrix() ===</span> <span class="success">CORRECT</span>
<span class="t-gray">  Retrieval accuracy before training: 1.7%  (chance is 1/60)</span>

<span class="header">=== Step 5: clip_loss() ===</span> <span class="success">CORRECT</span>
CLIP training: 400 steps of 32 image-caption pairs
  step    1   contrastive loss  3.468   batch retrieval acc  3.1%
  step   50   contrastive loss  0.626   batch retrieval acc 75.0%
  step  100   contrastive loss  0.371   batch retrieval acc 78.1%
  <span class="skipped">...</span>
  step  400   contrastive loss  0.477   batch retrieval acc 68.8%
  <span class="success">Held-out retrieval accuracy: 1.7% before training, 76.7% after</span>
  Saved retrieval heatmap to output/retrieval_heatmap.png
  <span class="success">CORRECT</span>    a uniform 4 x 4 matrix (no idea which caption matches) gives ln 4 = 1.386
  <span class="success">CORRECT</span>    a strong diagonal, 10 * identity(3), gives nearly zero loss (about 0.0001)
  <span class="success">CORRECT</span>    is symmetric: an uneven 3 x 3 matrix and its transpose give the same loss
  <span class="success">CORRECT</span>    equals the mean of F.cross_entropy over rows and over columns (random 5 x 5 logits)
  <span class="success">CORRECT</span>    held-out retrieval accuracy climbs above 50% after training (chance is 1/60)

<span class="header">=== Step 6: image_to_prefix() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">TODO: project the image embedding into visual prefix tokens</span>

<span class="skipped">...</span>
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
    raise NotImplementedError("TODO: project the image embedding into visual prefix tokens")
```
+++
**Hint:** pass the embeddings through `to_prefix`, then use `.view` to split the flat output into `prefix_len` vectors per image (let `-1` infer the last size).
+++
**Answer:**

```python
return to_prefix(image_embeds).view(image_embeds.shape[0], prefix_len, -1)
```
:::

---

:::step id="exercise-step7" title="Step 7: captioning_loss()"
```python
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

:::terminal id="exercise-output-bridge-train" title="Part 3: Training the Bridge" cmd="uv run python module_08_multimodal/src/main.py" maxw="1000px" caption="Four visual tokens at the language model's width sit in front of the 21 prompt tokens. The captioning loss falls from 8.9 to 0.009, and it scores only the response tokens."
<span class="header">=== Step 1: patchify() ===</span> <span class="success">CORRECT</span>
<span class="skipped">  ...</span>
<span class="header">=== Step 5: clip_loss() ===</span> <span class="success">CORRECT</span>
<span class="t-gray">  Held-out retrieval accuracy: 1.7% before training, 76.7% after</span>

<span class="header">=== Step 6: image_to_prefix() ===</span> <span class="success">CORRECT</span>
Projector: Linear(64 -&gt; 4 x 128), 33,280 parameters
  image embedding (1, 64) -&gt; visual prefix (1, 4, 128)
  prompt 'describe the image' as token embeddings (1, 21, 128)
  prefix + prompt, the language model's input (1, 25, 128)
  <span class="success">CORRECT</span>    returns (B, prefix_len, d_llm): 2 embeddings, K=4, Linear(8 -&gt; 64) give (2, 4, 16)
  <span class="success">CORRECT</span>    applies the projector, weights and bias: matches the Linear computed by hand
  <span class="success">CORRECT</span>    prefix vector k is slice k of the output: [0..11], K=3 gives [0..3], [4..7], [8..11]

<span class="header">=== Step 7: captioning_loss() ===</span> <span class="success">CORRECT</span>
Bridge training: 400 steps of 16 examples (a caption and 4 questions per scene)
  step    1   captioning loss  8.901
  step   50   captioning loss  0.216
  <span class="skipped">...</span>
  step  400   captioning loss  <span class="success">0.009</span>
  <span class="success">CORRECT</span>    averages -log p(target) over the response only: p = 0.5 and 0.75 give 0.490
  <span class="success">CORRECT</span>    scrambling the logits at masked-out positions leaves the loss unchanged
  <span class="success">CORRECT</span>    matches F.cross_entropy with the prompt positions set to ignore_index (random 6 x 10)
  <span class="success">CORRECT</span>    bridge training takes the captioning loss from about 9 to under 0.1 (last 50 steps)

<span class="header">=== Step 8: greedy_next_token() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">TODO: greedily pick the next token</span>
:::

---

:::step id="exercise-step8" title="Step 8: greedy_next_token()"
```python
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
    raise NotImplementedError("TODO: greedily pick the next token")
```
+++
**Hint:** slice out the last position along the sequence axis, then `argmax` over the vocabulary dimension.
+++
**Answer:**

```python
return logits[:, -1, :].argmax(dim=-1)
```
:::

---

:::terminal id="exercise-output-bridge" title="Part 3: The Answer Follows the Image" cmd="uv run python module_08_multimodal/src/main.py" maxw="1000px" caption="Held-out captions are 87% exact-match, and the same prompt returns a different, correct caption for each image. The last test checks that the captions differ: the model reads the image instead of guessing from priors."
<span class="header">=== Step 1: patchify() ===</span> <span class="success">CORRECT</span>
<span class="skipped">  ...</span>
<span class="header">=== Step 7: captioning_loss() ===</span> <span class="success">CORRECT</span>

<span class="header">=== Step 8: greedy_next_token() ===</span> <span class="success">CORRECT</span>
Before bridge training (projector is random), 'describe the image' gives:
    image['red triangle above blue triangle'] -&gt; <span class="t-fail">'tookn'</span>
    image['blue triangle above green triangle'] -&gt; <span class="t-fail">'toomme'</span>
    image['red square above red circle'] -&gt; <span class="t-fail">'toomb'</span>

After bridge training, held-out caption exact-match: <span class="success">52/60 = 86.7%</span>

Same prompt, different images (the grounding test):
    describe -&gt; 'red triangle above blue triangle'    (correct)
    describe -&gt; 'blue triangle above green triangle'  (correct)
    describe -&gt; 'red square above red circle'         (correct)

Grounded visual questions on one held-out image:
    what color is on top?      -&gt; 'red'      (correct)
    <span class="skipped">...</span>
    what shape is on bottom?   -&gt; 'triangle' (correct)
  <span class="success">CORRECT</span>    uses the last position: logits [0,5,1] then [2,0,9] give token 2
  <span class="success">CORRECT</span>    returns one token id per sequence: a (2, 3, 5) batch gives shape (2,)
  <span class="success">CORRECT</span>    returns integer token ids (torch.long), not logit values
  <span class="success">CORRECT</span>    held-out caption exact-match is at least 50% (the scenes were never trained on)
  <span class="success">CORRECT</span>    the same prompt returns a different caption for each of the 3 demo images
:::

---

<!-- .slide: id="exercise-extra-credit" -->

## Extra Credit

- **Patch-size sweep.** Set `PATCH_SIZE` to `4`, `8`, `16` in `src/vision.py` and measure the detail-versus-cost tradeoff.
- **Image ablation.** Zero the visual prefix (`prefix = prefix * 0`) and confirm the captioner falls back to **language priors**.
- **Held-out composition.** Remove one color-shape pairing from training and test whether grounding **composes** to it, or the model just memorized captions.
- **Visual hallucination probe.** Ask "what color is the triangle?" about an image with **no** triangle. Does the model admit absence or guess? <!-- .element: class="text-lg" style="margin-top: 8px;" -->
