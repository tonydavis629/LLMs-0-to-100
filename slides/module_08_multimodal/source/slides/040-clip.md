:::divider id="divider-clip" title="CLIP" sub="Contrastive image-text pretraining"
:::

---

<!-- .slide: id="fig-radford-clip" -->

:::figure img="images/radford.jpg" name="Alec Radford and the CLIP team" kicker="Learning Transferable Visual Models From Natural Language Supervision (CLIP, OpenAI, 2021)"
- Aligned images and text in one shared embedding space with **natural-language supervision**
- Learned from 400M noisy web image-caption pairs, not a hand-labeled class list
- Generalized zero-shot to tasks it was never explicitly trained on
:::

---

<!-- .slide: id="clip-architecture" -->

## CLIP Learns a Shared Space, Not Language

CLIP **does not generate language**. It only learns where images and text sit relative to each other.

:::columns cols="2" gap="34px"
**Two encoders**

- **Image encoder**: image to vector
- **Text encoder**: caption to vector
+++
**One objective**

- Pull **matched** image-text pairs together
- Push **mismatched** pairs apart
:::

The result is a shared space where the text "a photo of a dog" lands near pictures of dogs. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="clip-matrix" -->

## The Batch Is the Supervision

For a batch of $B$ image-caption pairs, compute a $B \times B$ similarity matrix:

$$s_{ij} = \frac{\langle \hat{v}_i, \hat{t}_j \rangle}{\tau}$$

<div class="diagram"><svg viewBox="0 0 430 320" width="430" role="img" aria-label="4 by 4 image-text similarity matrix; rows are image thumbnails, columns are captions, the matched diagonal is highlighted">
<text x="278" y="16" fill="#8fa0bd" font-size="12" text-anchor="middle">captions</text>
<text x="14" y="182" fill="#8fa0bd" font-size="12" text-anchor="middle" transform="rotate(-90 14 182)">images</text>
<g text-anchor="middle" fill="#e8eaf0" font-size="11"><text x="182" y="34">T1</text><text x="246" y="34">T2</text><text x="310" y="34">T3</text><text x="374" y="34">T4</text></g>
<g text-anchor="middle" font-size="9.5" fill="#c7d0e0"><text x="182" y="47">red</text><text x="182" y="58">square</text><text x="246" y="47">blue</text><text x="246" y="58">circle</text><text x="310" y="47">green</text><text x="310" y="58">triangle</text><text x="374" y="47">red</text><text x="374" y="58">circle</text></g>
<g><rect x="80" y="86" width="24" height="24" fill="#e05252"/><circle cx="92" cy="154" r="12" fill="#4a9eff"/><polygon points="92,197 79,221 105,221" fill="#50c878"/><circle cx="92" cy="266" r="12" fill="#e05252"/></g>
<g text-anchor="middle" fill="#e8eaf0" font-size="11"><text x="122" y="102">I1</text><text x="122" y="158">I2</text><text x="122" y="214">I3</text><text x="122" y="270">I4</text></g>
<g stroke="#2a3550" stroke-width="1.5">
<rect x="150" y="70" width="64" height="56" fill="#50c878"/><rect x="214" y="70" width="64" height="56" fill="#1b2436"/><rect x="278" y="70" width="64" height="56" fill="#1b2436"/><rect x="342" y="70" width="64" height="56" fill="#1b2436"/>
<rect x="150" y="126" width="64" height="56" fill="#1b2436"/><rect x="214" y="126" width="64" height="56" fill="#50c878"/><rect x="278" y="126" width="64" height="56" fill="#1b2436"/><rect x="342" y="126" width="64" height="56" fill="#1b2436"/>
<rect x="150" y="182" width="64" height="56" fill="#1b2436"/><rect x="214" y="182" width="64" height="56" fill="#1b2436"/><rect x="278" y="182" width="64" height="56" fill="#50c878"/><rect x="342" y="182" width="64" height="56" fill="#1b2436"/>
<rect x="150" y="238" width="64" height="56" fill="#1b2436"/><rect x="214" y="238" width="64" height="56" fill="#1b2436"/><rect x="278" y="238" width="64" height="56" fill="#1b2436"/><rect x="342" y="238" width="64" height="56" fill="#50c878"/>
</g>
</svg></div>

Rows are images, columns are captions. The **diagonal** is the matched pairs: maximize it, minimize everything off it. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="clip-loss" -->

## The Loss Uses Every Other Pair as a Negative

For image $i$, take a softmax over all $B$ captions in the batch:

$$\mathcal{L}\_i = -\log \frac{\exp(\textcolor{#50c878}{s\_{ii}})}{\exp(\textcolor{#50c878}{s\_{ii}}) + \sum\_{j \neq i} \exp(\textcolor{#e05252}{s\_{ij}})}$$

:::columns cols="2" gap="34px"
**One positive (diagonal)**

- Caption $i$, the one actually paired with image $i$
- The gradient pushes its similarity **up**
+++
**$B-1$ negatives (off-diagonal)**

- Every other caption in the batch
- The gradient pushes each one **down**, the most similar (hardest) ones the most
:::

The full loss averages this over rows (image to text) and columns (text to image). CLIP trained with $B = 32{,}768$, so each image was contrasted against 32,767 wrong captions per step. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="clip-zeroshot" -->

## Zero-Shot Classification Falls Out for Free

CLIP has no classifier head, yet it can classify:

1. Write **one text prompt per class**: "a photo of a cat", "a photo of a dog", &hellip;
2. **Embed** each prompt and embed the image
3. Pick the class prompt with the **highest similarity**

- The "classifier" is just more text: swap the prompts, get a new classifier, no retraining
- Noisy but abundant captions teach categories, attributes, styles, and concepts without a hand-labeled class list

---

<!-- .slide: id="clip-limits" -->

## What CLIP Is Not

CLIP aligns one **global** image vector with one global caption vector, and nothing finer.

<div class="card-grid cols-2">
<div class="card"><h4>A bag of words</h4><p>CLIP often scores "the horse is eating the grass" and "the grass is eating the horse" almost the same. It matches which words appear, not how they relate (Yuksekgonul et al., 2023).</p></div>
<div class="card"><h4>Not an assistant</h4><p>By itself it cannot produce detailed answers, follow instructions, count reliably, localize objects precisely, or reason over multiple images.</p></div>
</div>

Why: the contrastive task rarely needs word order. A batch almost never contains two captions with the same words in a different order, so ignoring order costs nothing during training. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="clip-slot" -->

## Where CLIP Fits in the Diagram

<div class="img-figure"><img src="images/arch_clip.svg" alt="The carried diagram with only the image encoder slot filled: a CLIP or SigLIP encoder trained contrastively; connector and language model still to come" style="max-height: 360px;"></div>

CLIP fills one slot: a pretrained **image encoder** whose features already sit near related text. An assistant also needs the language model to read those features while it generates, which is the next section. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="sq-weak-supervision" -->

## <span class="side-quest-badge">Side Quest</span> Why Captions Are Weak Supervision

- A caption rarely names every object, relation, style, and spatial detail: "a dog" ignores the breed, the grass, the ball in its mouth
- Contrastive learning still works: **scale and diversity** create useful pressure across millions of pairs
- The cost: **caption bias**, missing whatever the captions systematically leave out

Lots of noisy labels beat a few clean ones here. Whisper, later in this module, makes the same trade for speech. <!-- .element: class="text-lg" style="margin-top: 10px;" -->
