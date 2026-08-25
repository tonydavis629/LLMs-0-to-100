:::divider id="divider-eval" title="Evaluation, Safety, and Failure Modes" sub="Why seeing is not believing"
:::

---

<!-- .slide: id="eval-harder" -->

## Multimodal Evaluation Is Harder

Ground truth often depends on **visual detail, spatial relations, OCR, temporal events, or subjective judgment**. A keyword match captures none of these.

<div class="card-grid cols-3">
<div class="card"><h4>VQA and captioning</h4><p>Image-language understanding</p></div>
<div class="card"><h4>OCR and documents</h4><p>Reading text and layout</p></div>
<div class="card"><h4>Chart and diagram</h4><p>Marks, axes, legends, quantities</p></div>
<div class="card"><h4>MMMU</h4><p>College-level multi-discipline visual reasoning</p></div>
<div class="card"><h4>Video QA</h4><p>Temporal understanding</p></div>
<div class="card"><h4>MathVista</h4><p>Math reasoning in visual contexts</p></div>
</div>

**Exact match is brittle**: "two" and "there are two" are equivalent; a keyword can match while the answer is wrong. Module 11's evaluation tools return here. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="eval-hallucination" -->

## Hallucination Is Worse When It Looks Like Evidence

- Users treat image-grounded answers as **observational**: the model saw it, so it must be true
- The model can sound certain about an object, person, or detail **absent** from the image
- The danger: it borrows the credibility of a photograph for a claim the pixels never supported

---

<!-- .slide: id="eval-safety" -->

## Safety Issues Specific to Multimodal

<div class="card-grid cols-3">
<div class="card"><h4>Privacy</h4><p>Faces, addresses, screens, documents, children, bystanders</p></div>
<div class="card"><h4>Identity and biometrics</h4><p>Face recognition, emotion inference, attribute guessing &mdash; stricter rules than object description</p></div>
<div class="card"><h4>Prompt injection</h4><p>Text inside an image or document tries to override the user</p></div>
<div class="card"><h4>Adversarial examples</h4><p>Small pixel changes flip the prediction</p></div>
<div class="card"><h4>Deepfakes</h4><p>Generation can manufacture persuasive false evidence</p></div>
<div class="card"><h4>Accessibility</h4><p>Descriptions for blind users need calibrated uncertainty, not fluent guesses</p></div>
</div>

---

<!-- .slide: id="eval-preprocessing" -->

## Failures Before the Weights

**Preprocessing** can cause the failure before the model runs at all.

<div class="card-grid cols-2">
<div class="card"><h4>Resizing</h4><p>Erases small text before the encoder ever sees it</p></div>
<div class="card"><h4>Cropping</h4><p>Removes the relevant object from the frame</p></div>
<div class="card"><h4>Frame sampling</h4><p>Misses the key event in a video</p></div>
<div class="card"><h4>Compression</h4><p>Distorts the details the answer depends on</p></div>
</div>

Product lesson: expose uncertainty when perceptual evidence is weak; never overclaim on **identity, medical, legal, or safety-critical** decisions. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="sq-photography-bias" -->

## Side Quest: Photography Bias

Photography is not a neutral sample of reality.

- **Which images exist** and **how they were taken** shape the model's view of the world
- Cameras and datasets over-represent some subjects, viewpoints, lighting conditions, and skin tones
- The model inherits those blind spots and fails hardest on what was rarely photographed

More images do not make a model more objective. They bake in the **bias of who and what gets photographed.** <!-- .element: class="text-lg" style="margin-top: 10px;" -->
