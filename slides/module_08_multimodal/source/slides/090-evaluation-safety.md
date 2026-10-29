:::divider id="divider-eval" title="Evaluation, Safety, and Failure Modes" sub="Measuring and attacking multimodal models"
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

## Visual Hallucination Follows the Language Prior

<div class="img-figure"><img src="images/pope_probe.svg" alt="POPE probing: a table with a plate, fork, and cup; asking about a random absent object is easy, a popular absent object is harder, and an absent object that co-occurs with the scene, like a knife, is hardest" style="max-height: 300px;"></div>

POPE (Li et al., 2023) asks yes/no questions about objects that are **not** in the image. Models say "yes" most often for objects that usually appear with what is there. The model is filling in the scene it expects, the same shortcut as the five-legged zebra. A caption that reads like an observation can still come from the prior. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="eval-safety" -->

## Safety Issues Specific to Multimodal

<div class="card-grid cols-3">
<div class="card"><h4>Privacy</h4><p>Faces, addresses, screens, documents, children, bystanders</p></div>
<div class="card"><h4>Identity and biometrics</h4><p>Face recognition, emotion inference, and attribute guessing carry stricter rules than object description</p></div>
<div class="card"><h4>Prompt injection</h4><p>Text inside an image or document tries to override the user</p></div>
<div class="card"><h4>Adversarial examples</h4><p>Small pixel changes flip the prediction</p></div>
<div class="card"><h4>Deepfakes</h4><p>Generation can manufacture persuasive false evidence</p></div>
<div class="card"><h4>Accessibility</h4><p>Descriptions for blind users need calibrated uncertainty, not fluent guesses</p></div>
</div>

---

<!-- .slide: id="eval-adversarial" -->

## An Adversarial Example

<div class="adv-row">
<figure><img src="images/adv_panda.png" alt="A photo of a panda"><figcaption>"panda"<br>57.7% confidence</figcaption></figure>
<div class="adv-op">+ 0.007 &times;</div>
<figure><img src="images/adv_noise.png" alt="Noise pattern: the sign of the gradient of the panda classification loss with respect to each pixel"><figcaption>sign of the gradient of<br>the "panda" loss</figcaption></figure>
<div class="adv-op">=</div>
<figure><img src="images/adv_gibbon.png" alt="The same panda photo, visually unchanged"><figcaption>"gibbon"<br>99.3% confidence</figcaption></figure>
</div>

$$\tilde{x} = x + \epsilon \cdot \mathrm{sign}\left(\nabla_x \mathcal{L}(\theta, x, y)\right)$$

$\mathcal{L}(\theta, x, y)$ is the classifier's cross-entropy loss for the correct label $y$ = panda. Its gradient with respect to the image $x$ tells each **pixel** which way to move to make "panda" less likely, so this is gradient descent run in reverse on the input. The change is invisible to people. The same attack works on a vision encoder inside a multimodal LLM, and there the target can be a whole sentence, such as an injected instruction. Goodfellow et al., 2015, <a href="https://arxiv.org/abs/1412.6572">arXiv:1412.6572</a> <!-- .element: class="text-lg" style="margin-top: 4px;" -->

---

<!-- .slide: id="eval-preprocessing" -->

## Preprocessing

Some failures happen before the model runs.

<div class="card-grid cols-2">
<div class="card"><h4>Resizing</h4><p>Erases small text before the encoder ever sees it</p></div>
<div class="card"><h4>Cropping</h4><p>Removes the relevant object from the frame</p></div>
<div class="card"><h4>Frame sampling</h4><p>Misses the key event in a video</p></div>
<div class="card"><h4>Compression</h4><p>Distorts the details the answer depends on</p></div>
</div>

A product built on these models should say when the visual evidence is weak, and should not overclaim on **identity, medical, legal, or safety-critical** decisions. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="sq-photography-bias" -->

## <span class="side-quest-badge">Side Quest</span> Photography Bias

Photography is not a neutral sample of reality.

- **Which images exist** and **how they were taken** shape the model's view of the world
- Cameras and datasets over-represent some subjects, viewpoints, lighting conditions, and skin tones
- The model inherits those blind spots and fails hardest on what was rarely photographed

Adding more images does not remove this bias. It adds more of the **same skew in who and what gets photographed.** <!-- .element: class="text-lg" style="margin-top: 10px;" -->
