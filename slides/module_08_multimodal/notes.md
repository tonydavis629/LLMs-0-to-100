# Module 8: Multimodal Models — Lecture Notes

These notes give an explanation and a citation for every major claim on the
slides, map the equations to the visuals they appear on, and record the
historical context. Module 8 keeps the Module 3–7 transformer machinery intact
and expands the **input space**: the language model must now condition on images,
audio, video, and sensor streams. The unifying question is how to turn non-text
data into tokens or embeddings that can live beside words.

## Review

- Module 7 left us with the modern **post-training stack**: pretraining gives
  broad capability (Module 5), supervised finetuning teaches instruction-following
  (Module 6), and RL or preference optimization shapes behavior with scalar
  feedback (Module 7, Ouyang et al., 2022, arXiv:2203.02155).
- The transformer machinery is unchanged from Modules 3 and 4: embeddings enter a
  sequence model, attention mixes information (Vaswani et al., 2017,
  arXiv:1706.03762), logits predict tokens, and gradients update weights. What
  changes in Module 8 is the **input space** — text tokens are no longer the only
  things the model can condition on.
- The central question for the module: how do we turn non-text data (images,
  audio, video, sensor streams) into tokens or embeddings that can live beside
  words?
- Callback to Module 7: multimodal models reuse the same SFT, preference-tuning,
  and RLHF machinery, but the reward and evaluation problems get harder because
  visual and audio mistakes can be subtle, spatial, temporal, or safety-critical.

## a. What "multimodal" means

### Modalities and the three tasks
- A **modality** is a type of signal with its own structure: text, image, audio,
  video, depth, thermal, radar, lidar, robot joint states, sensor readings, or
  code-execution traces. A **multimodal model** connects two or more modalities in
  one system.
- The slide splits multimodal work into three tasks:
  - **Understanding**: condition on an image, audio clip, video, or document and
    answer in text. This is the module's main object.
  - **Retrieval**: match images to captions, speech to transcripts, screenshots to
    descriptions, videos to queries (the CLIP setting, section c).
  - **Action**: perceive a state, decide, and call a tool, control a robot, or
    drive a UI.
- For this course the primary object is a **multimodal LLM**: a language model
  whose context can include non-text representations, usually by prepending or
  interleaving learned visual or audio embeddings with ordinary text tokens.

### The unifying abstraction and its two tensions
- **Every modality must be compressed into a sequence.** That sequence can be
  literal discrete tokens (text tokens, audio-codec tokens) or continuous
  embedding vectors (image patch features projected into the LLM's hidden size).
  This is the through-line that makes one transformer able to serve all modalities.
- **First tension — bandwidth.** Text is already symbolic and low-bandwidth;
  pixels and waveforms are continuous, high-bandwidth, and spatial or temporal. A
  1,000-token paragraph is small compared with a single high-resolution image or
  one minute of audio. This is quantified in section f's video token arithmetic.
- **Second tension — structure.** Language is naturally ordered left to right, but
  images have 2-D layout, audio has fine-grained time, and video has both.
  Flattening into a sequence is convenient for transformers but discards structure
  unless we add the right positional information (section b).
- The practical lesson for students: multimodal LLMs are not magic vision systems
  glued to chatbots. They **choose a representation, align it with language, then
  train the language model to use it** — the arc of sections b through e.

## b. Images are not words: visual tokenization

### The raw object and the CNN-to-transformer shift
- An image is a tensor, usually $H \times W \times C$ (height, width, channels),
  not a list of words. Neighboring pixels matter, 2-D position matters, and small
  translations usually should not change meaning.
- The older computer-vision path used **convolutional neural networks** (LeCun et
  al., 1998; Krizhevsky et al., AlexNet, 2012), which build local edge and texture
  features with sliding filters. The **Vision Transformer (ViT)** (Dosovitskiy et
  al., 2020, arXiv:2010.11929) changed the recipe by treating an image as a
  **sequence of patches** and applying the ordinary transformer encoder.

### ViT patch tokenization (the visual analog of embedding)
- Split the image into non-overlapping $P \times P$ patches, flatten each patch,
  and project it into a vector. The number of visual tokens is
  $$N = \frac{H}{P} \cdot \frac{W}{P}.$$
  This equation appears on the tokenization slide. A $224 \times 224$ image with
  $16 \times 16$ patches gives $N = 14 \cdot 14 = 196$ patch tokens.
- Reducing the patch size $P$ increases visual detail but increases attention cost
  **quadratically** in $N$, tying directly back to Module 4's attention-cost
  discussion (self-attention is $O(N^2)$).
- The **patch projection** is the visual analog of a token embedding lookup:
  $$z_i = W_{\text{patch}}\,\mathrm{flatten}(p_i) + b_{\text{patch}},$$
  where $p_i$ is one $P \times P \times C$ image patch and $z_i$ is the learned
  vector representing it. Where text does a discrete embedding lookup, vision does
  a **linear projection** of continuous pixels.
- **Positional embeddings are not optional.** Flattening patches into a 1-D
  sequence discards layout; the model needs row and column position information to
  recover which patch was where. (This answers the module's first quiz question.)

### Why resolution and the "visual token" ambiguity matter
- Resolution matters because many visual tasks depend on fine detail: small text
  in screenshots, tick labels on charts, tiny objects, facial expressions, medical
  findings. A model can fail simply because the relevant pixels were never
  represented at enough resolution — a **preprocessing** failure, not a reasoning
  failure (revisited in sections e and g).
- **Vocabulary mismatch.** Text tokenizers have discrete vocabularies; vision
  encoders usually produce continuous features. "Visual token" is overloaded: it
  may mean a patch embedding, a learned-query output, a region feature, or a
  discrete codebook token (VQ-VAE, van den Oord et al., 2017, arXiv:1711.00937).
  Students should always ask which one is meant. This distinction is not its own
  slide, but it is tested by the module's second quiz question.
- Key distinction carried forward: an image encoder can produce useful visual
  features **without being a language model**. A multimodal LLM still needs a
  **bridge** that maps those features into the LLM's representational space
  (section d).

## c. Contrastive image-text pretraining and CLIP

### Two encoders, one shared space
- **CLIP** (Radford et al., 2021, arXiv:2103.00020) is the cleanest first example
  because it does **not** generate language — it only learns a shared embedding
  space for images and text. The architecture has two encoders: an **image
  encoder** maps an image to a vector, and a **text encoder** maps a caption or
  prompt to a vector. Training pulls matched image-text pairs together and pushes
  mismatched pairs apart.
- The same idea at even larger, noisier scale is **ALIGN** (Jia et al., 2021,
  arXiv:2102.05918), which used a billion-scale noisy alt-text corpus, reinforcing
  the lesson that scale beats curation for contrastive alignment.

### The contrastive objective (the slide's similarity matrix)
- For a batch of $B$ image-caption pairs, compute a $B \times B$ **similarity
  matrix**
  $$s_{ij} = \frac{\langle \hat{v}_i, \hat{t}_j \rangle}{\tau},$$
  where $\hat{v}_i$ and $\hat{t}_j$ are **L2-normalized** image and text embeddings
  (so the inner product is a cosine similarity) and $\tau$ is a learned or fixed
  **temperature** that scales the logits. This matrix is drawn on the CLIP slide
  and is exactly the heatmap the exercise produces.
- The loss is a **symmetric** pair of cross-entropies — image-to-text and
  text-to-image retrieval:
  $$\mathcal{L} = \tfrac{1}{2}\Big(
  -\tfrac{1}{B}\sum_{i}\log\frac{e^{s_{ii}}}{\sum_j e^{s_{ij}}}
  -\tfrac{1}{B}\sum_{j}\log\frac{e^{s_{jj}}}{\sum_i e^{s_{ij}}}\Big).$$
  For row $i$ the correct class is caption $i$; for column $j$ the correct class is
  image $j$. The diagonal entries $s_{ii}$ are the matched pairs. This is the same
  cross-entropy from Modules 2 and 5, but the **classes are the other examples in
  the batch** rather than vocabulary tokens (answering the module's third quiz
  question).

### Why natural-language supervision works, and zero-shot classification
- Captions and alt text are noisy, but they are **abundant**. Natural-language
  supervision lets the image encoder learn categories, attributes, styles, and
  concepts without a fixed hand-labeled class list — the central argument of the
  CLIP paper.
- **Zero-shot classification** falls out for free: write one text prompt per class
  ("a photo of a {class}"), embed each prompt, embed the image, and choose the
  prompt with highest similarity. No classifier head was trained for that specific
  label set (answering the fourth quiz question).
- **Limitation, stated as clearly as the success.** CLIP aligns **global** image
  and text embeddings; it does not by itself produce detailed answers, follow
  instructions, count reliably, localize objects precisely, or reason over multiple
  images. It is an alignment foundation, not a full visual assistant — which is
  exactly why section d adds a language model. This is also the motivation for the
  exercise: students implement a miniature CLIP objective on synthetic
  image-caption pairs and watch retrieval accuracy rise.

## d. Connecting vision encoders to LLMs

### The stitched (cascaded) recipe
- A visual assistant needs more than CLIP retrieval: it needs the language model to
  **condition on visual information while generating text autoregressively**. The
  stitched recipe has three pieces:
  - a **vision encoder** (often pretrained separately, e.g. a CLIP ViT) that turns
    the image into patch features or pooled features;
  - a **connector** that maps vision features into the LLM hidden size;
  - a **language model** that consumes those visual embeddings plus text tokens and
    predicts the next text token.
- The simplest connector is a **linear projector**:
  $$u_i = W_{\text{proj}}\, h_i + b_{\text{proj}},$$
  where $h_i$ is a visual feature and $u_i$ has the same width as the LLM's token
  embeddings. This is the connector students implement in the exercise.
- Once projected, visual embeddings are **prepended as prefix tokens**:
  $$[\,u_1, u_2, \dots, u_k,\; e_{\text{text}_1}, e_{\text{text}_2}, \dots\,].$$
  The LLM does not need to know whether the first vectors came from words or
  pixels; attention just mixes vectors. This visual-prefix sequence is drawn on the
  connector slide and is the sequence the exercise builds.
- The training objective stays **causal next-token prediction over the text
  response**:
  $$\mathcal{L} = -\frac{1}{|R|}\sum_{t \in R}\log p_\theta\big(x_t \mid \text{image}, x_{<t}\big),$$
  where $R$ is the set of response-token positions. This is Module 6's masked SFT
  loss with an image condition added — captioning is still next-token prediction
  (answering the sixth quiz question).

### Architectural variants
- **LLaVA-style projector** (Liu et al., 2023, arXiv:2304.08485): frozen or
  mostly-frozen vision encoder, project its features into the LLM, then
  instruction-tune on image-dialogue data. Simplest and the exercise's template.
- **BLIP-2-style Q-Former** (Li et al., 2023, arXiv:2301.12597): a small set of
  learned **query tokens** extract the most useful visual information from a frozen
  image encoder before handing it to a frozen LLM. InstructBLIP (Dai et al., 2023,
  arXiv:2305.06500) adds instruction tuning on top.
- **Flamingo-style cross-attention** (Alayrac et al., 2022, arXiv:2204.14198): keep
  the language model mostly intact and insert **gated cross-attention** layers that
  attend to visual features, enabling few-shot multimodal learning.
- **Early-fusion models**: tokenize multiple modalities into one shared sequence
  and train a single transformer over all of them (Chameleon, Chameleon Team, 2024,
  arXiv:2405.09818).

### Stitched versus native — a 2024–2025 architecture shift
- **Stitched systems** bolt together strong pretrained parts (vision/audio encoder
  + projector or cross-attention bridge + language model). They are modular,
  cheaper, and easy to debug, but the LLM did not learn the modality from the
  beginning.
- **Natively multimodal systems** train the modality interface and the core model
  together, often end-to-end across text, image, audio, and video. **GPT-4o**
  (OpenAI, 2024, "Hello GPT-4o") is the canonical product example: the old voice
  stack chained speech-to-text, a text model, and text-to-speech, whereas GPT-4o
  was trained as **one model** across text, vision, and audio. **Gemini 1.5**
  (Gemini Team, 2024, arXiv:2403.05530) is natively multimodal with very long
  context.
- **Early fusion vs native are related but not identical.** Early fusion describes
  the **input representation** (modalities placed in one shared sequence early);
  native multimodality describes the **training story** (the model learns those
  modalities together rather than inheriting a frozen text-only core). Do not
  oversimplify "native" as "raw pixels straight into attention" — real systems
  still use learned front ends (patching, convolutional stems, spectrograms, audio
  codecs). The difference is whether those front ends are trained as part of one
  multimodal model.

### The design tradeoff and why the projector is not trivial
- Freezing large pretrained encoders is cheaper and more stable; fully or jointly
  tuning more of the stack can improve alignment at higher compute and data cost.
- The projector is small but not trivial: it learns a **coordinate transformation**
  between the vision encoder's feature space and the LLM's language space. A bad
  projector makes a strong vision encoder useless to the LLM (answering the fifth
  quiz question). In the exercise, students project a tiny image embedding into
  NanoGPT's hidden size and train it to caption or answer from that image prefix.

## e. Multimodal instruction tuning and grounding

### From alignment to assistant behavior
- Pretraining an image-text embedding space gives **alignment** but not
  **assistant behavior**. Multimodal SFT teaches the model to respond to
  image-conditioned instructions in the expected conversational format — the
  visual analog of Module 6's instruction tuning, popularized by LLaVA (Liu et al.,
  2023, arXiv:2304.08485).
- Before instruction tuning, modern multimodal pretraining increasingly uses
  **interleaved image-text documents**: web pages, tutorials, papers, manuals,
  screenshots with surrounding prose, documents where multiple images appear in
  sequence with text between them. Flamingo (Alayrac et al., 2022) was an early
  demonstration that interleaved data enables in-context multimodal learning.
- Interleaved data matters because single caption pairs teach **global matching**,
  while documents teach **reference and context**: "the second diagram," "this
  chart," "the example above," "compare these two panels." That is the data shape
  needed for reasoning across multiple images.

### The instruction-tuning data unit and chat templates
- After broad alignment, the data unit becomes an **image plus a prompt-response
  pair**: "What color is the triangle?", "Read the axis label", "Describe the error
  message in this screenshot", "Which object is left of the chair?"
- Chat templates gain **placeholders for non-text inputs**, e.g. `<image>` in the
  prompt. The placeholder is not the image itself — the actual image features are
  inserted by the model wrapper (they are the projected $u_i$ vectors of section d).
- Common data sources: image captions and alt text (broad alignment); interleaved
  web documents, PDFs, notebooks, manuals, tutorials (multi-image context); visual
  question answering datasets (targeted answers); OCR and document datasets
  (reading text in images); referring-expression and grounding datasets (locating
  objects by language); and **synthetic image-dialogue data** generated by a
  stronger model then filtered and finetuned — the multimodal echo of Module 6's
  synthetic instruction data, and the method LLaVA introduced.

### Grounding, and its characteristic failure modes
- **Grounding** means connecting words to the right parts of the perceptual input:
  "the red square" should refer to the red square, not merely to the fact that red
  squares are common in the dataset.
- **Spatial language** is a good classroom stress test — left/right, above/below,
  inside/outside, count, same/different, nearest/farthest — easy for humans but
  exposing whether the model really uses visual layout.
- **OCR is a separate failure mode**: a model may recognize objects yet fail to
  read small text, or read text but misunderstand the layout of a form, chart, or
  UI (see the "OCR trap" side quest and section g).
- **Visual hallucination** is the multimodal analog of text hallucination: the
  model confidently describes objects, text, colors, or relationships that are not
  present in the input.
- **Shortcut / language-prior behavior.** If training answers carry strong language
  priors, the model may answer from text patterns instead of from the image.
  Diagnostic: zero out or swap the image and see whether the answer changes — if it
  does not, the model is not using the image (the exercise's image-ablation extra
  credit, answering the seventh quiz question). Controlled synthetic data lets
  students test whether the image condition is actually used.

## f. Beyond still images: audio, speech, video, and sensors

### Audio and speech
- Audio begins as a **waveform**: a high-frequency sequence of amplitude values.
  Common representations are **spectrograms** (time-frequency images), **learned
  audio embeddings**, or **discrete codec tokens**.
- Speech systems come in several framings:
  - **Automatic speech recognition (ASR)** maps audio to text. **Whisper** (Radford
    et al., 2022, arXiv:2212.04356) showed robust ASR trained on large-scale weak
    supervision (680k hours), the audio analog of CLIP's "scale beats curation."
  - **Text-to-speech (TTS)** maps text to audio.
  - **Speech-language models** condition on speech and respond in text or speech;
    **AudioPaLM** (Rubenstein et al., 2023, arXiv:2306.12925) unifies speech
    understanding and generation inside a single LLM by treating audio as tokens.
  - **Fully spoken assistants** must handle both linguistic content and
    paralinguistic cues (timing, tone, speaker turns) — the GPT-4o voice case.

### Video and the token-budget arithmetic
- Video adds a **time axis** to images. A model can sample frames (often sparsely),
  encode each frame with an image encoder, then add **temporal positional
  information** before feeding features to the LLM (Video-LLaVA, Lin et al., 2023,
  arXiv:2311.10122).
- Video is expensive because token counts grow with frames, resolution, and patch
  density. The slide makes this concrete. A 10-second clip at 30 fps has
  $10 \times 30 = 300$ frames. If each $224 \times 224$ frame is split into
  $16 \times 16$ patches, each frame has $196$ patch tokens (from
  $N = \tfrac{224}{16}\cdot\tfrac{224}{16} = 196$), so the raw visual sequence is
  $$300 \times 196 = 58{,}800 \text{ visual tokens}.$$
  That already exceeds half of a $100\text{k}$ context window before the prompt,
  answer, audio, metadata, or any extra visual query tokens.
- At $448 \times 448$ with the same $16 \times 16$ patch size, each frame has
  $\tfrac{448}{16}\cdot\tfrac{448}{16} = 784$ patch tokens, so the same clip becomes
  $$300 \times 784 = 235{,}200 \text{ visual tokens},$$
  which no longer fits in $100\text{k}$. This is why video models **sample frames,
  pool patches, use learned query tokens, compress with temporal encoders, or
  retrieve only relevant segments**. The issue is not just context length —
  attention cost scales roughly with the **square** of the sequence length (Module
  4), so doubling resolution per side quadruples tokens and roughly $16\times$ the
  attention cost. Gemini 1.5 (arXiv:2403.05530) is the notable counterpoint, pushing
  context to millions of tokens to hold long video directly.

### Temporal reasoning, stream alignment, and other sensors
- The hard part of video is not seeing objects frame by frame; it is **temporal
  reasoning**: what changed, which event caused another, what happened before the
  person opened the door, whether an action completed.
- **Audio-video models must align streams with different rates**: a video frame may
  arrive every 33 ms while audio samples arrive thousands of times per second.
- **Sensor modalities generalize the same template**: depth, thermal, radar/RF,
  lidar, robot joint states, medical images, or tabular time series can all be
  encoded and aligned to language. **ImageBind** (Girdhar et al., 2023,
  arXiv:2305.05665) binds six modalities into one embedding space using image-paired
  data as the hub. The representation changes; the **bridge problem stays the same**
  — this is the section's main point.

> Note: an earlier draft of this module included a short generation section
> (diffusion, unified autoregressive token models, DALL-E, the LLM-as-controller
> pattern). It was cut to keep the module focused on multimodal *understanding*
> LLMs rather than image generation, so there is no section g. Sections h and i were
> renumbered to g and h.

## g. Evaluation, safety, and failure modes

### Why multimodal evaluation is harder
- Ground truth often depends on visual detail, spatial relations, OCR, temporal
  events, or subjective judgment, so evaluation is harder than text evaluation.
- Useful benchmark categories on the slide:
  - **VQA and captioning** for image-language understanding.
  - **OCR and document understanding** for reading text and layout.
  - **Chart and diagram reasoning** for connecting marks, axes, legends, quantities
    (e.g. **MathVista**, Lu et al., 2023, arXiv:2310.02255, visual mathematical
    reasoning).
  - **Multidisciplinary benchmarks** such as **MMMU** (Yue et al., 2023,
    arXiv:2311.16502), college-level visual reasoning across many disciplines.
  - **Video question answering** for temporal understanding.
- **Exact-match scoring is often too brittle**: "two" and "there are two" are
  equivalent, and a model can be semantically wrong while matching a keyword.
  Module 11's evaluation tooling returns here (answering the eighth quiz question in
  spirit: OCR/chart failures are often perception, not reasoning).

### Hallucination and multimodal-specific safety
- **Hallucination** is especially damaging in visual settings because users treat
  image-grounded answers as **observational evidence**; the model can sound certain
  about an absent object, person, or detail.
- Safety issues specific to multimodal systems (the slide's list):
  - **Privacy**: images and audio can contain faces, addresses, screens, documents,
    children, or bystanders.
  - **Identity and biometrics**: face recognition, emotion inference, and sensitive
    attribute guessing need stricter rules than ordinary object description (see the
    GPT-4o System Card, OpenAI, 2024).
  - **Prompt injection**: an image or document can contain text that tries to
    override the user's instruction or exfiltrate data. The safe design treats
    in-image text as untrusted **input**, never as a command — a boundary Module 9
    revisits.
  - **Adversarial examples**: small visual changes can flip a prediction.
  - **Deepfakes and synthetic media**: generators can create persuasive false
    evidence.
  - **Accessibility**: models describing images for blind users need **calibrated
    uncertainty**, not fluent guesses.
- **Preprocessing failures, not just weights**: resizing can erase small text,
  cropping can remove the relevant object, frame sampling can miss the key event,
  and compression can distort details — connecting back to section b's resolution
  point and the "OCR trap."
- **Product lesson**: the model should expose uncertainty when perceptual evidence
  is weak and avoid overclaiming about identity, medical diagnosis, legal
  interpretation, or safety-critical decisions (the "photography bias" side quest).

## h. The modern multimodal stack, and the handoff to Module 9

- **Name the full stack**: modality-specific encoders, a connector or shared
  tokenizer, a language model, multimodal instruction tuning, preference or RL
  post-training, retrieval and tool systems, and deployment infrastructure. Every
  earlier section is one layer of this stack.
- **The course through-line is intact.** Module 5 gave next-token pretraining,
  Module 6 taught response format, Module 7 shaped behavior with feedback, and
  Module 8 adds **perceptual conditions** to the context. The architectural slogan:
  **same transformer, larger interface.** The model still moves vectors through
  attention layers; the new work is deciding which vectors represent the non-text
  input.
- **The deployment implication that points to Module 9**: multimodal inputs are
  expensive. Vision encoders add latency, images add preprocessing, visual tokens
  increase context length (section f's arithmetic), and video or audio can overwhelm
  memory bandwidth.
- **Handoff to Module 9**: now that the model can read more than text, serving it
  efficiently becomes a systems problem — batching, KV-cache size, image
  preprocessing, memory bandwidth, quantization, and model routing become the
  practical constraints.

## Notable Figures

- **Alexey Dosovitskiy and the ViT team (Google Research)** — introduced the
  Vision Transformer framing of images as patch sequences ("An Image is Worth 16x16
  Words," 2020, arXiv:2010.11929), the representation the whole module builds on.
  Introduced during section b.
- **Alec Radford and the CLIP team (OpenAI)** — showed that large-scale
  natural-language supervision aligns images and text in a shared embedding space
  ("Learning Transferable Visual Models From Natural Language Supervision," 2021,
  arXiv:2103.00020). Radford also led **Whisper** (2022, arXiv:2212.04356), the
  audio analog. Introduced during section c, returns in section f.
- **Jean-Baptiste Alayrac and the Flamingo team (DeepMind)** — connected frozen
  language models to visual inputs with gated cross-attention and few-shot
  multimodal learning (2022, arXiv:2204.14198). Mentioned during sections d and e.
- **Junnan Li and collaborators** — introduced BLIP-2 and the **Q-Former** bridge
  between a frozen image encoder and a frozen LLM (2023, arXiv:2301.12597).
  Mentioned during section d.
- **Haotian Liu, Chunyuan Li, Yuheng Li, Yong Jae Lee, and the LLaVA
  collaborators** — popularized **visual instruction tuning** with a simple
  projector from vision features into an LLM (2023, arXiv:2304.08485), the exercise's
  direct template. Introduced during sections d and e.

On the slides, the Radford (CLIP) and Radford (Whisper) figures use the photo-reveal
format; Dosovitskiy, Alayrac/Li, and Liu are shown as text contributor cards.

## Side Quests

These are genuine asides — interesting but not on the critical path.

- **Visual registers: why does a ViT need extra tokens?** (near section b) — Some
  vision transformers develop **high-norm patch tokens** that act like accidental
  scratch space rather than clean local features. **Register tokens** (Darcet et
  al., "Vision Transformers Need Registers," 2023, arXiv:2309.16588) give the model
  dedicated extra slots for global information, improving dense visual features and
  making attention maps easier to interpret. The lesson: "attention over patches"
  does not mean every patch token has a clean human-readable meaning.
- **Why captions are weak supervision** (near section c) — A caption rarely names
  every object, relation, style, and spatial detail in an image. Contrastive
  learning succeeds anyway because **scale and diversity** create useful pressure
  (CLIP; ALIGN, arXiv:2102.05918), but the model can learn **caption bias** and miss
  uncaptioned details.
- **The OCR trap** (near section e or g) — Show a screenshot where the answer
  depends on a tiny word or axis label, then ask whether the failure is a
  **reasoning** failure, a **perception** failure, a **resolution** failure, or an
  **OCR** failure. Often it is preprocessing (section b), not reasoning.
- **Photography bias** (near section g) — A model's view of the world is shaped by
  which images exist and how they were taken. Cameras and datasets over-represent
  some subjects, viewpoints, lighting conditions, and skin tones and under-represent
  others, so a model inherits those blind spots and fails hardest on rarely
  photographed cases. Seeing more images does not make a model more objective; it
  makes the model reflect the bias in who and what gets photographed. This replaces
  the earlier "the camera is not ground truth" framing.
- **One model or a system of models?** (near section h) — Compare a **unified
  mixed-token model** (Chameleon, arXiv:2405.09818) against a **routed product
  stack** with a language model, image encoder, OCR model, retriever, and image
  generator. The cleanest research objective is often not the cheapest or most
  reliable product architecture.

## References

- Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image
  Recognition at Scale" (ViT), 2020. <https://arxiv.org/abs/2010.11929>
- Radford et al., "Learning Transferable Visual Models From Natural Language
  Supervision" (CLIP), 2021. <https://arxiv.org/abs/2103.00020>
- Jia et al., "Scaling Up Visual and Vision-Language Representation Learning With
  Noisy Text Supervision" (ALIGN), 2021. <https://arxiv.org/abs/2102.05918>
- Alayrac et al., "Flamingo: a Visual Language Model for Few-Shot Learning," 2022.
  <https://arxiv.org/abs/2204.14198>
- Li et al., "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image
  Encoders and Large Language Models," 2023. <https://arxiv.org/abs/2301.12597>
- Liu et al., "Visual Instruction Tuning" (LLaVA), 2023.
  <https://arxiv.org/abs/2304.08485>
- Dai et al., "InstructBLIP: Towards General-purpose Vision-Language Models with
  Instruction Tuning," 2023. <https://arxiv.org/abs/2305.06500>
- Girdhar et al., "ImageBind: One Embedding Space To Bind Them All," 2023.
  <https://arxiv.org/abs/2305.05665>
- Darcet et al., "Vision Transformers Need Registers," 2023.
  <https://arxiv.org/abs/2309.16588>
- Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision"
  (Whisper), 2022. <https://arxiv.org/abs/2212.04356>
- Rubenstein et al., "AudioPaLM: A Large Language Model That Can Speak and Listen,"
  2023. <https://arxiv.org/abs/2306.12925>
- Lin et al., "Video-LLaVA: Learning United Visual Representation by Alignment
  Before Projection," 2023. <https://arxiv.org/abs/2311.10122>
- Chameleon Team, "Chameleon: Mixed-Modal Early-Fusion Foundation Models," 2024.
  <https://arxiv.org/abs/2405.09818>
- OpenAI, "Hello GPT-4o" (single end-to-end model across text, vision, and audio),
  2024. <https://openai.com/index/hello-gpt-4o/>
- OpenAI, "GPT-4o System Card" (multimodal safety and limitations), 2024.
  <https://openai.com/index/gpt-4o-system-card/>
- Gemini Team, "Gemini 1.5: Unlocking multimodal understanding across millions of
  tokens of context," 2024. <https://arxiv.org/abs/2403.05530>
- Yue et al., "MMMU: A Massive Multi-discipline Multimodal Understanding and
  Reasoning Benchmark for Expert AGI," 2023. <https://arxiv.org/abs/2311.16502>
- Lu et al., "MathVista: Evaluating Mathematical Reasoning of Foundation Models in
  Visual Contexts," 2023. <https://arxiv.org/abs/2310.02255>
- van den Oord et al., "Neural Discrete Representation Learning" (VQ-VAE), 2017.
  <https://arxiv.org/abs/1711.00937>
- Vaswani et al., "Attention Is All You Need," 2017.
  <https://arxiv.org/abs/1706.03762>
- Ouyang et al., "Training Language Models to Follow Instructions with Human
  Feedback" (InstructGPT), 2022. <https://arxiv.org/abs/2203.02155>
