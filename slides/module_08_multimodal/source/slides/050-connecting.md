:::divider id="divider-bridge" title="Connecting Vision to Language" sub="The bridge"
:::

---

<!-- .slide: id="bridge-recipe" -->

## Encoder, Connector, Language Model

To answer a question about an image, the language model has to read the image while it generates text. The usual recipe joins three parts:

<div class="img-figure"><img src="images/arch_generic.svg" alt="The carried diagram filled in: image to vision encoder to connector to language model, alongside the text prompt lane" style="max-height: 340px;"></div>

The vision encoder is usually **pretrained separately** (like CLIP). The new work is the connector and how the LLM reads its output. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="bridge-projector" -->

## The Simplest Connector Is a Linear Projector

Map each visual feature $h_i$ into the LLM's token-embedding width:

$$u_i = W_{\text{proj}}\ h_i + b_{\text{proj}}$$

Then **prepend** the projected vectors ahead of the text:

<div class="diagram"><svg viewBox="0 0 920 175" width="900" role="img" aria-label="visual feature projected by W_proj into an LLM-width vector, then prepended in front of the text tokens">
<g text-anchor="middle"><rect x="20" y="60" width="66" height="60" rx="6" fill="#1b2436" stroke="#4a9eff" stroke-width="2"/><text x="53" y="86" fill="#e8eaf0" font-size="14">h_i</text><text x="53" y="103" fill="#8fa0bd" font-size="10">vision space</text>
<rect x="150" y="55" width="86" height="70" rx="6" fill="#1b2436" stroke="#50c878" stroke-width="2"/><text x="193" y="85" fill="#e8eaf0" font-size="14">W_proj</text><text x="193" y="103" fill="#8fa0bd" font-size="10">linear</text>
<rect x="300" y="60" width="66" height="60" rx="6" fill="#1b2436" stroke="#a06bd4" stroke-width="2"/><text x="333" y="86" fill="#e8eaf0" font-size="14">u_i</text><text x="333" y="103" fill="#8fa0bd" font-size="10">LLM width</text></g>
<g stroke="#8fa0bd" stroke-width="2" fill="none" marker-end="url(#ahp)"><line x1="86" y1="90" x2="147" y2="90"/><line x1="236" y1="90" x2="297" y2="90"/><line x1="366" y1="90" x2="432" y2="90"/></g>
<text x="399" y="80" fill="#c7d0e0" font-size="10.5" text-anchor="middle">prepend</text>
<g text-anchor="middle" font-size="13"><rect x="440" y="62" width="52" height="56" rx="5" fill="#12331f" stroke="#50c878" stroke-width="2"/><text x="466" y="94" fill="#e8eaf0">u_1</text><rect x="497" y="62" width="52" height="56" rx="5" fill="#12331f" stroke="#50c878" stroke-width="2"/><text x="523" y="94" fill="#e8eaf0">u_k</text><rect x="566" y="62" width="52" height="56" rx="5" fill="#1b2436" stroke="#4a9eff" stroke-width="2"/><text x="592" y="94" fill="#e8eaf0">e_1</text><rect x="623" y="62" width="52" height="56" rx="5" fill="#1b2436" stroke="#4a9eff" stroke-width="2"/><text x="649" y="94" fill="#e8eaf0">e_2</text><rect x="680" y="62" width="52" height="56" rx="5" fill="#1b2436" stroke="#4a9eff" stroke-width="2"/><text x="706" y="94" fill="#e8eaf0">e_3</text></g>
<g text-anchor="middle" font-size="11"><text x="529" y="52" fill="#50c878">visual prefix</text><text x="649" y="52" fill="#4a9eff">text tokens</text></g>
<defs><marker id="ahp" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 Z" fill="#8fa0bd"/></marker></defs>
</svg></div>

Nothing tells the LLM whether a vector came from a word or a patch. Attention treats the visual vectors like any other earlier positions in the context. <!-- .element: class="text-lg" style="margin-top: 10px;" -->

---

<!-- .slide: id="bridge-loss" -->

## The Loss Is Still Next-Token Prediction

Training is **causal next-token prediction over the text response**, now conditioned on the image:

$$\mathcal{L} = -\frac{1}{|R|}\sum_{t \in R}\log p_\theta\left(x_t \mid \textcolor{#f5a623}{\text{image}}, x_{<t}\right)$$

- **Module 6's masked SFT loss**: score only the response tokens $R$
- New: an image condition in the context
- The transformer is unchanged; we added vectors to the front of the sequence and kept the same objective

---

<!-- .slide: id="bridge-variants" -->

## Four Ways to Build the Bridge

<div class="taxonomy-table">
<table>
<thead><tr><th>Approach</th><th>Idea</th><th>Who</th></tr></thead>
<tbody>
<tr><td><strong>LLaVA projector</strong></td><td>Frozen vision encoder, project features into the LLM, instruction-tune on image dialogue</td><td>Liu et al.</td></tr>
<tr><td><strong>BLIP-2 Q-Former</strong></td><td>Learn a small set of query tokens that <em>extract</em> the useful visual information first</td><td>Li et al.</td></tr>
<tr><td><strong>Flamingo cross-attn</strong></td><td>Keep the LLM mostly intact; insert cross-attention layers that attend to visual features</td><td>Alayrac et al.</td></tr>
<tr><td><strong>Early fusion</strong></td><td>Tokenize every modality into one shared sequence; train a single transformer over all of it</td><td>Chameleon, others</td></tr>
</tbody>
</table>
</div>

They differ in **how much of the stack is trained** and **where the visual information enters**. All four still train a transformer with next-token prediction. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="bridge-llava" -->

## LLaVA: Project Every Patch Into the Prompt

<div class="img-figure"><img src="images/arch_llava.svg" alt="LLaVA in the carried diagram: frozen CLIP ViT-L/14, trained two-layer MLP projector, trained Vicuna LLM; 576 visual tokens enter the input sequence" style="max-height: 330px;"></div>

- All 576 patch features become prefix tokens, so the LLM attends to every patch directly
- Simple and strong, but the visual prefix is long, and it grows with resolution

Liu et al., "Visual Instruction Tuning" (2023), <a href="https://arxiv.org/abs/2304.08485">arXiv:2304.08485</a>; LLaVA-1.5, <a href="https://arxiv.org/abs/2310.03744">arXiv:2310.03744</a> <!-- .element: class="text-lg" style="margin-top: 6px;" -->

---

<!-- .slide: id="bridge-blip2" -->

## BLIP-2: Compress Before the LLM Sees It

<div class="img-figure"><img src="images/arch_blip2.svg" alt="BLIP-2 in the carried diagram: frozen EVA-CLIP encoder, trained Q-Former with 32 learned queries, frozen LLM; only 32 visual tokens enter" style="max-height: 330px;"></div>

- 32 learned queries cross-attend to the patch features and pull out what matters for language
- Both large models stay frozen; only the Q-Former and a linear layer train

Li et al., "BLIP-2" (Salesforce, 2023), <a href="https://arxiv.org/abs/2301.12597">arXiv:2301.12597</a> <!-- .element: class="text-lg" style="margin-top: 6px;" -->

---

<!-- .slide: id="bridge-flamingo" -->

## Flamingo: Visual Tokens as Keys and Values

<div class="img-figure"><img src="images/arch_flamingo.svg" alt="Flamingo in the carried diagram: frozen NFNet encoder, trained Perceiver Resampler, frozen Chinchilla LLM with trained gated cross-attention layers inserted between its blocks" style="max-height: 330px;"></div>

- The image never joins the text sequence; new cross-attention layers read it at several depths
- Each new layer starts gated to zero, so on day one the frozen LLM behaves exactly as before

Alayrac et al., "Flamingo" (DeepMind, 2022), <a href="https://arxiv.org/abs/2204.14198">arXiv:2204.14198</a> <!-- .element: class="text-lg" style="margin-top: 6px;" -->

---

<!-- .slide: id="bridge-fusion" -->

## Early Fusion: Image Ids in the Vocabulary

<div class="img-figure"><img src="images/arch_fusion.svg" alt="Early fusion in the carried diagram: a VQ image tokenizer turns the image into 1,024 discrete ids that go into a shared embedding table; no connector, because the ids go straight into the embedding table; one transformer trained from scratch" style="max-height: 330px;"></div>

- A VQ tokenizer (the VQ-VAE section) turns the image into codebook ids
- The ids are added to the vocabulary and looked up in the same embedding table as text ids, so there is nothing for a connector to do
- One transformer trains on the mixed sequences and can **output** image tokens too

Chameleon Team, "Chameleon: Mixed-Modal Early-Fusion Foundation Models" (Meta, 2024), <a href="https://arxiv.org/abs/2405.09818">arXiv:2405.09818</a> <!-- .element: class="text-lg" style="margin-top: 6px;" -->

---

<!-- .slide: id="bridge-stitched-native" -->

## Stitched vs Native

A main architecture shift of 2024&ndash;2025.

<div class="compare-table">
<table>
<thead><tr><th>Stitched (cascaded)</th><th>Natively multimodal</th></tr></thead>
<tbody>
<tr><td>Bolt together strong pretrained parts: an encoder, a projector or cross-attention bridge, and an LLM</td><td>Train the modality interface and the core model <strong>together</strong>, often end to end across text, image, audio, video</td></tr>
<tr><td>Modular, cheaper, easy to debug</td><td>The old voice stack was speech&rarr;text, a text model, then text&rarr;speech; <strong>GPT-4o</strong> was trained as one model across text, vision, audio</td></tr>
<tr><td>The LLM never learned every modality from the start</td><td>The model learns the modalities together rather than inheriting a frozen text-only core</td></tr>
</tbody>
</table>
</div>

"Native" does not mean raw pixels into attention. Real systems still use learned front ends (patching, conv stems, spectrograms, audio codecs). The difference: those front ends are **trained as part of one model**. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="bridge-projector-matters" -->

## A Small Projector Still Matters

:::columns cols="2" gap="34px"
**It is tiny**

- Often a single linear layer or a two-layer MLP
- About 5M to 30M parameters in LLaVA, against a 7B or 13B LLM
+++
**It does the translation**

- Its weights are **trained** to map vision-feature space into the LLM's embedding space, a learned change of coordinates
- If that mapping is poor, the LLM cannot use even a strong vision encoder
:::

Freezing the large encoder is cheaper and more stable. Tuning more of the stack can improve alignment but costs more. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

The exercise builds this bridge: project a tiny image embedding into NanoGPT's hidden size, train it to caption. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="bridge-train-stages" -->

## Multimodal Training Adds a Stage

<div class="img-figure"><img src="images/train_stages.svg" alt="Four training stages: pretrained encoder and LLM; modality alignment where only the projector trains on image-caption pairs; multimodal SFT where projector and LLM train on image conversations; preference tuning" style="max-height: 370px;"></div>

The text model already went through pretraining, SFT, and RL. Before it can follow image instructions, a new **alignment** stage teaches it to read the encoder's output at all. Then comes assistant behavior. <!-- .element: class="text-lg" -->
