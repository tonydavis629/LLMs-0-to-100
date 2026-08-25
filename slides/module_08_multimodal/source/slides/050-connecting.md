:::divider id="divider-bridge" title="Connecting Vision to Language" sub="The bridge"
:::

---

<!-- .slide: id="bridge-recipe" -->

## The Stitched Recipe: Three Pieces

The language model must **condition on an image while generating text autoregressively**. The cascaded recipe bolts together three parts:

<div class="diagram"><svg viewBox="0 0 1000 210" width="1000" role="img" aria-label="pipeline: image to vision encoder to connector to language model to text, with each arrow labeled by what flows">
<g text-anchor="middle">
<rect x="15" y="80" width="110" height="70" rx="8" fill="#1b2436" stroke="#4a9eff" stroke-width="2"/><text x="70" y="110" fill="#e8eaf0" font-size="14">Image</text><text x="70" y="130" fill="#8fa0bd" font-size="11">H x W x C</text>
<rect x="200" y="80" width="150" height="70" rx="8" fill="#1b2436" stroke="#f5a623" stroke-width="2"/><text x="275" y="110" fill="#e8eaf0" font-size="14">Vision encoder</text><text x="275" y="130" fill="#8fa0bd" font-size="11">pixels to features</text>
<rect x="425" y="80" width="150" height="70" rx="8" fill="#1b2436" stroke="#50c878" stroke-width="2"/><text x="500" y="110" fill="#e8eaf0" font-size="14">Connector</text><text x="500" y="130" fill="#8fa0bd" font-size="11">features to LLM width</text>
<rect x="650" y="80" width="160" height="70" rx="8" fill="#1b2436" stroke="#4a9eff" stroke-width="2"/><text x="730" y="106" fill="#e8eaf0" font-size="14">Language model</text><text x="730" y="126" fill="#8fa0bd" font-size="11">attends over all vectors</text>
<rect x="885" y="80" width="100" height="70" rx="8" fill="#1b2436" stroke="#a06bd4" stroke-width="2"/><text x="935" y="115" fill="#e8eaf0" font-size="14">Text</text></g>
<g stroke="#8fa0bd" stroke-width="2" fill="none" marker-end="url(#ah8)"><line x1="125" y1="115" x2="197" y2="115"/><line x1="350" y1="115" x2="422" y2="115"/><line x1="575" y1="115" x2="647" y2="115"/><line x1="810" y1="115" x2="882" y2="115"/></g>
<g text-anchor="middle" fill="#c7d0e0" font-size="10.5"><text x="161" y="66">raw</text><text x="161" y="78">pixels</text><text x="386" y="66">patch</text><text x="386" y="78">features</text><text x="611" y="66">LLM-width</text><text x="611" y="78">vectors</text><text x="846" y="66">next-token</text><text x="846" y="78">logits</text></g>
<g text-anchor="middle" font-size="11" fill="#6f7f9c"><text x="70" y="172">input</text><text x="935" y="172">output</text></g>
<defs><marker id="ah8" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 Z" fill="#8fa0bd"/></marker></defs>
</svg></div>

The vision encoder is often **pretrained separately** (like CLIP). The connector and how the LLM reads visual features are the new work. <!-- .element: class="text-lg" -->

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

The LLM never learns whether the first vectors came from words or pixels. Attention mixes vectors; a visual prefix is just a prefix the text attends back to. <!-- .element: class="text-lg" style="margin-top: 10px;" -->

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

They differ in **how much of the stack is trained** and **where the visual information enters**. All four end at a transformer predicting the next text token. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="fig-flamingo-blip" -->

:::figure img="images/alayrac.png" name="Jean-Baptiste Alayrac and the Flamingo team" kicker="Flamingo: a Visual Language Model for Few-Shot Learning (DeepMind, 2022)"
- Connected **frozen** language models to visual inputs with gated cross-attention
- Learns few-shot from a handful of interleaved image-text examples in context
- <a href="https://arxiv.org/abs/2204.14198">arXiv:2204.14198</a>
:::

---

<!-- .slide: id="fig-blip2" -->

:::figure img="images/junnan_li.jpg" name="Junnan Li and collaborators" kicker="BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Encoders and LLMs (Salesforce Research, 2023)"
- Introduced the **Q-Former**: a lightweight bridge that queries a frozen image encoder
- Hands a compact set of vectors to a frozen LLM, so neither large model is retrained
- Strong results at a fraction of the training cost &mdash; <a href="https://arxiv.org/abs/2301.12597">arXiv:2301.12597</a>
:::

---

<!-- .slide: id="fig-llava" -->

:::figure img="images/haotian_liu.jpg" name="Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee" kicker="Visual Instruction Tuning (LLaVA, 2023)"
- Popularized **visual instruction tuning**: a simple projector from a frozen CLIP vision encoder into an LLM
- Finetuned on image-dialogue data, much of it generated by a stronger model
- Cheap, effective, and the template this module's exercise follows &mdash; <a href="https://arxiv.org/abs/2304.08485">arXiv:2304.08485</a>
:::

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

## The Projector Is Small but Not Trivial

:::columns cols="2" gap="34px"
**Why it looks trivial**

- Often a single linear layer
- A few hundred thousand parameters against a multi-billion-parameter LLM
+++
**Why it is not**

- It learns a **coordinate transformation** from vision-feature space to LLM language space
- A bad projector makes a strong vision encoder **useless** to the LLM
:::

Tradeoff: **freezing** large encoders is cheaper and more stable; **jointly tuning** more of the stack can improve alignment at higher cost. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

The exercise builds this bridge: project a tiny image embedding into NanoGPT's hidden size, train it to caption. <!-- .element: class="text-lg" -->
