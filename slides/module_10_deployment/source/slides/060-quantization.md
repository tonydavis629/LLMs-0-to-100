:::divider id="divider-quant" title="Making Tokens Cheaper" sub="Quantization, distillation, and speculative decoding"
:::

---

<!-- .slide: id="quant-side-quest-home" -->

## Side Quest: Can You Run It at Home?

What fewer bytes buys. Weight memory at three precisions, against real machines:

<div class="bench-table dense">
<table>
<thead><tr><th>Model</th><th>fp16</th><th>int8</th><th>int4</th><th>Fits where at int4?</th></tr></thead>
<tbody>
<tr><td>Qwen2.5-0.5B</td><td class="num">1.0 GB</td><td class="num">0.5 GB</td><td class="num">0.2 GB</td><td>A phone</td></tr>
<tr><td>Llama-3-8B</td><td class="num">16.1 GB</td><td class="num">8.0 GB</td><td class="num">4.0 GB</td><td>Any recent laptop</td></tr>
<tr><td>Llama-3-70B</td><td class="num">141.2 GB</td><td class="num">70.6 GB</td><td class="num">35.3 GB</td><td>A 64 GB workstation</td></tr>
</tbody>
</table>
</div>

Each column halves the bytes, which halves the memory **and** doubles the decode speed limit. The whole local-LLM world lives in the right-hand columns. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="quant-how" -->

## Quantization: Fewer Bits per Weight

Inference weights tolerate 8-bit and even 4-bit integers, with one scale factor per small group preserving the range:

<div class="metric-box">

$$w \approx s \cdot q, \qquad q \in \lbrace -8, \dots, 7 \rbrace\ \text{for 4-bit}$$

<p>Store small integers; keep one fp16 scale <strong>s</strong> per group of ~32 weights; multiply back at compute time.</p>
</div>

<div style="display: flex; justify-content: center; margin: 8px 0;">
<svg viewBox="0 0 960 170" style="max-width: 920px; width: 100%;" font-family="inherit">
  <text x="130" y="46" fill="#8b98b8" font-size="15" text-anchor="end">fp16</text>
  <g>
    <rect x="150" y="26" width="176" height="32" rx="4" fill="rgba(139,152,184,0.25)" stroke="#8b98b8"/>
    <rect x="330" y="26" width="176" height="32" rx="4" fill="rgba(139,152,184,0.25)" stroke="#8b98b8"/>
    <rect x="510" y="26" width="176" height="32" rx="4" fill="rgba(139,152,184,0.25)" stroke="#8b98b8"/>
  </g>
  <text x="418" y="47" fill="#e6edf3" font-size="14" text-anchor="middle">16 bits per weight</text>
  <text x="740" y="47" fill="#8b98b8" font-size="15">14 GB for 7B</text>
  <text x="130" y="120" fill="#8b98b8" font-size="15" text-anchor="end">int4</text>
  <g>
    <rect x="150" y="100" width="44" height="32" rx="4" fill="rgba(80,200,120,0.3)" stroke="#50c878"/>
    <rect x="196" y="100" width="44" height="32" rx="4" fill="rgba(80,200,120,0.3)" stroke="#50c878"/>
    <rect x="242" y="100" width="44" height="32" rx="4" fill="rgba(80,200,120,0.3)" stroke="#50c878"/>
    <rect x="288" y="100" width="44" height="32" rx="4" fill="rgba(80,200,120,0.3)" stroke="#50c878"/>
    <rect x="334" y="100" width="30" height="32" rx="4" fill="rgba(76,155,232,0.35)" stroke="#4c9be8"/>
  </g>
  <text x="242" y="152" fill="#50c878" font-size="13" text-anchor="middle">4 bits each</text>
  <text x="360" y="152" fill="#4c9be8" font-size="13" text-anchor="start">shared scale s</text>
  <text x="420" y="121" fill="#50c878" font-size="15">3.5 GB for 7B: 4x less traffic, 4x the speed limit, fits on a laptop</text>
</svg>
</div>

---

:::interactive id="quant-explorer" widget="quantExplorer" title="Rounding Weights onto a Grid"
:::

---

<!-- .slide: id="quant-catch" -->

## The Catch, and the Module 9 Connection

Quantization is a quality trade, and the loss is **uneven**: small on average, occasionally large on a specific capability.

<div class="card-grid cols-2">
<div class="card warn"><h4>Do not trust the average</h4><p>A tiny perplexity delta can hide a real regression on, say, math or a low-resource language. Aggregates hide exactly what you need to know; Module 9's oldest lesson.</p></div>
<div class="card good"><h4>The honest procedure</h4><p><strong>Re-run the evaluation suite on the quantized model.</strong> Ship it like a new model, because it is one.</p></div>
</div>

Names on download pages: <!-- .element: class="text-lg" style="margin-top: 10px;" -->

- **GPTQ**, **AWQ**: weight-only quantization, computed after training
- **GGUF**: llama.cpp's file family
- **fp8**: the emerging served-model default; frontier labs increasingly train at low precision from the start

:::note
Distillation is the other shrink: train a small model on a large model's outputs (Module 6 machinery). It cuts parameter count, not bytes per parameter. Most small production models are distilled from larger siblings.
:::

---

:::figure img="images/georgi_gerganov.jpg" name="Georgi Gerganov" kicker="llama.cpp (2023)" alt="Georgi Gerganov"
Wrote a plain C/C++ inference engine, essentially alone, that ran Llama on a MacBook days after the weights leaked. Aggressive quantization plus careful CPU code proved serious models do not need a datacenter. The GGUF ecosystem around it (Ollama descends from it) put local inference in everyone's hands.

One engineer, zero GPUs. In the exercise, his server is the fallback when you have no NVIDIA card.
:::

---

<!-- .slide: id="quant-speculative" -->

## Speculative Decoding: Spend Prefill to Save Decode

A small **draft model** guesses ahead; the large model **verifies all k guesses in one prefill-shaped pass**:

<div style="display: flex; justify-content: center; margin: 6px 0;">
<svg viewBox="0 0 960 300" style="max-width: 920px; width: 100%;" font-family="inherit">
  <text x="120" y="60" fill="#e8a34c" font-size="16" font-weight="700" text-anchor="end">draft model</text>
  <text x="120" y="82" fill="#8b98b8" font-size="13" text-anchor="end">cheap, fast</text>
  <g font-size="15" text-anchor="middle">
    <rect x="150" y="42" width="110" height="34" rx="6" fill="rgba(232,163,76,0.15)" stroke="#e8a34c"/><text x="205" y="64" fill="#e6edf3">The</text>
    <rect x="268" y="42" width="110" height="34" rx="6" fill="rgba(232,163,76,0.15)" stroke="#e8a34c"/><text x="323" y="64" fill="#e6edf3">sky</text>
    <rect x="386" y="42" width="110" height="34" rx="6" fill="rgba(232,163,76,0.15)" stroke="#e8a34c"/><text x="441" y="64" fill="#e6edf3">is</text>
    <rect x="504" y="42" width="110" height="34" rx="6" fill="rgba(232,163,76,0.15)" stroke="#e8a34c"/><text x="559" y="64" fill="#e6edf3">green</text>
  </g>
  <text x="700" y="64" fill="#8b98b8" font-size="14">k = 4 drafted tokens</text>
  <g stroke="#4c9be8" stroke-width="2">
    <line x1="205" y1="80" x2="205" y2="130"/><line x1="323" y1="80" x2="323" y2="130"/><line x1="441" y1="80" x2="441" y2="130"/><line x1="559" y1="80" x2="559" y2="130"/>
  </g>
  <rect x="150" y="132" width="464" height="44" rx="8" fill="rgba(76,155,232,0.15)" stroke="#4c9be8" stroke-width="2"/>
  <text x="382" y="160" fill="#e6edf3" font-size="16" text-anchor="middle">large model: one parallel pass checks all four</text>
  <g font-size="20" text-anchor="middle">
    <text x="205" y="215" fill="#50c878">&#10003;</text><text x="323" y="215" fill="#50c878">&#10003;</text><text x="441" y="215" fill="#50c878">&#10003;</text><text x="559" y="215" fill="#e06c75">&#10007;</text>
  </g>
  <text x="323" y="248" fill="#50c878" font-size="15" text-anchor="middle">3 tokens accepted for ~1 token's traffic</text>
  <text x="660" y="215" fill="#e06c75" font-size="15">replaced by the large</text>
  <text x="660" y="235" fill="#e06c75" font-size="15">model's own token ("blue")</text>
</svg>
</div>

- Accept/reject rule leaves the output distribution **provably unchanged**
- Typical speedup: **2-3x** on predictable stretches
- Nothing changes what the model knows, only **bytes moved per token**
