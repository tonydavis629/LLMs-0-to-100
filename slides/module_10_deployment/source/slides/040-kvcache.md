:::divider id="divider-kvcache" title="The KV Cache" sub="The memory a conversation takes up"
:::

---

<!-- .slide: id="kvcache-what" -->

## Attention Remembers So You Don't Recompute

Attention at position t needs the **keys and values of every earlier position** (Module 3). Recomputing them every step is quadratic waste. The server keeps them: the **KV cache**.

<div style="display: flex; justify-content: center; margin: 4px 0;">
<svg viewBox="0 0 940 250" style="max-width: 880px; width: 100%;" font-family="inherit">
  <g font-size="15" fill="#e6edf3" text-anchor="middle">
    <rect x="60" y="30" width="90" height="32" rx="6" fill="none" stroke="#8b98b8"/><text x="105" y="51">The</text>
    <rect x="160" y="30" width="90" height="32" rx="6" fill="none" stroke="#8b98b8"/><text x="205" y="51">sky</text>
    <rect x="260" y="30" width="90" height="32" rx="6" fill="none" stroke="#8b98b8"/><text x="305" y="51">is</text>
    <rect x="360" y="30" width="90" height="32" rx="6" fill="none" stroke="#e8a34c" stroke-width="2"/><text x="405" y="51" fill="#e8a34c">blue</text>
  </g>
  <g font-size="13" text-anchor="middle">
    <rect x="72" y="86" width="66" height="26" rx="4" fill="rgba(76,155,232,0.2)" stroke="#4c9be8"/><text x="105" y="104" fill="#e6edf3">K, V</text>
    <rect x="72" y="118" width="66" height="26" rx="4" fill="rgba(76,155,232,0.2)" stroke="#4c9be8"/><text x="105" y="136" fill="#e6edf3">K, V</text>
    <rect x="72" y="150" width="66" height="26" rx="4" fill="rgba(76,155,232,0.2)" stroke="#4c9be8"/><text x="105" y="168" fill="#e6edf3">K, V</text>
    <rect x="172" y="86" width="66" height="26" rx="4" fill="rgba(76,155,232,0.2)" stroke="#4c9be8"/><text x="205" y="104" fill="#e6edf3">K, V</text>
    <rect x="172" y="118" width="66" height="26" rx="4" fill="rgba(76,155,232,0.2)" stroke="#4c9be8"/><text x="205" y="136" fill="#e6edf3">K, V</text>
    <rect x="172" y="150" width="66" height="26" rx="4" fill="rgba(76,155,232,0.2)" stroke="#4c9be8"/><text x="205" y="168" fill="#e6edf3">K, V</text>
    <rect x="272" y="86" width="66" height="26" rx="4" fill="rgba(76,155,232,0.2)" stroke="#4c9be8"/><text x="305" y="104" fill="#e6edf3">K, V</text>
    <rect x="272" y="118" width="66" height="26" rx="4" fill="rgba(76,155,232,0.2)" stroke="#4c9be8"/><text x="305" y="136" fill="#e6edf3">K, V</text>
    <rect x="272" y="150" width="66" height="26" rx="4" fill="rgba(76,155,232,0.2)" stroke="#4c9be8"/><text x="305" y="168" fill="#e6edf3">K, V</text>
    <rect x="372" y="86" width="66" height="26" rx="4" fill="rgba(232,163,76,0.2)" stroke="#e8a34c"/><text x="405" y="104" fill="#e6edf3">K, V</text>
    <rect x="372" y="118" width="66" height="26" rx="4" fill="rgba(232,163,76,0.2)" stroke="#e8a34c"/><text x="405" y="136" fill="#e6edf3">K, V</text>
    <rect x="372" y="150" width="66" height="26" rx="4" fill="rgba(232,163,76,0.2)" stroke="#e8a34c"/><text x="405" y="168" fill="#e6edf3">K, V</text>
  </g>
  <text x="42" y="104" fill="#8b98b8" font-size="13" text-anchor="end">layer 1</text>
  <text x="42" y="136" fill="#8b98b8" font-size="13" text-anchor="end">layer 2</text>
  <text x="42" y="168" fill="#8b98b8" font-size="13" text-anchor="end">layer N</text>
  <text x="250" y="205" fill="#4c9be8" font-size="15" text-anchor="middle">kept from earlier steps, read every step</text>
  <text x="405" y="228" fill="#e8a34c" font-size="15" text-anchor="middle">the new token appends one column</text>
  <g fill="#e6edf3" font-size="16">
    <text x="530" y="100">Each token of context adds one</text>
    <text x="530" y="126">K and one V vector <tspan fill="#4c9be8">per layer</tspan>,</text>
    <text x="530" y="152"><tspan fill="#4c9be8">per KV head</tspan>. The cache never</text>
    <text x="530" y="178">shrinks while the request lives.</text>
  </g>
</svg>
</div>

<div class="metric-box">

$$\text{bytes per token} = 2 \cdot n_{\text{layers}} \cdot n_{\text{kv-heads}} \cdot d_{\text{head}} \cdot \text{bytes per value}$$

<p>The 2 is one key and one value.</p>
</div>

---

<!-- .slide: id="kvcache-cost" -->

## Work It Out for a 7B-Class Model

32 layers, 32 heads of dimension 128, fp16. That is 2 &times; 32 &times; 32 &times; 128 &times; 2 bytes, about **0.5 MB per token of context**.

<div class="card-grid cols-3">
<div class="card"><h4>One user, 4K context</h4><p>4,096 tokens &times; 0.5 MB = <strong>about 2 GB</strong> of GPU memory, for one conversation.</p></div>
<div class="card warn"><h4>Eight such users</h4><p><strong>16 GB of cache</strong>: more memory than the entire 14 GB model. The cache, not compute, decides how many users fit.</p></div>
<div class="card"><h4>And it is traffic too</h4><p>Every decode step <strong>reads the whole cache</strong> on top of the weights, so long conversations also decode slower.</p></div>
</div>

This is why long contexts cost more: expensive in memory and bandwidth even when the arithmetic is cheap. <!-- .element: class="text-lg" style="margin-top: 10px;" -->

---

:::interactive id="kv-budget" widget="kvBudget" title="Fit It on the Card"
:::

---

<!-- .slide: id="kvcache-architecture" -->

## Architecture Fights Back: MQA and GQA

The cache scales with the number of **KV heads**, so shrink that number.

:::columns cols="2" gap="34px"
**Multi-query attention (2019)**

- All query heads share **one** K/V set
- Cache divided by 32 in our example
- Shazeer, years ahead of the field
+++
**Grouped-query attention (2023)**

- The compromise that won: a few KV heads (Llama 3 8B uses 8)
- Most of the saving, almost none of the quality loss
- Module 4's modern block
:::

Serving costs reached back and **changed model architecture**. Watch for this again with MoE. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

:::figure img="images/woosuk_kwon.jpg" name="Woosuk Kwon" kicker="vLLM and PagedAttention (2023), with Zhuohan Li and the Berkeley team" alt="Woosuk Kwon"
Serving engines gave each request one **contiguous slab** of cache memory, sized for the maximum context. Measured waste: 60-80% of cache memory sat empty.

The fix imported a fifty-year-old operating systems idea: manage the cache in small **pages** with a lookup table, like virtual memory. Near-zero waste, far more concurrent users per card. vLLM became the open-source default; you run it in the exercise.
:::

---

<!-- .slide: id="kvcache-prefix" -->

## Prefix Caching: The Part Users Actually Meet

Requests that share a prefix (system prompt, long document, the conversation so far) can **share the cache** for those tokens. Only the new suffix needs prefill.

<div class="card-grid cols-2">
<div class="card good"><h4>Why providers sell "prompt caching"</h4><p>Prefill for a stable prefix is computed once, then billed at a discount. The KV cache, productized.</p></div>
<div class="card"><h4>The design rule</h4><p><strong>Stable content first</strong> (system prompt, tools, examples), variable content last. Module 11 builds on this.</p></div>
</div>
