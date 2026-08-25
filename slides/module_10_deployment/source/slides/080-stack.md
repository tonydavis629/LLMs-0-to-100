:::divider id="divider-stack" title="The Serving Stack" sub="Engines, the API, and the build-or-buy decision"
:::

---

<!-- .slide: id="stack-engines" -->

## Nobody Writes This From Scratch

<div class="bench-table">
<table>
<thead><tr><th>Engine</th><th>What it is</th><th>When you meet it</th></tr></thead>
<tbody>
<tr><td><strong>vLLM</strong></td><td>PagedAttention + continuous batching</td><td>The open-source default for GPU serving; this module's exercise</td></tr>
<tr><td><strong>SGLang</strong></td><td>Same problem, emphasis on structured output and prefix reuse</td><td>Agentic and constrained-generation workloads</td></tr>
<tr><td><strong>TensorRT-LLM</strong></td><td>NVIDIA's compiled kernels</td><td>Squeezing the last 2x out of NVIDIA hardware</td></tr>
<tr><td><strong>llama.cpp / Ollama</strong></td><td>Quantized GGUF inference, CPU included</td><td>Local machines, the exercise's laptop fallback</td></tr>
</tbody>
</table>
</div>

All expose the same thing: an HTTP server speaking the **OpenAI-compatible chat completions API**, the de facto wire protocol. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="stack-api" -->

## The API Request Is the Course in Miniature

```json
{
  "model": "Qwen2.5-0.5B-Instruct",
  "messages": [{"role": "user", "content": "Explain HBM in one sentence."}],
  "temperature": 0.7,
  "max_tokens": 120,
  "stream": true
}
```

<div class="bench-table dense">
<table>
<thead><tr><th>Field</th><th>Where you learned it</th></tr></thead>
<tbody>
<tr><td><code>messages</code></td><td>The chat template, applied server-side (Module 6). The client never sees tokens, only text</td></tr>
<tr><td><code>temperature</code></td><td>Sampling (Module 4)</td></tr>
<tr><td><code>max_tokens</code></td><td>A cap on the decode loop (this module)</td></tr>
<tr><td><code>stream</code></td><td>Send each token as it is made; tokens arrive as server-sent events</td></tr>
<tr><td><code>usage</code> in the reply</td><td>Prompt and completion token counts: <strong>the billing meter</strong></td></tr>
</tbody>
</table>
</div>

---

<!-- .slide: id="stack-buy-or-build" -->

## Rent the API, or Run Your Own?

The crossover is a spreadsheet. You know every row in it.

:::columns cols="2" gap="34px"
**Rent when**

- Load is small or spiky (you pay only for tokens)
- You need frontier-class models
- Undifferentiated ops work is not your job
+++
**Self-host when**

- Token volume is high and steady; at scale the crossover comes fast
- Data cannot leave, or latency must be controlled
- The model is your own finetune (Module 6)
:::

---

<!-- .slide: id="stack-hygiene" -->

## Production Hygiene, and the Next Class

- Run the **Module 9 regression suite** before swapping any model or quantization into production; a quantized model ships like a new model
- **Canary** a new model on a slice of traffic before it takes all of it
- **Log per-request latency and token counts**: they are the cost model and the debugging record in one

The model is now a URL: tokens in, tokens out, at a price per token. Everything in Module 11 (prompting, retrieval, tools, agents) is engineering built on top of that URL. <!-- .element: class="text-lg" style="margin-top: 14px;" -->
