:::divider id="divider-kv-cache" title="KV Cache" sub="Reusing work during generation"
:::

---

<!-- .slide: id="kv-cache-idea" -->

## The KV Cache

During autoregressive inference, the model generates one token at a time, and each new token attends to **every** token before it. Done naively, every step re-projects the keys and values for all previous tokens &mdash; the same vectors, computed again and again.

The keys and values for past tokens never change. So compute them once and **store** them.

---

<!-- .slide: id="kv-cache-demo" -->

:::interactive id="kv-cache-viz" widget="kvCache" title="Recompute Every Step, or Cache and Reuse"
:::

---

<!-- .slide: id="kv-cache-tradeoffs" -->

## KV Cache Tradeoffs

The KV cache is a classic memory-vs-compute tradeoff:

:::columns cols="2" gap="30px"
**Speed win**

Each generation step does $O(1)$ projection work instead of $O(n)$. Total generation cost drops from $O(n^2)$ to $O(n)$.
+++
**Memory cost**

The cache stores $2 \times n_{\text{layers}} \times n_{\text{heads}} \times n_{\text{tokens}} \times d_k$ floats. For a 70B model with 128K context, that can be tens of gigabytes.
:::

:::note
**KV cache memory is a major limit** for long context and large-batch serving. This is why MQA and GQA exist &mdash; fewer unique K and V heads means a smaller cache.
:::

---

<!-- .slide: id="attention-sinks" -->

## Attention Sinks

In a trained language model, a large share of every head's attention lands on the **very first token** &mdash; even when that token is something contentless like a start marker.

<div style="text-align: center; margin: 6px 0;">
<svg viewBox="0 0 660 130" width="100%" style="max-height: 120px;">
  <text x="30" y="20" fill="#8892a4" font-size="12">attention from a token late in the sequence:</text>
  <g text-anchor="middle" font-size="11">
    <rect x="40" y="30" width="60" height="70" rx="3" fill="rgba(245,166,35,0.55)"/><text x="70" y="115" fill="#e8eaf0">tok 1</text><text x="70" y="70" fill="#e8eaf0" font-weight="600">0.62</text>
    <rect x="110" y="86" width="60" height="14" rx="3" fill="rgba(74,158,255,0.45)"/><text x="140" y="115" fill="#8892a4">tok 2</text>
    <rect x="180" y="90" width="60" height="10" rx="3" fill="rgba(74,158,255,0.45)"/><text x="210" y="115" fill="#8892a4">tok 3</text>
    <rect x="250" y="84" width="60" height="16" rx="3" fill="rgba(74,158,255,0.45)"/><text x="280" y="115" fill="#8892a4">tok 4</text>
    <rect x="320" y="92" width="60" height="8" rx="3" fill="rgba(74,158,255,0.45)"/><text x="350" y="115" fill="#8892a4">tok 5</text>
    <rect x="390" y="78" width="60" height="22" rx="3" fill="rgba(74,158,255,0.45)"/><text x="420" y="115" fill="#8892a4">tok 6</text>
    <rect x="460" y="88" width="60" height="12" rx="3" fill="rgba(74,158,255,0.45)"/><text x="490" y="115" fill="#8892a4">tok 7</text>
  </g>
</svg>
</div>

**Why would the model dump attention on a token that carries no relevant meaning?**

:::note reveal="fragment"
Softmax weights must sum to 1, so a head always has to put its weight **somewhere**. When a head has nothing useful to attend to on a given token, it needs a no-op target to dump the leftover mass. Under causal masking, the first token is the one position visible to **every** later token, which makes it the natural sink. Xiao et al. (2023, StreamingLLM) showed that simply keeping these first tokens in the cache stabilizes very long generation.
:::

---

<!-- .slide: id="attention-maps" -->

## Attention Maps Are Not Always Explanations

Attention weights are a useful diagnostic &mdash; they show where the model is looking. But high attention does not prove causal importance.

:::columns cols="2" gap="30px"
**Attention weights can suggest hypotheses**

- A pronoun attending to its antecedent is a real, interpretable pattern
- Useful for debugging and building intuition
+++
**But high attention is not proof**

- Attention is one of many operations in the network
- The model can route information through other pathways
- Ablation tests are needed for stronger causal claims
:::

**The distinction:** visualization suggests hypotheses; ablation tests confirm them.
