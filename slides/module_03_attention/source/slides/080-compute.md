:::divider id="divider-compute" title="Memory and Compute" sub="The cost of every token seeing every other token"
:::

---

<!-- .slide: id="n-squared-cost" -->

## The $O(n^2)$ Cost

Attention builds an $n \times n$ score matrix: every token compared against every other token. That matrix is the bottleneck.

<div style="text-align: center; margin: 4px 0;">
<svg viewBox="0 0 720 250" width="100%" style="max-height: 240px;">
  <!-- big n x n grid -->
  <text x="170" y="22" fill="#8892a4" font-size="12" text-anchor="middle">n keys</text>
  <text x="36" y="135" fill="#8892a4" font-size="12" text-anchor="middle" transform="rotate(-90 36 135)">n queries</text>
  <g>
    <rect x="60" y="36" width="200" height="200" fill="rgba(74,158,255,0.18)" stroke="#4a9eff" stroke-width="1.5"/>
    <!-- grid lines -->
    <g stroke="#4a9eff" stroke-width="0.5" opacity="0.35">
      <line x1="85" y1="36" x2="85" y2="236"/><line x1="110" y1="36" x2="110" y2="236"/><line x1="135" y1="36" x2="135" y2="236"/><line x1="160" y1="36" x2="160" y2="236"/><line x1="185" y1="36" x2="185" y2="236"/><line x1="210" y1="36" x2="210" y2="236"/><line x1="235" y1="36" x2="235" y2="236"/>
      <line x1="60" y1="61" x2="260" y2="61"/><line x1="60" y1="86" x2="260" y2="86"/><line x1="60" y1="111" x2="260" y2="111"/><line x1="60" y1="136" x2="260" y2="136"/><line x1="60" y1="161" x2="260" y2="161"/><line x1="60" y1="186" x2="260" y2="186"/><line x1="60" y1="211" x2="260" y2="211"/>
    </g>
    <text x="160" y="140" fill="#e8eaf0" font-size="16" text-anchor="middle" font-weight="600">n&#178; entries</text>
  </g>
  <!-- text panel -->
  <text x="320" y="70" fill="#e8eaf0" font-size="14">Compute: O(n&#178; &middot; d_k)</text>
  <text x="320" y="100" fill="#e8eaf0" font-size="14">Memory: O(n&#178;)</text>
  <text x="320" y="140" fill="#f5a623" font-size="14">Double the context, and the</text>
  <text x="320" y="162" fill="#f5a623" font-size="14">attention cost quadruples.</text>
  <text x="320" y="200" fill="#8892a4" font-size="13">At 128K tokens, the matrix has</text>
  <text x="320" y="220" fill="#8892a4" font-size="13">about 16 billion entries.</text>
</svg>
</div>

The $O(n^2)$ cost is the main barrier to very long contexts: it grows far faster than the per-token feed-forward work, which is only $O(n)$.

---

:::figure img="images/dao.jpg" name="Tri Dao" kicker="Made Exact Attention Fast"
- Tri Dao and collaborators (2022): FlashAttention
- Observed that materializing the full $n \times n$ matrix in slow GPU memory (HBM) is the real bottleneck
- FlashAttention computes **exact** attention without ever writing the full matrix to HBM
- IO-aware: it tiles the computation to keep intermediate results in fast on-chip SRAM
- Identical result to standard attention, but 2-4x faster and 5-20x less memory
- FlashAttention-2 and -3 pushed further on newer GPUs
:::

---

<!-- .slide: id="mitigations" -->

## Modern Mitigations

Several approaches reduce the $O(n^2)$ cost without sacrificing too much quality:

:::columns cols="2" gap="30px"
**FlashAttention**

Computes exact attention without materializing the full matrix. Same result, far less memory traffic.
+++
**Sliding-window attention**

Each token attends only to a local window of neighbors (plus a few global tokens). Reduces the effective $n$.
+++
**Multi-query attention (MQA)**

All heads share one K and V projection. Only Q is per-head. Shrinks the KV cache.
+++
**Grouped-query attention (GQA)**

A compromise: heads are divided into groups that share K and V. Used in LLaMA 2/3.
:::
