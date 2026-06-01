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

<!-- .slide: id="flash-attention-explained" -->

## FlashAttention: Same Math, Less Memory Traffic

Standard attention often writes the full $n \times n$ attention matrix to GPU memory, then reads it back to multiply by $V$. FlashAttention avoids that expensive round trip.
<div style="text-align: center; margin: 8px 0;">
<svg viewBox="0 0 820 250" width="100%" style="max-height: 235px;">
  <text x="190" y="24" fill="#e74c3c" font-size="13" text-anchor="middle" font-weight="600">standard attention</text>
  <text x="620" y="24" fill="#3fb950" font-size="13" text-anchor="middle" font-weight="600">FlashAttention</text>
  <rect x="70" y="52" width="240" height="130" rx="6" fill="rgba(231,76,60,0.08)" stroke="#e74c3c" stroke-width="1.5"/>
  <text x="190" y="76" fill="#e8eaf0" font-size="12" text-anchor="middle">materialize full score matrix</text>
  <rect x="116" y="94" width="148" height="64" fill="rgba(74,158,255,0.20)" stroke="#4a9eff" stroke-width="1.2"/>
  <g stroke="#4a9eff" stroke-width="0.6" opacity="0.45">
    <line x1="140" y1="94" x2="140" y2="158"/><line x1="164" y1="94" x2="164" y2="158"/><line x1="188" y1="94" x2="188" y2="158"/><line x1="212" y1="94" x2="212" y2="158"/><line x1="236" y1="94" x2="236" y2="158"/>
    <line x1="116" y1="110" x2="264" y2="110"/><line x1="116" y1="126" x2="264" y2="126"/><line x1="116" y1="142" x2="264" y2="142"/>
  </g>
  <text x="190" y="206" fill="#e74c3c" font-size="12" text-anchor="middle">large HBM reads and writes</text>
  <rect x="500" y="52" width="240" height="130" rx="6" fill="rgba(63,185,80,0.08)" stroke="#3fb950" stroke-width="1.5"/>
  <text x="620" y="76" fill="#e8eaf0" font-size="12" text-anchor="middle">tile Q, K, V in small blocks</text>
  <g>
    <rect x="548" y="94" width="148" height="64" fill="rgba(74,158,255,0.12)" stroke="#4a9eff" stroke-width="1.2"/>
    <rect x="548" y="94" width="50" height="22" fill="rgba(63,185,80,0.55)" stroke="#3fb950"/>
    <rect x="598" y="116" width="50" height="22" fill="rgba(63,185,80,0.45)" stroke="#3fb950"/>
    <rect x="648" y="136" width="48" height="22" fill="rgba(63,185,80,0.35)" stroke="#3fb950"/>
  </g>
  <text x="620" y="206" fill="#3fb950" font-size="12" text-anchor="middle">keep intermediates in fast SRAM</text>
  <line x1="350" y1="118" x2="470" y2="118" stroke="#8892a4" stroke-width="1.5" marker-end="url(#arrfa)"/>
  <text x="410" y="106" fill="#8892a4" font-size="12" text-anchor="middle">reorder computation</text>
  <defs><marker id="arrfa" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#8892a4"/></marker></defs>
</svg>
</div>

The result is still exact scaled dot-product attention. The speedup comes from doing less slow memory movement, not from changing the equation. Reference: Dao et al. (2022), *FlashAttention*.

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
