:::divider id="divider-batching" title="Batching" sub="Serving many users, and the economics of the API business"
:::

---

<!-- .slide: id="batching-insight" -->

## Cashing In the Escape Hatch

<div style="display: flex; justify-content: center; margin: 8px 0;">
<svg viewBox="0 0 960 330" style="max-width: 920px; width: 100%;" font-family="inherit">
  <rect x="40" y="90" width="150" height="110" rx="10" fill="rgba(139,152,184,0.12)" stroke="#8b98b8" stroke-width="2"/>
  <text x="115" y="135" fill="#e6edf3" font-size="17" text-anchor="middle">weights</text>
  <text x="115" y="160" fill="#8b98b8" font-size="14" text-anchor="middle">14 GB in HBM</text>
  <line x1="190" y1="145" x2="280" y2="145" stroke="#e06c75" stroke-width="3"/>
  <rect x="285" y="128" width="110" height="34" rx="6" fill="none" stroke="#e06c75" stroke-width="2"/>
  <text x="340" y="150" fill="#e06c75" font-size="15" text-anchor="middle">1 token</text>
  <text x="218" y="240" fill="#e6edf3" font-size="16" text-anchor="middle">batch of 1: one read,</text>
  <text x="218" y="263" fill="#e06c75" font-size="16" font-weight="700" text-anchor="middle">one token</text>
  <line x1="480" y1="40" x2="480" y2="290" stroke="#2a3450" stroke-width="2"/>
  <rect x="530" y="90" width="150" height="110" rx="10" fill="rgba(139,152,184,0.12)" stroke="#8b98b8" stroke-width="2"/>
  <text x="605" y="135" fill="#e6edf3" font-size="17" text-anchor="middle">weights</text>
  <text x="605" y="160" fill="#8b98b8" font-size="14" text-anchor="middle">same 14 GB</text>
  <g stroke="#50c878" stroke-width="2.5">
    <line x1="680" y1="120" x2="760" y2="62"/><line x1="680" y1="132" x2="760" y2="103"/><line x1="680" y1="145" x2="760" y2="145"/><line x1="680" y1="158" x2="760" y2="187"/><line x1="680" y1="170" x2="760" y2="228"/>
  </g>
  <g font-size="14" fill="#50c878" text-anchor="middle">
    <rect x="765" y="46" width="120" height="30" rx="6" fill="none" stroke="#50c878" stroke-width="2"/><text x="825" y="66">user 1's token</text>
    <rect x="765" y="88" width="120" height="30" rx="6" fill="none" stroke="#50c878" stroke-width="2"/><text x="825" y="108">user 2's token</text>
    <rect x="765" y="130" width="120" height="30" rx="6" fill="none" stroke="#50c878" stroke-width="2"/><text x="825" y="150">user 3's token</text>
    <text x="825" y="192">&#8942;</text>
    <rect x="765" y="212" width="120" height="30" rx="6" fill="none" stroke="#50c878" stroke-width="2"/><text x="825" y="232">user 32's token</text>
  </g>
  <text x="710" y="285" fill="#e6edf3" font-size="16" text-anchor="middle">batch of 32: one read, <tspan fill="#50c878" font-weight="700">32 tokens</tspan></text>
</svg>
</div>

Throughput scales almost linearly with batch size until compute binds or KV caches fill memory. A full batch divides cost per token by **10x or more**. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="batching-continuous" -->

## Static Batching Fails; Continuous Batching Won

Replies finish at wildly different lengths, and that kills the convoy. Continuous batching (Orca, 2022) reschedules **every decode step**:

<div style="display: flex; justify-content: center; margin: 6px 0;">
<svg viewBox="0 0 960 350" style="max-width: 920px; width: 100%;" font-family="inherit">
  <text x="230" y="26" fill="#e06c75" font-size="19" font-weight="700" text-anchor="middle">static: batch as convoy</text>
  <g font-size="13" fill="#8b98b8" text-anchor="end">
    <text x="66" y="66">slot 1</text><text x="66" y="106">slot 2</text><text x="66" y="146">slot 3</text>
  </g>
  <rect x="78" y="48" width="330" height="26" rx="5" fill="rgba(76,155,232,0.5)"/>
  <rect x="78" y="88" width="90" height="26" rx="5" fill="rgba(76,155,232,0.5)"/>
  <rect x="170" y="88" width="238" height="26" rx="5" fill="rgba(224,108,117,0.25)" stroke="#e06c75" stroke-dasharray="5 4"/>
  <text x="289" y="106" fill="#e06c75" font-size="13" text-anchor="middle">idle: waits for slot 1</text>
  <rect x="78" y="128" width="180" height="26" rx="5" fill="rgba(76,155,232,0.5)"/>
  <rect x="260" y="128" width="148" height="26" rx="5" fill="rgba(224,108,117,0.25)" stroke="#e06c75" stroke-dasharray="5 4"/>
  <text x="334" y="146" fill="#e06c75" font-size="13" text-anchor="middle">idle</text>
  <rect x="78" y="168" width="330" height="26" rx="5" fill="rgba(139,152,184,0.18)" stroke="#8b98b8" stroke-dasharray="5 4"/>
  <text x="243" y="186" fill="#8b98b8" font-size="13" text-anchor="middle">new arrivals wait for the whole convoy</text>
  <text x="243" y="230" fill="#8b98b8" font-size="14" text-anchor="middle">everyone returns when the essay finishes</text>
  <line x1="480" y1="20" x2="480" y2="250" stroke="#2a3450" stroke-width="2"/>
  <text x="730" y="26" fill="#50c878" font-size="19" font-weight="700" text-anchor="middle">continuous: rolling population</text>
  <g font-size="13" fill="#8b98b8" text-anchor="end">
    <text x="566" y="66">slot 1</text><text x="566" y="106">slot 2</text><text x="566" y="146">slot 3</text>
  </g>
  <rect x="578" y="48" width="330" height="26" rx="5" fill="rgba(76,155,232,0.5)"/>
  <rect x="578" y="88" width="90" height="26" rx="5" fill="rgba(76,155,232,0.5)"/>
  <rect x="672" y="88" width="236" height="26" rx="5" fill="rgba(80,200,120,0.45)"/>
  <text x="790" y="106" fill="#0b1020" font-size="13" font-weight="700" text-anchor="middle">next request enters mid-flight</text>
  <rect x="578" y="128" width="180" height="26" rx="5" fill="rgba(76,155,232,0.5)"/>
  <rect x="762" y="128" width="146" height="26" rx="5" fill="rgba(80,200,120,0.45)"/>
  <text x="835" y="146" fill="#0b1020" font-size="13" font-weight="700" text-anchor="middle">another one</text>
  <text x="743" y="230" fill="#8b98b8" font-size="14" text-anchor="middle">finished requests leave, waiting ones enter, every step</text>
  <text x="480" y="290" fill="#e6edf3" font-size="16" text-anchor="middle">time &#8594;&#160;&#160;&#160;blue = a request decoding&#160;&#160;&#160;red = wasted slot&#160;&#160;&#160;green = reclaimed slot</text>
</svg>
</div>

:::note reveal="fragment"
Remaining interference: a newcomer's prefill is a compute burst that stalls everyone's decode. Schedulers chunk prefills and interleave the pieces. Unchunked, you feel it as jittery streaming.
:::

---

<!-- .slide: id="batching-goodput" -->

## What the Operator Actually Tunes

- The knobs: batch size and cache memory, against a latency promise
- More concurrency: more throughput, slower streams
- Somewhere there is a line the user experience must not cross

**Goodput** is the honest metric: throughput that still meets the latency target. Raw throughput can always be inflated by making everyone wait. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="batching-side-quest-cost" -->

## Side Quest: What Does a Token Actually Cost?

Work backward from a rented H100 at about $3/hour and a realistic batched throughput for an 8B model:

<div class="bench-table">
<table>
<thead><tr><th>Quantity</th><th>Value</th></tr></thead>
<tbody>
<tr><td>Batched throughput (continuous batching, dozens of streams)</td><td class="num">~5,000 tok/s</td></tr>
<tr><td>Tokens per hour</td><td class="num">~18M</td></tr>
<tr><td><strong>Hardware cost per million output tokens</strong></td><td class="num"><strong>~$0.17</strong></td></tr>
<tr><td>Same card, one user at a time (~150 tok/s)</td><td class="num">~$5.50 per million</td></tr>
</tbody>
</table>
</div>

The 30x gap between those last two rows is batching. It is also roughly the gap between a public API's price and what a naive self-hoster pays. <!-- .element: class="text-lg" -->
