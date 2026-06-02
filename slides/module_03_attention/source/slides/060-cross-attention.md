:::divider id="divider-cross-attention" title="Cross Attention" sub="When queries and keys come from different places"
:::

---

<!-- .slide: id="cross-vs-self" -->

## Cross Attention vs. Self-Attention

<div style="text-align: center; margin: 6px 0;">
<svg viewBox="0 0 820 250" width="100%" style="max-height: 240px;">
  <!-- Self-attention panel -->
  <text x="195" y="22" fill="#4a9eff" font-size="14" text-anchor="middle" font-weight="600">Self-Attention</text>
  <g font-size="13" text-anchor="middle" font-weight="600">
    <rect x="70" y="100" width="74" height="32" rx="4" fill="#0d1225" stroke="#4a9eff" stroke-width="1.5"/><text x="107" y="121" fill="#e8eaf0">the</text>
    <rect x="158" y="100" width="74" height="32" rx="4" fill="#0d1225" stroke="#4a9eff" stroke-width="1.5"/><text x="195" y="121" fill="#e8eaf0">cat</text>
    <rect x="246" y="100" width="74" height="32" rx="4" fill="#0d1225" stroke="#4a9eff" stroke-width="1.5"/><text x="283" y="121" fill="#e8eaf0">sat</text>
  </g>
  <!-- self arcs (token to token within same row) -->
  <path d="M195 100 Q151 64 107 100" fill="none" stroke="#f5a623" stroke-width="2" marker-end="url(#arrcx)"/>
  <path d="M195 100 Q239 64 283 100" fill="none" stroke="#f5a623" stroke-width="2" marker-end="url(#arrcx)"/>
  <text x="195" y="200" fill="#8892a4" font-size="12" text-anchor="middle">Q, K, V all from the</text>
  <text x="195" y="218" fill="#8892a4" font-size="12" text-anchor="middle">same sequence</text>
  <!-- divider -->
  <line x1="410" y1="40" x2="410" y2="220" stroke="#2a3450" stroke-width="1"/>
  <!-- Cross-attention panel -->
  <text x="615" y="22" fill="#50c878" font-size="14" text-anchor="middle" font-weight="600">Cross-Attention</text>
  <!-- decoder row (queries) -->
  <text x="455" y="62" fill="#8892a4" font-size="11" text-anchor="end">target side (Q)</text>
  <g font-size="13" text-anchor="middle" font-weight="600">
    <rect x="470" y="46" width="74" height="32" rx="4" fill="#0d1225" stroke="#50c878" stroke-width="1.5"/><text x="507" y="67" fill="#e8eaf0">the</text>
    <rect x="558" y="46" width="74" height="32" rx="4" fill="#0d1225" stroke="#50c878" stroke-width="2.5"/><text x="595" y="67" fill="#e8eaf0">cat</text>
  </g>
  <!-- encoder row (keys/values) -->
  <text x="455" y="162" fill="#8892a4" font-size="11" text-anchor="end">source side (K, V)</text>
  <g font-size="13" text-anchor="middle" font-weight="600">
    <rect x="470" y="146" width="74" height="32" rx="4" fill="#0d1225" stroke="#f5a623" stroke-width="1.5"/><text x="507" y="167" fill="#e8eaf0">le</text>
    <rect x="558" y="146" width="74" height="32" rx="4" fill="#0d1225" stroke="#f5a623" stroke-width="2.5"/><text x="595" y="167" fill="#e8eaf0">chat</text>
    <rect x="646" y="146" width="100" height="32" rx="4" fill="#0d1225" stroke="#f5a623" stroke-width="1.5"/><text x="696" y="167" fill="#e8eaf0">sportif</text>
  </g>
  <!-- cross arrows: "cat" query attends to "chat" key -->
  <line x1="595" y1="78" x2="595" y2="144" stroke="#50c878" stroke-width="3" marker-end="url(#arrcxgr)"/>
  <line x1="595" y1="78" x2="515" y2="144" stroke="#8892a4" stroke-width="1.2" opacity="0.5" marker-end="url(#arrcxg)"/>
  <line x1="595" y1="78" x2="690" y2="144" stroke="#8892a4" stroke-width="1.2" opacity="0.5" marker-end="url(#arrcxg)"/>
  <text x="615" y="210" fill="#8892a4" font-size="12" text-anchor="middle">Q from one sequence, K and V</text>
  <text x="615" y="228" fill="#8892a4" font-size="12" text-anchor="middle">from another ("cat" finds "chat")</text>
  <defs>
    <marker id="arrcx" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#f5a623"/></marker>
    <marker id="arrcxg" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#8892a4"/></marker>
    <marker id="arrcxgr" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#50c878"/></marker>
  </defs>
</svg>
</div>

In self-attention, one sequence looks at itself. In cross-attention, one sequence asks questions while another supplies the keys and values. This is useful when text retrieves from an image, an audio stream, a database, or a source sentence in translation.
