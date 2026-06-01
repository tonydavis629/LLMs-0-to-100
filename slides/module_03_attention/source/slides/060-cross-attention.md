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
  <path d="M107 132 Q151 168 195 132" fill="none" stroke="#8892a4" stroke-width="1.2" opacity="0.6" marker-end="url(#arrcxg)"/>
  <text x="195" y="200" fill="#8892a4" font-size="12" text-anchor="middle">Q, K, V all from the</text>
  <text x="195" y="218" fill="#8892a4" font-size="12" text-anchor="middle">same sequence</text>
  <!-- divider -->
  <line x1="410" y1="40" x2="410" y2="220" stroke="#2a3450" stroke-width="1"/>
  <!-- Cross-attention panel -->
  <text x="615" y="22" fill="#50c878" font-size="14" text-anchor="middle" font-weight="600">Cross-Attention</text>
  <!-- decoder row (queries) -->
  <text x="455" y="62" fill="#8892a4" font-size="11" text-anchor="end">decoder (Q)</text>
  <g font-size="13" text-anchor="middle" font-weight="600">
    <rect x="470" y="46" width="74" height="32" rx="4" fill="#0d1225" stroke="#50c878" stroke-width="1.5"/><text x="507" y="67" fill="#e8eaf0">the</text>
    <rect x="558" y="46" width="74" height="32" rx="4" fill="#0d1225" stroke="#50c878" stroke-width="2.5"/><text x="595" y="67" fill="#e8eaf0">cat</text>
  </g>
  <!-- encoder row (keys/values) -->
  <text x="455" y="162" fill="#8892a4" font-size="11" text-anchor="end">encoder (K, V)</text>
  <g font-size="13" text-anchor="middle" font-weight="600">
    <rect x="470" y="146" width="74" height="32" rx="4" fill="#0d1225" stroke="#f5a623" stroke-width="1.5"/><text x="507" y="167" fill="#e8eaf0">le</text>
    <rect x="558" y="146" width="74" height="32" rx="4" fill="#0d1225" stroke="#f5a623" stroke-width="2.5"/><text x="595" y="167" fill="#e8eaf0">chat</text>
    <rect x="646" y="146" width="100" height="32" rx="4" fill="#0d1225" stroke="#f5a623" stroke-width="1.5"/><text x="696" y="167" fill="#e8eaf0">sportif</text>
  </g>
  <!-- cross arrows: "cat" query attends to "chat" key -->
  <line x1="595" y1="78" x2="595" y2="144" stroke="#50c878" stroke-width="3" marker-end="url(#arrcxgr)"/>
  <line x1="595" y1="78" x2="515" y2="144" stroke="#8892a4" stroke-width="1.2" opacity="0.5" marker-end="url(#arrcxg)"/>
  <line x1="595" y1="78" x2="690" y2="144" stroke="#8892a4" stroke-width="1.2" opacity="0.5" marker-end="url(#arrcxg)"/>
  <text x="615" y="210" fill="#8892a4" font-size="12" text-anchor="middle">Q from the decoder, K and V</text>
  <text x="615" y="228" fill="#8892a4" font-size="12" text-anchor="middle">from the encoder ("cat" finds "chat")</text>
  <defs>
    <marker id="arrcx" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#f5a623"/></marker>
    <marker id="arrcxg" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#8892a4"/></marker>
    <marker id="arrcxgr" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#50c878"/></marker>
  </defs>
</svg>
</div>

In self-attention a sequence looks at itself; in cross-attention one sequence selects information from another. Cross-attention is how a decoder pulls relevant content from an encoder.

---

<!-- .slide: id="encoder-decoder" -->

## The Encoder-Decoder Transformer

The original Transformer (Vaswani et al.) combined both: the encoder uses self-attention; the decoder uses masked self-attention **and** cross-attention back into the encoder.

<div style="text-align: center; margin: 4px 0;">
<svg viewBox="0 0 760 320" width="100%" style="max-height: 300px;">
  <!-- Encoder column -->
  <rect x="60" y="40" width="230" height="210" rx="8" fill="none" stroke="#4a9eff" stroke-width="1.5" stroke-dasharray="5,3"/>
  <text x="175" y="32" fill="#4a9eff" font-size="13" text-anchor="middle" font-weight="600">Encoder (N&times;)</text>
  <rect x="90" y="170" width="170" height="36" rx="5" fill="rgba(74,158,255,0.12)" stroke="#4a9eff" stroke-width="1.5"/>
  <text x="175" y="192" fill="#e8eaf0" font-size="12" text-anchor="middle">self-attention</text>
  <rect x="90" y="118" width="170" height="36" rx="5" fill="rgba(245,166,35,0.12)" stroke="#f5a623" stroke-width="1.5"/>
  <text x="175" y="140" fill="#e8eaf0" font-size="12" text-anchor="middle">feed-forward (MLP)</text>
  <rect x="90" y="222" width="170" height="22" rx="4" fill="#0d1225" stroke="#8892a4" stroke-width="1"/>
  <text x="175" y="238" fill="#8892a4" font-size="11" text-anchor="middle">input embeddings + pos</text>
  <g stroke="#8892a4" stroke-width="1.2" marker-end="url(#arred)">
    <line x1="175" y1="222" x2="175" y2="208"/><line x1="175" y1="170" x2="175" y2="156"/><line x1="175" y1="118" x2="175" y2="104"/>
  </g>
  <!-- Decoder column -->
  <rect x="470" y="40" width="230" height="250" rx="8" fill="none" stroke="#50c878" stroke-width="1.5" stroke-dasharray="5,3"/>
  <text x="585" y="32" fill="#50c878" font-size="13" text-anchor="middle" font-weight="600">Decoder (N&times;)</text>
  <rect x="500" y="222" width="170" height="34" rx="5" fill="rgba(74,158,255,0.12)" stroke="#4a9eff" stroke-width="1.5"/>
  <text x="585" y="243" fill="#e8eaf0" font-size="11" text-anchor="middle">masked self-attention</text>
  <rect x="500" y="170" width="170" height="34" rx="5" fill="rgba(80,200,120,0.16)" stroke="#50c878" stroke-width="1.5"/>
  <text x="585" y="191" fill="#e8eaf0" font-size="11" text-anchor="middle">cross-attention</text>
  <rect x="500" y="118" width="170" height="34" rx="5" fill="rgba(245,166,35,0.12)" stroke="#f5a623" stroke-width="1.5"/>
  <text x="585" y="139" fill="#e8eaf0" font-size="11" text-anchor="middle">feed-forward (MLP)</text>
  <rect x="500" y="272" width="170" height="20" rx="4" fill="#0d1225" stroke="#8892a4" stroke-width="1"/>
  <text x="585" y="287" fill="#8892a4" font-size="10" text-anchor="middle">output embeddings + pos</text>
  <rect x="500" y="74" width="170" height="30" rx="5" fill="#0d1225" stroke="#8892a4" stroke-width="1"/>
  <text x="585" y="93" fill="#8892a4" font-size="11" text-anchor="middle">linear + softmax</text>
  <g stroke="#8892a4" stroke-width="1.2" marker-end="url(#arred)">
    <line x1="585" y1="272" x2="585" y2="258"/><line x1="585" y1="222" x2="585" y2="206"/><line x1="585" y1="170" x2="585" y2="154"/><line x1="585" y1="118" x2="585" y2="106"/>
  </g>
  <!-- cross-attention link from encoder to decoder -->
  <path d="M260 136 Q380 136 380 187 Q380 187 500 187" fill="none" stroke="#50c878" stroke-width="2" stroke-dasharray="4,3" marker-end="url(#arredgr)"/>
  <text x="380" y="128" fill="#50c878" font-size="11" text-anchor="middle">K, V to cross-attention</text>
  <defs>
    <marker id="arred" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#8892a4"/></marker>
    <marker id="arredgr" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#50c878"/></marker>
  </defs>
</svg>
</div>

Modern LLMs (GPT, LLaMA) keep only the decoder with causal self-attention. The encoder-decoder pattern remains common in translation and speech.
