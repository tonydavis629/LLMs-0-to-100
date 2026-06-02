<!-- .slide: id="tensor-shapes" -->

## Review: The Shape of the Data

Before stacking heads and adding positions, picture what a batch of text looks like inside the model. Everything flows as one tensor of shape **(B, seq_len, d_model)**.

<div style="text-align: center; margin: 6px 0;">
<svg viewBox="0 0 900 380" width="100%" style="max-height: 350px;">
  <text x="450" y="22" fill="#8892a4" font-size="12" text-anchor="middle">a batch of sequences, stored as one tensor</text>
  <g stroke="#46506e" stroke-width="1.2">
    <line x1="210" y1="120" x2="298" y2="60"/>
    <line x1="510" y1="120" x2="598" y2="60" marker-end="url(#arrdim)"/>
    <line x1="510" y1="320" x2="598" y2="260"/>
  </g>
  <rect x="298" y="60" width="300" height="200" rx="3" fill="rgba(74,158,255,0.05)" stroke="#2a3450" stroke-width="1.2"/>
  <rect x="254" y="90" width="300" height="200" rx="3" fill="rgba(74,158,255,0.07)" stroke="#2a3450" stroke-width="1.2"/>
  <rect x="210" y="120" width="300" height="200" rx="3" fill="rgba(74,158,255,0.10)" stroke="#4a9eff" stroke-width="1.6"/>
  <g stroke="rgba(74,158,255,0.18)" stroke-width="1">
    <line x1="260" y1="120" x2="260" y2="320"/><line x1="310" y1="120" x2="310" y2="320"/>
    <line x1="360" y1="120" x2="360" y2="320"/><line x1="410" y1="120" x2="410" y2="320"/>
    <line x1="460" y1="120" x2="460" y2="320"/>
  </g>
  <g stroke="#4a9eff" stroke-width="1">
    <line x1="210" y1="160" x2="510" y2="160"/><line x1="210" y1="200" x2="510" y2="200"/>
    <line x1="210" y1="240" x2="510" y2="240"/><line x1="210" y1="280" x2="510" y2="280"/>
  </g>
  <rect x="210" y="160" width="300" height="40" fill="rgba(245,166,35,0.14)" stroke="#f5a623" stroke-width="1.8"/>
  <g font-size="13" fill="#e8eaf0" font-weight="600">
    <text x="222" y="145">the</text>
    <text x="222" y="185">cat</text>
    <text x="222" y="225">sat</text>
    <text x="222" y="265">on</text>
    <text x="222" y="305">mat</text>
  </g>
  <text x="612" y="64" fill="#4a9eff" font-size="13" font-weight="600">batch B</text>
  <text x="612" y="80" fill="#8892a4" font-size="11">sequences in parallel</text>
  <g stroke="#8892a4" stroke-width="1.2">
    <line x1="190" y1="120" x2="184" y2="120"/><line x1="184" y1="120" x2="184" y2="320"/><line x1="184" y1="320" x2="190" y2="320"/>
  </g>
  <text x="150" y="226" fill="#8892a4" font-size="12" text-anchor="middle" transform="rotate(-90 150 226)">seq_len (tokens)</text>
  <g stroke="#8892a4" stroke-width="1.2">
    <line x1="210" y1="334" x2="210" y2="340"/><line x1="210" y1="340" x2="510" y2="340"/><line x1="510" y1="340" x2="510" y2="334"/>
  </g>
  <text x="360" y="360" fill="#8892a4" font-size="12" text-anchor="middle">d_model features per token</text>
  <line x1="510" y1="180" x2="624" y2="180" stroke="#f5a623" stroke-width="1.4" marker-end="url(#arrsec)"/>
  <text x="630" y="176" fill="#f5a623" font-size="12">one row = one token,</text>
  <text x="630" y="194" fill="#f5a623" font-size="12">a vector of length d_model</text>
  <defs>
    <marker id="arrdim" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#46506e"/></marker>
    <marker id="arrsec" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#f5a623"/></marker>
  </defs>
</svg>
</div>

:::columns cols="3" gap="20px"
**batch (B)**

Several sequences run through the model in parallel. They never attend to each other.
+++
**seq_len (tokens)**

One row per token. **Positional encodings** add a signal along this axis so order is preserved.
+++
**d_model (features)**

Each token's feature vector. **Multi-head attention** splits this axis into parallel heads.
:::
