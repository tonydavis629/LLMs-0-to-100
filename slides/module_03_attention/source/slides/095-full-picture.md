<!-- .slide: id="full-picture" -->

## Putting It Together

Every piece from this module, in order: tokens in, next-token probabilities out.

<div style="text-align: center; margin: 4px 0;">
<svg viewBox="0 30 880 230" width="100%" style="max-height: 330px;">
  <!-- block outline -->
  <rect x="246" y="40" width="404" height="200" rx="8" fill="rgba(136,146,164,0.05)" stroke="#8892a4" stroke-width="1.3" stroke-dasharray="6 4"/>
  <text x="260" y="60" fill="#8892a4" font-size="12" font-weight="600">transformer block</text>
  <text x="636" y="60" fill="#c792ea" font-size="13" font-weight="600" text-anchor="end">× N</text>
  <!-- input side -->
  <rect x="4" y="128" width="62" height="44" rx="5" fill="#0d1225" stroke="#e8eaf0" stroke-width="1.5"/>
  <text x="35" y="148" fill="#e8eaf0" font-size="12" text-anchor="middle" font-weight="600">tokens</text>
  <text x="35" y="164" fill="#8892a4" font-size="10" text-anchor="middle">n ids</text>
  <rect x="84" y="128" width="80" height="44" rx="5" fill="rgba(232,234,240,0.08)" stroke="#e8eaf0" stroke-width="1.5"/>
  <text x="124" y="148" fill="#e8eaf0" font-size="12" text-anchor="middle" font-weight="600">embed</text>
  <text x="124" y="164" fill="#8892a4" font-size="10" text-anchor="middle">n × 512</text>
  <rect x="164" y="196" width="72" height="40" rx="5" fill="rgba(199,146,234,0.12)" stroke="#c792ea" stroke-width="1.5"/>
  <text x="200" y="213" fill="#c792ea" font-size="11" text-anchor="middle" font-weight="600">positional</text>
  <text x="200" y="228" fill="#c792ea" font-size="11" text-anchor="middle" font-weight="600">encoding</text>
  <circle cx="200" cy="150" r="12" fill="#0d1225" stroke="#e8eaf0" stroke-width="1.5"/>
  <text x="200" y="156" fill="#e8eaf0" font-size="17" text-anchor="middle" font-weight="600">+</text>
  <!-- main residual line -->
  <g stroke="#e8eaf0" stroke-width="2" marker-end="url(#arrfpw)">
    <line x1="66" y1="150" x2="82" y2="150"/>
    <line x1="164" y1="150" x2="186" y2="150"/>
    <line x1="212" y1="150" x2="414" y2="150"/>
    <line x1="440" y1="150" x2="604" y2="150"/>
    <line x1="630" y1="150" x2="662" y2="150"/>
  </g>
  <line x1="200" y1="196" x2="200" y2="164" stroke="#c792ea" stroke-width="1.5" marker-end="url(#arrfpp)"/>
  <text x="430" y="194" fill="#8892a4" font-size="11" text-anchor="middle">residual: x + sublayer(x)</text>
  <!-- attention sublayer -->
  <path d="M262 150 L262 90 L270 90" fill="none" stroke="#4a9eff" stroke-width="1.5" marker-end="url(#arrfpb)"/>
  <rect x="272" y="66" width="140" height="62" rx="6" fill="rgba(74,158,255,0.12)" stroke="#4a9eff" stroke-width="1.5"/>
  <text x="342" y="84" fill="#4a9eff" font-size="12" text-anchor="middle" font-weight="600">Multi-Head Attention</text>
  <text x="342" y="102" fill="#e8eaf0" font-size="11" text-anchor="middle">softmax(QKᵀ/√d<tspan dy="3" font-size="8">k</tspan><tspan dy="-3">)V</tspan></text>
  <text x="342" y="118" fill="#8892a4" font-size="10" text-anchor="middle">8 heads, causal mask</text>
  <path d="M412 90 L428 90 L428 136" fill="none" stroke="#4a9eff" stroke-width="1.5" marker-end="url(#arrfpb)"/>
  <circle cx="428" cy="150" r="12" fill="#0d1225" stroke="#e8eaf0" stroke-width="1.5"/>
  <text x="428" y="156" fill="#e8eaf0" font-size="17" text-anchor="middle" font-weight="600">+</text>
  <text x="342" y="216" fill="#8892a4" font-size="11" text-anchor="middle">tokens share information</text>
  <!-- MLP sublayer -->
  <path d="M452 150 L452 90 L460 90" fill="none" stroke="#f5a623" stroke-width="1.5" marker-end="url(#arrfpo)"/>
  <rect x="462" y="66" width="130" height="62" rx="6" fill="rgba(245,166,35,0.12)" stroke="#f5a623" stroke-width="1.5"/>
  <text x="527" y="84" fill="#f5a623" font-size="13" text-anchor="middle" font-weight="600">MLP</text>
  <text x="527" y="102" fill="#e8eaf0" font-size="11" text-anchor="middle">512 → 2048 → 512</text>
  <text x="527" y="118" fill="#8892a4" font-size="10" text-anchor="middle">same weights per token</text>
  <path d="M592 90 L618 90 L618 136" fill="none" stroke="#f5a623" stroke-width="1.5" marker-end="url(#arrfpo)"/>
  <circle cx="618" cy="150" r="12" fill="#0d1225" stroke="#e8eaf0" stroke-width="1.5"/>
  <text x="618" y="156" fill="#e8eaf0" font-size="17" text-anchor="middle" font-weight="600">+</text>
  <text x="527" y="216" fill="#8892a4" font-size="11" text-anchor="middle">each token on its own</text>
  <!-- output side -->
  <rect x="664" y="128" width="72" height="44" rx="5" fill="rgba(232,234,240,0.08)" stroke="#e8eaf0" stroke-width="1.5"/>
  <text x="700" y="148" fill="#e8eaf0" font-size="12" text-anchor="middle" font-weight="600">linear</text>
  <text x="700" y="164" fill="#8892a4" font-size="10" text-anchor="middle">512 → vocab</text>
  <line x1="736" y1="150" x2="752" y2="150" stroke="#e8eaf0" stroke-width="2" marker-end="url(#arrfpw)"/>
  <rect x="754" y="128" width="64" height="44" rx="5" fill="rgba(80,200,120,0.12)" stroke="#50c878" stroke-width="1.5"/>
  <text x="786" y="148" fill="#50c878" font-size="12" text-anchor="middle" font-weight="600">softmax</text>
  <text x="786" y="164" fill="#8892a4" font-size="10" text-anchor="middle">logits → p</text>
  <line x1="818" y1="150" x2="832" y2="150" stroke="#e8eaf0" stroke-width="2" marker-end="url(#arrfpw)"/>
  <g fill="#50c878">
    <rect x="836" y="160" width="6" height="12"/><rect x="844" y="128" width="6" height="44"/>
    <rect x="852" y="152" width="6" height="20"/><rect x="860" y="164" width="6" height="8"/>
    <rect x="868" y="167" width="6" height="5"/>
  </g>
  <line x1="834" y1="172" x2="876" y2="172" stroke="#8892a4" stroke-width="1"/>
  <text x="855" y="188" fill="#8892a4" font-size="10" text-anchor="middle">P(next</text>
  <text x="855" y="200" fill="#8892a4" font-size="10" text-anchor="middle">token)</text>
  <defs>
    <marker id="arrfpw" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 Z" fill="#e8eaf0"/></marker>
    <marker id="arrfpb" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#4a9eff"/></marker>
    <marker id="arrfpo" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#f5a623"/></marker>
    <marker id="arrfpp" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#c792ea"/></marker>
  </defs>
</svg>
</div>

Softmax appears twice: inside every attention head to turn scores into weights, and once at the end to turn logits into a distribution over the vocabulary. Layer normalization is omitted here; Module 4 builds the full block.
