:::divider id="divider-masks" title="Attention Masks" sub="Controlling what each token can see"
:::

---

<!-- .slide: id="padding-masks" -->

## Padding Masks

In a batch, sequences have different lengths, so shorter ones are padded with a special token. A **padding mask** stops attention from flowing to those fake positions.

<div style="text-align: center; margin: 8px 0;">
<svg viewBox="0 0 820 190" width="100%" style="max-height: 180px;">
  <text x="150" y="24" fill="#8892a4" font-size="12">A query attends over the keys (one padded sequence):</text>
  <g font-size="13" text-anchor="middle" font-weight="600">
    <rect x="150" y="36" width="86" height="32" rx="4" fill="#0d1225" stroke="#4a9eff" stroke-width="1.5"/><text x="193" y="57" fill="#e8eaf0">the</text>
    <rect x="246" y="36" width="86" height="32" rx="4" fill="#0d1225" stroke="#4a9eff" stroke-width="1.5"/><text x="289" y="57" fill="#e8eaf0">cat</text>
    <rect x="342" y="36" width="86" height="32" rx="4" fill="#0d1225" stroke="#4a9eff" stroke-width="1.5"/><text x="385" y="57" fill="#e8eaf0">sat</text>
    <rect x="438" y="36" width="86" height="32" rx="4" fill="#0d1225" stroke="#e74c3c" stroke-width="1.5"/><text x="481" y="57" fill="#8892a4">[PAD]</text>
    <rect x="534" y="36" width="86" height="32" rx="4" fill="#0d1225" stroke="#e74c3c" stroke-width="1.5"/><text x="577" y="57" fill="#8892a4">[PAD]</text>
  </g>
  <!-- mask row -->
  <text x="138" y="98" fill="#8892a4" font-size="12" text-anchor="end">add mask</text>
  <g font-size="12" text-anchor="middle" font-family="monospace">
    <text x="193" y="98" fill="#3fb950">0</text>
    <text x="289" y="98" fill="#3fb950">0</text>
    <text x="385" y="98" fill="#3fb950">0</text>
    <text x="481" y="98" fill="#e74c3c">-inf</text>
    <text x="577" y="98" fill="#e74c3c">-inf</text>
  </g>
  <!-- weights row -->
  <text x="138" y="145" fill="#8892a4" font-size="12" text-anchor="end">after softmax</text>
  <g font-size="13" text-anchor="middle">
    <rect x="150" y="120" width="86" height="40" rx="4" fill="rgba(74,158,255,0.45)"/><text x="193" y="145" fill="#e8eaf0">0.41</text>
    <rect x="246" y="120" width="86" height="40" rx="4" fill="rgba(74,158,255,0.38)"/><text x="289" y="145" fill="#e8eaf0">0.34</text>
    <rect x="342" y="120" width="86" height="40" rx="4" fill="rgba(74,158,255,0.28)"/><text x="385" y="145" fill="#e8eaf0">0.25</text>
    <rect x="438" y="120" width="86" height="40" rx="4" fill="rgba(231,76,60,0.10)" stroke="#e74c3c" stroke-width="1" stroke-dasharray="3,2"/><text x="481" y="145" fill="#8892a4">0.00</text>
    <rect x="534" y="120" width="86" height="40" rx="4" fill="rgba(231,76,60,0.10)" stroke="#e74c3c" stroke-width="1" stroke-dasharray="3,2"/><text x="577" y="145" fill="#8892a4">0.00</text>
  </g>
</svg>
</div>

Setting padded positions to $-\infty$ before softmax sends their weight to exactly zero. Padding masks are a practical necessity for batched training.

---

<!-- .slide: id="causal-masks" -->

## Causal Masks

A **causal mask** stops each token from attending to **future** tokens. This is essential for autoregressive generation (GPT-style). The allowed region is a lower triangle:

<div style="text-align: center; margin: 6px 0;">
<svg viewBox="0 0 560 320" width="100%" style="max-height: 300px;">
  <text x="280" y="18" fill="#8892a4" font-size="12" text-anchor="middle">rows = query token &middot; columns = key token</text>
  <!-- column labels -->
  <g font-size="12" text-anchor="middle" fill="#f5a623" font-weight="600">
    <text x="170" y="48">the</text><text x="234" y="48">cat</text><text x="298" y="48">sat</text><text x="362" y="48">on</text><text x="426" y="48">mat</text>
  </g>
  <!-- row labels + grid -->
  <g font-size="12" fill="#4a9eff" font-weight="600">
    <text x="100" y="78" text-anchor="end">the</text>
    <text x="100" y="122" text-anchor="end">cat</text>
    <text x="100" y="166" text-anchor="end">sat</text>
    <text x="100" y="210" text-anchor="end">on</text>
    <text x="100" y="254" text-anchor="end">mat</text>
  </g>
  <!-- cells: 5x5, cell size 56, origin x=140 y=56 -->
  <!-- allowed = lower triangle (j<=i): blue; blocked: dark with -inf -->
  <g font-size="11" text-anchor="middle">
    <!-- row 0 (the): only col0 -->
    <rect x="140" y="56" width="60" height="40" rx="3" fill="rgba(74,158,255,0.40)" stroke="#2a3450"/><text x="170" y="81" fill="#e8eaf0">&#10003;</text>
    <rect x="204" y="56" width="60" height="40" rx="3" fill="#0d1225" stroke="#2a3450"/><text x="234" y="81" fill="#46506a">-inf</text>
    <rect x="268" y="56" width="60" height="40" rx="3" fill="#0d1225" stroke="#2a3450"/><text x="298" y="81" fill="#46506a">-inf</text>
    <rect x="332" y="56" width="60" height="40" rx="3" fill="#0d1225" stroke="#2a3450"/><text x="362" y="81" fill="#46506a">-inf</text>
    <rect x="396" y="56" width="60" height="40" rx="3" fill="#0d1225" stroke="#2a3450"/><text x="426" y="81" fill="#46506a">-inf</text>
    <!-- row 1 (cat): col0,1 -->
    <rect x="140" y="100" width="60" height="40" rx="3" fill="rgba(74,158,255,0.40)" stroke="#2a3450"/><text x="170" y="125" fill="#e8eaf0">&#10003;</text>
    <rect x="204" y="100" width="60" height="40" rx="3" fill="rgba(74,158,255,0.40)" stroke="#2a3450"/><text x="234" y="125" fill="#e8eaf0">&#10003;</text>
    <rect x="268" y="100" width="60" height="40" rx="3" fill="#0d1225" stroke="#2a3450"/><text x="298" y="125" fill="#46506a">-inf</text>
    <rect x="332" y="100" width="60" height="40" rx="3" fill="#0d1225" stroke="#2a3450"/><text x="362" y="125" fill="#46506a">-inf</text>
    <rect x="396" y="100" width="60" height="40" rx="3" fill="#0d1225" stroke="#2a3450"/><text x="426" y="125" fill="#46506a">-inf</text>
    <!-- row 2 (sat): col0,1,2 -->
    <rect x="140" y="144" width="60" height="40" rx="3" fill="rgba(74,158,255,0.40)" stroke="#2a3450"/><text x="170" y="169" fill="#e8eaf0">&#10003;</text>
    <rect x="204" y="144" width="60" height="40" rx="3" fill="rgba(74,158,255,0.40)" stroke="#2a3450"/><text x="234" y="169" fill="#e8eaf0">&#10003;</text>
    <rect x="268" y="144" width="60" height="40" rx="3" fill="rgba(74,158,255,0.40)" stroke="#2a3450"/><text x="298" y="169" fill="#e8eaf0">&#10003;</text>
    <rect x="332" y="144" width="60" height="40" rx="3" fill="#0d1225" stroke="#2a3450"/><text x="362" y="169" fill="#46506a">-inf</text>
    <rect x="396" y="144" width="60" height="40" rx="3" fill="#0d1225" stroke="#2a3450"/><text x="426" y="169" fill="#46506a">-inf</text>
    <!-- row 3 (on): col0-3 -->
    <rect x="140" y="188" width="60" height="40" rx="3" fill="rgba(74,158,255,0.40)" stroke="#2a3450"/><text x="170" y="213" fill="#e8eaf0">&#10003;</text>
    <rect x="204" y="188" width="60" height="40" rx="3" fill="rgba(74,158,255,0.40)" stroke="#2a3450"/><text x="234" y="213" fill="#e8eaf0">&#10003;</text>
    <rect x="268" y="188" width="60" height="40" rx="3" fill="rgba(74,158,255,0.40)" stroke="#2a3450"/><text x="298" y="213" fill="#e8eaf0">&#10003;</text>
    <rect x="332" y="188" width="60" height="40" rx="3" fill="rgba(74,158,255,0.40)" stroke="#2a3450"/><text x="362" y="213" fill="#e8eaf0">&#10003;</text>
    <rect x="396" y="188" width="60" height="40" rx="3" fill="#0d1225" stroke="#2a3450"/><text x="426" y="213" fill="#46506a">-inf</text>
    <!-- row 4 (mat): all -->
    <rect x="140" y="232" width="60" height="40" rx="3" fill="rgba(74,158,255,0.40)" stroke="#2a3450"/><text x="170" y="257" fill="#e8eaf0">&#10003;</text>
    <rect x="204" y="232" width="60" height="40" rx="3" fill="rgba(74,158,255,0.40)" stroke="#2a3450"/><text x="234" y="257" fill="#e8eaf0">&#10003;</text>
    <rect x="268" y="232" width="60" height="40" rx="3" fill="rgba(74,158,255,0.40)" stroke="#2a3450"/><text x="298" y="257" fill="#e8eaf0">&#10003;</text>
    <rect x="332" y="232" width="60" height="40" rx="3" fill="rgba(74,158,255,0.40)" stroke="#2a3450"/><text x="362" y="257" fill="#e8eaf0">&#10003;</text>
    <rect x="396" y="232" width="60" height="40" rx="3" fill="rgba(74,158,255,0.40)" stroke="#2a3450"/><text x="426" y="257" fill="#e8eaf0">&#10003;</text>
  </g>
  <text x="298" y="298" fill="#8892a4" font-size="12" text-anchor="middle">each token sees only itself and the tokens before it</text>
</svg>
</div>

---

<!-- .slide: id="why-causal" -->

## Why Causal Masking?

In a language model, we predict the next token given all previous tokens:

$$P(w_t \mid w_1, w_2, \ldots, w_{t-1})$$

:::note
If token $t$ could see token $t+1$, the model would be **cheating** during training &mdash; it could copy the answer instead of learning to predict. Causal masking enforces the same information constraint during training that the model will face at inference, when future tokens do not exist yet.
:::
