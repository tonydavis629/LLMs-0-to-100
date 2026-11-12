:::divider id="divider-science" title="LLMs for Science"
:::

---

<!-- .slide: id="sci-examples" -->

## Where LLMs Are Used in Science

<div class="card-grid cols-4">
<div class="card"><h4>Literature</h4><p>Deep research over papers: find the prior work, read it, and summarize what is known.</p></div>
<div class="card"><h4>Protein language models</h4><p>ESM-2 (2023) is a transformer trained to predict masked amino acids. The tokens are amino acids, and its representations are enough to predict a protein's <strong>3D structure</strong>.</p></div>
<div class="card"><h4>Program search</h4><p>FunSearch (2023): an LLM writes candidate programs and an evaluator scores them. It found cap sets <strong>larger than any known before</strong>.</p></div>
<div class="card"><h4>Formal mathematics</h4><p>AlphaProof (2024) writes proofs in Lean, and Lean checks them. With AlphaGeometry 2 it solved <strong>4 of 6</strong> problems at the 2024 Olympiad.</p></div>
</div>

---

<!-- .slide: id="sci-loop" -->

## Generate and Verify

<div style="text-align: center; margin: 6px 0;">
<svg viewBox="0 0 960 250" width="100%" style="max-height: 340px;">
  <defs>
    <marker id="sci-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#8892a4"/>
    </marker>
    <marker id="sci-arrow-blue" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#4a9eff"/>
    </marker>
  </defs>
  <g stroke="#2a3450" stroke-width="1.5" fill="rgba(74,158,255,0.05)">
    <rect x="40" y="90" width="250" height="74" rx="8"/>
    <rect x="370" y="90" width="230" height="74" rx="8" stroke="#f5a623"/>
    <rect x="680" y="90" width="240" height="74" rx="8"/>
  </g>
  <g text-anchor="middle" font-size="16" font-weight="600" fill="#4a9eff">
    <text x="165" y="122">LLM proposes</text>
    <text x="485" y="122" fill="#f5a623">Verifier scores</text>
    <text x="800" y="122">Keep the best</text>
  </g>
  <g text-anchor="middle" font-size="12" fill="#8892a4">
    <text x="165" y="146">a program, a proof, a hypothesis</text>
    <text x="485" y="146">run it, check it, or test it in a lab</text>
    <text x="800" y="146">a pool of top candidates</text>
  </g>
  <g stroke="#8892a4" stroke-width="1.5" marker-end="url(#sci-arrow)">
    <line x1="290" y1="127" x2="366" y2="127"/>
    <line x1="600" y1="127" x2="676" y2="127"/>
  </g>
  <path d="M 800 88 L 800 42 Q 800 30 788 30 L 177 30 Q 165 30 165 42 L 165 84" stroke="#4a9eff" stroke-width="1.5" fill="none" marker-end="url(#sci-arrow-blue)"/>
  <text x="482" y="20" text-anchor="middle" font-size="12.5" fill="#8892a4">the best candidates go into the next prompt</text>
  <g font-size="12.5" fill="#8892a4" text-anchor="middle">
    <text x="485" y="196">a program evaluator or proof checker: seconds per check</text>
    <text x="485" y="216">a lab experiment: days or months per check</text>
  </g>
</svg>
</div>

This is the coding agent's loop with a different verifier. FunSearch and AlphaProof work because a computer can check each candidate. For hypotheses that need an experiment, such as those from Google's AI co-scientist (2025), people still run the check, and the loop is slow. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="sci-handoff" -->

## Looking Ahead

Every application in this module used the same component: a transformer with fixed weights that writes one token after another.

<div class="metric-box">
<p>Scientific work strains that design. The model does not learn from the result of an experiment unless someone retrains it, and a hard problem gets the same computation per token as an easy one. Module 12 looks at research that drops these assumptions.</p>
</div>
