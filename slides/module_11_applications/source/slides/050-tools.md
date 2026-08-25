:::divider id="divider-tools" title="Tool Use"
:::

---

<!-- .slide: id="tools-mechanism" -->

## How a Tool Call Works

<div class="card-grid cols-4">
<div class="card"><h4>1. Emit</h4><p>The model generates a <strong>structured call</strong> in its output: a tool name and arguments, as tokens like any others.</p></div>
<div class="card"><h4>2. Pause</h4><p>The runtime detects the call and <strong>stops generation</strong>.</p></div>
<div class="card"><h4>3. Execute</h4><p>Ordinary software runs the call: a search, a database query, a calculation. <strong>The model never executes anything itself.</strong></p></div>
<div class="card"><h4>4. Resume</h4><p>The result is appended to the context <strong>as tokens</strong>, and the model continues generating, now conditioned on it.</p></div>
</div>

The model is still just predicting next tokens. Tool use is a protocol layered on top of sampling, enforced by the runtime, not a new capability in the weights. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="tools-diagram" -->

## The Round Trip, Drawn

<div style="text-align: center; margin: 8px 0;">
<svg viewBox="0 0 960 320" width="100%" style="max-height: 460px;">
  <defs>
    <marker id="tools-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#8892a4"/>
    </marker>
  </defs>
  <text x="250" y="28" text-anchor="middle" fill="#e8eaf0" font-size="17" font-weight="600">The model (frozen weights)</text>
  <text x="710" y="28" text-anchor="middle" fill="#e8eaf0" font-size="17" font-weight="600">The runtime (ordinary software)</text>
  <g stroke="#2a3450" stroke-width="1.5" fill="rgba(74,158,255,0.05)">
    <rect x="110" y="50" width="280" height="68" rx="8"/>
    <rect x="570" y="135" width="280" height="68" rx="8" stroke="#f5a623"/>
    <rect x="110" y="235" width="280" height="68" rx="8"/>
  </g>
  <g text-anchor="middle" font-size="16" font-weight="600" fill="#4a9eff">
    <text x="250" y="78">1. Emit</text>
    <text x="710" y="163" fill="#f5a623">2. Pause, 3. Execute</text>
    <text x="250" y="263">4. Resume</text>
  </g>
  <g text-anchor="middle" font-size="12" fill="#8892a4">
    <text x="250" y="101" font-family="monospace">search("PX-220 error codes")</text>
    <text x="710" y="186">generation stops; software runs the call</text>
    <text x="250" y="286">the next tokens condition on the result</text>
  </g>
  <g stroke="#8892a4" stroke-width="1.5" marker-end="url(#tools-arrow)">
    <line x1="390" y1="95" x2="566" y2="145"/>
    <line x1="570" y1="192" x2="394" y2="243"/>
  </g>
  <g font-size="12.5" fill="#8892a4" text-anchor="middle">
    <text x="615" y="110">a structured call, as tokens</text>
    <text x="335" y="220">the result, appended as tokens</text>
  </g>
</svg>
</div>

The four cards from the last slide, as one round trip: tokens out, ordinary software in the middle, tokens back in. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="tools-trained" -->

## Where the Ability Comes From

Emitting well-formed calls at the right moments is **trained in, not innate.**

:::columns cols="2" gap="34px"
**The recipe**

- An SFT pass (Module 6) over examples: when a question needs a tool, how to format the call, how to read the result
- The chat template grows a **tool role** alongside system, user, assistant
+++
**The research version**

- Toolformer (Schick et al., 2023): the model annotates its own training text with API calls
- Keep the calls that reduce loss on the following tokens; finetune on the result
- The model teaches itself where a calculator would have helped
:::

Prompted base models imitate the format; finetuned models are reliable enough to build on. The difference is a dataset, not an architecture. <!-- .element: class="text-lg" style="margin-top: 10px;" -->

---

<!-- .slide: id="tools-set" -->

## The Standard Toolbox

<div class="card-grid cols-4">
<div class="card"><h4>Calculator</h4><p>Digits are just tokens (Module 4), so models are unreliable at arithmetic. A calculator is exact for free.</p></div>
<div class="card"><h4>Web search</h4><p>Fresh facts past the training cutoff. Retrieval, pointed at the live internet.</p></div>
<div class="card"><h4>Code execution</h4><p>The universal escape hatch: anything computable becomes a tool call away.</p></div>
<div class="card"><h4>Retrieval</h4><p>The RAG pipeline, packaged as just another tool the model can decide to invoke.</p></div>
</div>

**Tools let the model outsource what next-token prediction is bad at:** precise computation, current knowledge, and side effects move into software that is good at them. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="tools-apis" -->

## Function Calling

Every major API ships the same shape:

<div class="metric-box">
<p>The developer sends <strong>JSON schemas</strong> describing the available tools. The model returns a call matching one of the schemas. <strong>Constrained decoding</strong> masks invalid tokens during sampling, so the call is guaranteed to parse: the schema is compiled into a grammar, and the grammar filters the logits.</p>
</div>

A tool call is constrained generation with a runtime listening: Module 5's sampling loop plus Module 6's format finetuning plus structured output. <!-- .element: class="text-lg" -->
