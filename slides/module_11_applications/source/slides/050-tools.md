:::divider id="divider-tools" title="Tool Use"
:::

---

<!-- .slide: id="tools-mechanism" -->

## How a Tool Call Works

<div class="card-grid cols-4">
<div class="card"><h4>1. Emit</h4><p>The model generates a <strong>structured call</strong>: a tool name and arguments, written as ordinary tokens.</p></div>
<div class="card"><h4>2. Pause</h4><p>The runtime detects the call and <strong>stops generation</strong>.</p></div>
<div class="card"><h4>3. Execute</h4><p>Ordinary software runs the call: a search, a database query, a calculation. <strong>The model itself executes nothing.</strong></p></div>
<div class="card"><h4>4. Resume</h4><p>The result is appended to the context <strong>as tokens</strong>, and the model keeps generating with the result in view.</p></div>
</div>

The model is still predicting next tokens. Tool use is a protocol on top of sampling, and the runtime is what enforces it. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="tools-diagram" -->

## Tool Call Round Trip

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

The same four steps as one round trip. Tokens go out, ordinary software runs, and tokens come back. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="tools-trained" -->

## Where the Ability Comes From

Models are **trained** to emit well-formed calls at the right moments.

:::columns cols="2" gap="34px"
**In practice**

- An SFT pass (Module 6) over examples that show when a question needs a tool, how to format the call, and how to read the result
- The chat template gains a **tool role** alongside system, user, and assistant
+++
**In research**

- Toolformer (Schick et al., 2023): the model annotates its own training text with API calls
- Calls that reduce the loss on the following tokens are kept, and the model is finetuned on the result
- The model learns where a calculator would have helped
:::

A base model can imitate the call format from a prompt. A finetuned model does it reliably enough to build on, and the only difference between them is a dataset. <!-- .element: class="text-lg" style="margin-top: 10px;" -->

---

<!-- .slide: id="tools-set" -->

## Common Tools

<div class="card-grid cols-4">
<div class="card"><h4>Calculator</h4><p>Digits are tokens like any others (Module 4), so models are unreliable at arithmetic. A calculator is exact.</p></div>
<div class="card"><h4>Web search</h4><p>Facts from after the training cutoff. This is retrieval over the live internet.</p></div>
<div class="card"><h4>Code execution</h4><p>The most general tool. If a program can compute it, the model can ask for it.</p></div>
<div class="card"><h4>Retrieval</h4><p>The RAG pipeline, offered as a tool the model can choose to call.</p></div>
</div>

**Tools take over what next-token prediction does badly:** exact computation, current knowledge, and actions with side effects. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="tools-apis" -->

## Function Calling

The major APIs all work the same way:

<div class="metric-box">
<p>The developer sends <strong>JSON schemas</strong> that describe the available tools. The model returns a call that matches one of them. With <strong>constrained decoding</strong>, the schema is compiled into a grammar that masks invalid tokens during sampling, so the call always parses.</p>
</div>

A tool call combines three things from earlier: the sampling loop (Module 4), format finetuning (Module 6), and structured output. <!-- .element: class="text-lg" -->
