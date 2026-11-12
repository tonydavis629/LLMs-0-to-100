:::divider id="divider-deep-research" title="Deep Research"
:::

---

<!-- .slide: id="dr-loop" -->

## Deep Research Agents

A deep research agent answers one question by searching and reading for several minutes, then writing a report with citations.

<div class="card-grid cols-4">
<div class="card"><h4>Plan</h4><p>Break the question into sub-questions.</p></div>
<div class="card"><h4>Search</h4><p>Write search queries and run them. The agent chooses its own queries.</p></div>
<div class="card"><h4>Read</h4><p>Open the pages and take notes on what each source says.</p></div>
<div class="card"><h4>Write</h4><p>Draft the report and attach a source to each claim.</p></div>
</div>

After reading, the agent decides what is still missing and searches again. It writes only when the notes cover the question or the budget runs out. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="dr-vs-rag" -->

## RAG and Deep Research Compared

<div class="compare-table">
<table>
<thead><tr><th></th><th>RAG pipeline</th><th>Deep research</th></tr></thead>
<tbody>
<tr><td><strong>Retrievals</strong></td><td>One per question</td><td>Tens to hundreds</td></tr>
<tr><td><strong>Query</strong></td><td>The user's question, embedded as written</td><td>Written by the model and revised after each result</td></tr>
<tr><td><strong>Corpus</strong></td><td>A fixed index</td><td>The live web, sometimes with private files</td></tr>
<tr><td><strong>Control flow</strong></td><td>Fixed: retrieve, then generate</td><td>An agent loop that decides when to stop</td></tr>
<tr><td><strong>Latency</strong></td><td>Seconds</td><td>Minutes</td></tr>
</tbody>
</table>
</div>

Deep research is retrieval used as a tool inside an agent loop. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="dr-subagents" -->

## Splitting the Work Across Subagents

<div style="text-align: center; margin: 6px 0;">
<svg viewBox="0 0 960 300" width="100%" style="max-height: 400px;">
  <defs>
    <marker id="dr-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#8892a4"/>
    </marker>
  </defs>
  <g stroke="#2a3450" stroke-width="1.5" fill="rgba(74,158,255,0.05)">
    <rect x="60" y="80" width="250" height="70" rx="8" stroke="#f5a623"/>
    <rect x="60" y="220" width="250" height="70" rx="8" stroke="#f5a623"/>
    <rect x="620" y="10" width="280" height="70" rx="8"/>
    <rect x="620" y="115" width="280" height="70" rx="8"/>
    <rect x="620" y="220" width="280" height="70" rx="8"/>
  </g>
  <g text-anchor="middle" font-size="16" font-weight="600" fill="#4a9eff">
    <text x="185" y="110" fill="#f5a623">Lead agent</text>
    <text x="185" y="250" fill="#f5a623">Cited report</text>
    <text x="760" y="40">Subagent 1</text>
    <text x="760" y="145">Subagent 2</text>
    <text x="760" y="250">Subagent 3</text>
  </g>
  <g text-anchor="middle" font-size="12" fill="#8892a4">
    <text x="185" y="133">plans, then merges the summaries</text>
    <text x="185" y="273">written by the lead agent</text>
    <text x="760" y="63">own context: searches, reads pages</text>
    <text x="760" y="168">own context: searches, reads pages</text>
    <text x="760" y="273">own context: searches, reads pages</text>
  </g>
  <g stroke="#8892a4" stroke-width="1.5" marker-end="url(#dr-arrow)" marker-start="url(#dr-arrow)">
    <line x1="314" y1="100" x2="616" y2="48"/>
    <line x1="314" y1="118" x2="616" y2="148"/>
    <line x1="314" y1="136" x2="616" y2="250"/>
  </g>
  <line x1="185" y1="150" x2="185" y2="216" stroke="#8892a4" stroke-width="1.5" marker-end="url(#dr-arrow)"/>
  <text x="465" y="52" font-size="12.5" fill="#8892a4" text-anchor="middle">sub-question out, short summary back</text>
</svg>
</div>

A hundred web pages do not fit in one context window. Each subagent reads in its own context and returns a short summary, so the lead agent's context stays small. The cost is tokens: Anthropic reports that its multi-agent research system uses about 15 times as many as a chat. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="dr-verify" -->

## Research Has No Test Suite

:::columns cols="2" gap="34px"
**Coding agent**

- The tests pass or fail
- The loop sees the failure and tries again
+++
**Research agent**

- Nothing checks automatically that the report is right
- A cited page may not say what the report claims
- The page itself may be wrong
:::

<div class="metric-box">
<p>Current systems add a <strong>citation check</strong>: a second model pass compares each claim with the passage it cites. That is an LLM judge (Module 9), so it reduces errors without guaranteeing anything. The citations are there so a person can check.</p>
</div>
