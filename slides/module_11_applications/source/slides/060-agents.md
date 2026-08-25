:::divider id="divider-agents" title="Agents" sub="Put the tool call in a loop and let the world talk back"
:::

---

<!-- .slide: id="agents-loop" -->

## From One Call to a Loop

One tool call answers a question. An **agent** is what you get when the call sits inside a loop:

<div class="card-grid cols-4">
<div class="card"><h4>Generate</h4><p>Reason about the goal and decide the next action.</p></div>
<div class="card"><h4>Act</h4><p>Call a tool: search, read, edit, run.</p></div>
<div class="card"><h4>Observe</h4><p>The result lands in the context as tokens.</p></div>
<div class="card"><h4>Repeat</h4><p>Until a stop condition: goal met, budget spent, or help needed.</p></div>
</div>

The loop turns a text predictor into a system that **acts on the world and reacts to what happens**. The rest of this section: what the loop needs to keep from falling over. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="agents-diagram" -->

## The Loop, Drawn

<div style="text-align: center; margin: 8px 0;">
<svg viewBox="0 0 960 340" width="100%" style="max-height: 480px;">
  <defs>
    <marker id="agents-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#8892a4"/>
    </marker>
    <marker id="agents-arrow-blue" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#4a9eff"/>
    </marker>
  </defs>
  <g stroke="#2a3450" stroke-width="1.5" fill="rgba(74,158,255,0.05)">
    <rect x="50" y="120" width="240" height="70" rx="8"/>
    <rect x="370" y="120" width="220" height="70" rx="8"/>
    <rect x="670" y="120" width="250" height="70" rx="8"/>
    <rect x="50" y="250" width="240" height="70" rx="8" stroke="#f5a623"/>
  </g>
  <g text-anchor="middle" font-size="16" font-weight="600" fill="#4a9eff">
    <text x="170" y="150">Generate</text>
    <text x="480" y="150">Act</text>
    <text x="795" y="150">Observe</text>
    <text x="170" y="280" fill="#f5a623">Stop</text>
  </g>
  <g text-anchor="middle" font-size="12" fill="#8892a4">
    <text x="170" y="174">reason, choose an action</text>
    <text x="480" y="174">call a tool</text>
    <text x="795" y="174">the result lands in context</text>
    <text x="170" y="304">goal met, budget spent, help needed</text>
  </g>
  <g stroke="#8892a4" stroke-width="1.5" marker-end="url(#agents-arrow)" fill="none">
    <line x1="290" y1="155" x2="366" y2="155"/>
    <line x1="590" y1="155" x2="666" y2="155"/>
    <line x1="170" y1="190" x2="170" y2="246"/>
  </g>
  <path d="M 795 118 L 795 60 Q 795 48 783 48 L 182 48 Q 170 48 170 60 L 170 114" stroke="#4a9eff" stroke-width="1.5" fill="none" marker-end="url(#agents-arrow-blue)"/>
  <text x="480" y="36" text-anchor="middle" font-size="12.5" fill="#8892a4">every cycle appends to the context: the transcript is the memory</text>
  <text x="182" y="222" text-anchor="start" font-size="12" fill="#8892a4">stop condition met</text>
</svg>
</div>

Three arrows forward, one arrow back. The back edge is what makes it an agent: the next thought sees what the world just did. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="agents-react" -->

## ReAct: Where the Pattern Comes From

Yao et al. (2022) interleaved **reasoning traces** with **actions** in one generation stream:

```text
Thought:  The question asks about the 2022 error code list.
          My training data ends before that. I should search.
Action:   search("PX-220 error code list 2022")
Observation: [search results appear here as tokens]
Thought:  Result 2 has the code table. I can answer now.
```

Chain-of-thought, with the world talking back between thoughts:

- Each observation informs the next reasoning step
- The model can recover from a bad search instead of committing to it
- ReAct (Reason + Act) is the skeleton of nearly every agent shipping today

---

:::figure img="images/shunyu_yao.jpg" name="Shunyu Yao" kicker="ReAct (2022), tau-bench (2024)" alt="Shunyu Yao"
As a Princeton PhD student, wrote the paper that fixed reason-act-observe as the standard agent loop. ReAct's contribution was not a new model but a new **shape for the context**: reasoning and observations interleaved in one stream.

Yao also built tau-bench, the Module 9f agent benchmark that scores the **end state of the environment** rather than the transcript. Yao went on to OpenAI.
:::

---

<!-- .slide: id="agents-context" -->

## The Context Window as Working Memory

Every thought, call, and observation accumulates in the context. On a long task the transcript **overflows the window**: the agent forgets its own history.

<div class="card-grid cols-3">
<div class="card"><h4>Summarize</h4><p>Periodically compress the transcript so far into a summary and continue from that.</p></div>
<div class="card"><h4>Prune</h4><p>Drop stale tool output: yesterday's directory listing does not need to occupy today's tokens.</p></div>
<div class="card"><h4>Pin</h4><p>Keep the goal and key constraints at a fixed place in the prompt so no amount of scrolling loses them.</p></div>
</div>

The context is the agent's entire memory, and it is also the entire bill. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="agents-errors" -->

## Errors Compound

A per-step success rate that sounds excellent becomes a task failure rate that sounds broken:

$$0.95^{20} \approx 0.36$$

<div class="metric-box">
<p>Twenty steps at 95% each: the task succeeds <strong>36%</strong> of the time. This explains why agent reliability lags single-shot quality. The field's answer is <strong>verification</strong>, not slightly better steps: run the tests, check the end state, confirm before the irreversible step. A verifier lets a 95% agent catch its own 5%.</p>
</div>

Coding agents lean on the test suite because it is a verifier someone already wrote. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="agents-mcp" -->

## The Model Context Protocol

Every tool used to need a custom integration per agent. The **Model Context Protocol** is the standard interface: any client that speaks it can use any tool server that speaks it.

:::columns cols="2" gap="34px"
**What it standardizes**

- How a client discovers a server's tools
- What a call and its result look like on the wire
- How resources and prompts are exposed alongside tools
+++
**What it signals**

- Protocols appear when a pattern stops being research
- Tool-using agents crossed that line
- Like the web: agree on the interface, compete on everything else
:::
