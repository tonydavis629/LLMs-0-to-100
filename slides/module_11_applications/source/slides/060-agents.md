:::divider id="divider-agents" title="Agents"
:::

---

<!-- .slide: id="agents-loop" -->

## From One Call to a Loop

A single tool call answers a question. An **agent** runs tool calls in a loop:

<div class="card-grid cols-4">
<div class="card"><h4>Generate</h4><p>Reason about the goal and decide the next action.</p></div>
<div class="card"><h4>Act</h4><p>Call a tool: search, read, edit, run.</p></div>
<div class="card"><h4>Observe</h4><p>The result enters the context as tokens.</p></div>
<div class="card"><h4>Repeat</h4><p>Until a stop condition: the goal is met, the budget is spent, or the agent needs help.</p></div>
</div>

With the loop, the model can **act, see the result, and adjust**. The rest of this section covers what keeps the loop reliable. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="agents-diagram" -->

## The Agent Loop

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
  <text x="480" y="36" text-anchor="middle" font-size="12.5" fill="#8892a4">every cycle appends to the context, which is the agent's only memory</text>
  <text x="182" y="222" text-anchor="start" font-size="12" fill="#8892a4">stop condition met</text>
</svg>
</div>

The arrow from Observe back to Generate is what separates an agent from a single tool call. Each new step sees the result of the last one. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="agents-react" -->

## ReAct

Yao et al. (2022) interleaved **reasoning traces** with **actions** in one generation stream:

```text
Thought:  The question asks about the 2022 error code list.
          My training data ends before that. I should search.
Action:   search("PX-220 error code list 2022")
Observation: [search results appear here as tokens]
Thought:  Result 2 has the code table. I can answer now.
```

This is chain-of-thought with tool results arriving between the thoughts.

- Each observation informs the next reasoning step
- The model can recover from a bad search
- Most agents in use today follow this pattern. The name is short for Reason + Act

---

:::figure img="images/shunyu_yao.jpg" name="Shunyu Yao" kicker="ReAct: Synergizing Reasoning and Acting in Language Models (2022)" alt="Shunyu Yao"
- Wrote ReAct as a Princeton PhD student. Reason, act, observe became the standard agent loop
- The idea was a layout for the context: reasoning and tool results **interleaved in one stream**
- Also built tau-bench, the Module 9 agent benchmark that scores the environment's end state
:::

---

<!-- .slide: id="agents-context" -->

## The Context Window as Working Memory

Every thought, call, and observation accumulates in the context. On a long task the transcript **outgrows the window** and the agent loses its own history.

<div class="card-grid cols-3">
<div class="card"><h4>Summarize</h4><p>Periodically compress the transcript into a summary and continue from that.</p></div>
<div class="card"><h4>Prune</h4><p>Drop stale tool output. An old directory listing does not need to stay in the context.</p></div>
<div class="card"><h4>Pin</h4><p>Keep the goal and the main constraints at a fixed place in the prompt so they are never dropped.</p></div>
</div>

The context is all the memory the agent has, and every token in it is billed on every step. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="agents-errors" -->

## Errors Compound

A good success rate per step turns into a poor success rate per task:

$$0.95^{20} \approx 0.36$$

<div class="metric-box">
<p>Twenty steps at 95% each succeed together <strong>36%</strong> of the time, which is why agents are less reliable than single answers from the same model. Slightly better steps do not fix this. <strong>Verification</strong> does: run the tests, check the end state, confirm before an irreversible action. With a verifier, the agent can catch a failed step and retry it.</p>
</div>

Coding agents rely on the test suite because it is a verifier that already exists. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="agents-mcp" -->

## The Model Context Protocol

Each tool used to need a custom integration for each agent. The **Model Context Protocol** (MCP) is a standard interface: any client that speaks it can use any tool server that speaks it.

:::columns cols="2" gap="34px"
**What it standardizes**

- How a client discovers a server's tools
- The format of a call and its result
- How a server exposes resources and prompts alongside tools
+++
**Why it exists now**

- A protocol is worth writing once many people are building the same thing
- Tool-using agents reached that point
- As with HTTP, vendors share the interface and compete on what sits behind it
:::
