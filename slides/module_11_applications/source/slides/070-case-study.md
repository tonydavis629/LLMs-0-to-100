:::divider id="divider-case" title="Case Study: The Agentic Coding Assistant" sub="Claude Code and Codex as the synthesis exhibit"
:::

---

<!-- .slide: id="case-anatomy" -->

## Anatomy of a Coding Agent

Take apart Claude Code or Codex: every section of this lecture is a component.

<div class="compare-table">
<table>
<thead><tr><th>Component</th><th>What it is</th><th>Where we covered it</th></tr></thead>
<tbody>
<tr><td><strong>System prompt</strong></td><td>The rules of engagement: how to edit, when to ask, what never to do</td><td>In-context learning</td></tr>
<tr><td><strong>Tool set</strong></td><td>Read file, edit file, search, run command</td><td>Tool use</td></tr>
<tr><td><strong>Retrieval</strong></td><td>Search over the repository to find the relevant code, since no repo fits in a context window</td><td>Retrieval-augmented generation</td></tr>
<tr><td><strong>Agent loop</strong></td><td>Reason, edit, run, read the error, edit again, with context management across a long session</td><td>Agents</td></tr>
<tr><td><strong>Verifier</strong></td><td>The project's own test suite</td><td>Next slide</td></tr>
</tbody>
</table>
</div>

---

<!-- .slide: id="case-verifier" -->

## Tests as the Verifier

The agent proposes; the tests dispose.

<div class="metric-box">
<p><strong>Module 7's verifiable-reward idea, reused at inference time.</strong> In RL training, a checkable signal told the optimizer which reasoning to reinforce. Here it tells the loop whether to stop or try again. No judge, no rubric: the tests pass or they do not.</p>
</div>

Why coding became the flagship agent domain:

- **Text-native**: code is tokens
- **Tool-rich**: compilers, linters, test runners predate the agent
- **Checkable**: the verifier was already written, for free

Few domains hand you all three. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="case-stack" -->

## Read the Stack Top to Bottom: The Whole Course Is in It

<div class="compare-table">
<table>
<thead><tr><th>Layer</th><th>Contribution</th><th>Module</th></tr></thead>
<tbody>
<tr><td>Pretraining</td><td>The capability: code and language, learned from the web</td><td>5</td></tr>
<tr><td>SFT</td><td>The instruction format and the tool-calling syntax</td><td>6</td></tr>
<tr><td>RL</td><td>Judgment on long, hard, verifiable problems</td><td>7</td></tr>
<tr><td>Evaluation</td><td>The yardstick: SWE-bench, end-state scoring</td><td>9</td></tr>
<tr><td>Serving</td><td>The tokens per second, and the KV cache behind them</td><td>10</td></tr>
<tr><td>Application</td><td>Prompt, tools, retrieval, loop, verifier: everything wrapped around the API call</td><td>11</td></tr>
</tbody>
</table>
</div>
