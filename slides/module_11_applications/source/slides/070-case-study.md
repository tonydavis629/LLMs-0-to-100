:::divider id="divider-case" title="Case Study: Coding Agents"
:::

---

<!-- .slide: id="case-anatomy" -->

## Anatomy of a Coding Agent

Claude Code and Codex are built from the parts covered in this lecture.

<div class="compare-table">
<table>
<thead><tr><th>Component</th><th>What it is</th><th>Where we covered it</th></tr></thead>
<tbody>
<tr><td><strong>System prompt</strong></td><td>Rules for the session: how to edit, when to ask, what never to do</td><td>In-context learning</td></tr>
<tr><td><strong>Tool set</strong></td><td>Read file, edit file, search, run command</td><td>Tool use</td></tr>
<tr><td><strong>Retrieval</strong></td><td>Search over the repository for the relevant code, because most repositories do not fit in a context window</td><td>Retrieval-augmented generation</td></tr>
<tr><td><strong>Agent loop</strong></td><td>Reason, edit, run, read the error, edit again. Context management keeps a long session going</td><td>Agents</td></tr>
<tr><td><strong>Verifier</strong></td><td>The project's own test suite</td><td>Next slide</td></tr>
</tbody>
</table>
</div>

---

<!-- .slide: id="case-verifier" -->

## Tests as the Verifier

The agent writes a change, and the test suite says whether it worked.

<div class="metric-box">
<p><strong>This is Module 7's verifiable reward, used at inference time.</strong> In RL training, a checkable signal told the optimizer which reasoning to reinforce. Here the same kind of signal tells the loop whether to stop or try again. It needs no judge and no rubric, because the tests either pass or fail.</p>
</div>

Three properties made coding the first domain where agents worked well:

- **Text-native**: code is tokens
- **Tool-rich**: compilers, linters, and test runners already existed
- **Checkable**: someone had already written the verifier

Few other domains have all three. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="case-stack" -->

## Where Each Module Shows Up

<div class="compare-table">
<table>
<thead><tr><th>Layer</th><th>Contribution</th><th>Module</th></tr></thead>
<tbody>
<tr><td>Pretraining</td><td>Knowledge of code and language, learned from the web</td><td>5</td></tr>
<tr><td>SFT</td><td>The instruction format and the tool-calling syntax</td><td>6</td></tr>
<tr><td>RL</td><td>Judgment on long, hard, verifiable problems</td><td>7</td></tr>
<tr><td>Evaluation</td><td>SWE-bench and end-state scoring, which measure whether the agent works</td><td>9</td></tr>
<tr><td>Serving</td><td>Tokens per second, and the KV cache that makes them affordable</td><td>10</td></tr>
<tr><td>Application</td><td>The prompt, tools, retrieval, loop, and verifier around the API call</td><td>11</td></tr>
</tbody>
</table>
</div>
