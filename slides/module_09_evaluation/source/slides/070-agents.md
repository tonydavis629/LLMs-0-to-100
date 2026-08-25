:::divider id="divider-agents" title="Evaluating Agents" sub="Not `is this answer right` but `did the task get done`"
:::

---

<!-- .slide: id="agents-shift" -->

## The Question Changes

An agent is a model **plus tools, plus many turns**. Scoring shifts from inspecting a string to inspecting the **final state of an environment**.

<div class="card-grid cols-2">
<div class="card"><h4>Answer evaluation</h4><p>Compare the output to a key. One turn, one string, deterministic scoring.</p></div>
<div class="card"><h4>Task evaluation</h4><p>Run the project's test suite. Query the database. Check whether the file exists and contains what it should. <strong>The environment is the grader.</strong></p></div>
</div>

Evaluation stops being a metric on outputs and becomes an **integration test**: engineering practice, not research practice. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="agents-benchmarks" -->

## The Agent Benchmarks You Will Hear About

<div class="bench-table">
<table>
<thead><tr><th>Benchmark</th><th>Environment</th><th>How it is scored</th></tr></thead>
<tbody>
<tr><td><strong>SWE-bench</strong> (2023)</td><td>A real GitHub repository with a real issue</td><td>Apply the model's patch and run the project's test suite. <strong>SWE-bench Verified</strong> is the human-filtered subset everyone reports.</td></tr>
<tr><td><strong>GAIA</strong></td><td>Browsing, files, and tools</td><td>General-assistant questions with short verifiable answers.</td></tr>
<tr><td><strong>WebArena, OSWorld</strong></td><td>A simulated browser or desktop</td><td>Inspect the end state: was the form submitted, the file saved, the setting changed?</td></tr>
<tr><td><strong>tau-bench</strong></td><td>Multi-turn customer service with tool calls</td><td>Did the agent follow the policy rules and reach the correct final state?</td></tr>
</tbody>
</table>
</div>

Modules 10 and 11 build the systems these benchmarks measure. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="agents-hard" -->

## Why Agent Evaluation Is Hard

<div class="card-grid cols-4">
<div class="card"><h4>Expensive</h4><p>Runs are long and multi-turn. A full sweep costs real money and hours.</p></div>
<div class="card"><h4>Stateful and flaky</h4><p>Environments drift, networks fail, and a rerun is not the same run.</p></div>
<div class="card"><h4>Partial progress</h4><p>Getting 80% of the way is real value and hard to score. Most suites score pass/fail anyway.</p></div>
<div class="card warn"><h4>Right for the wrong reason</h4><p>The agent edits the tests instead of the code. The grader says pass.</p></div>
</div>

The last one is reward hacking where it is hardest to notice: **the scoring function is a program the agent can reach.** <!-- .element: class="text-lg" -->
