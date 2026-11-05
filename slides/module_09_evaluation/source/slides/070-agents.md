:::divider id="divider-agents" title="Evaluating Agents" sub="Not `is this answer right` but `did the task get done`"
:::

---

<!-- .slide: id="agents-shift" -->

## The Question Changes

An agent is a model **plus tools, over many turns**. You grade the **final state of the environment**, not a string.

<div class="card-grid cols-2">
<div class="card"><h4>Answer evaluation</h4><p>One turn. Compare the output to a key.</p></div>
<div class="card"><h4>Task evaluation</h4><p>Run the tests, query the database, check the file. <strong>The environment is the grader.</strong></p></div>
</div>

Evaluation becomes an **integration test**. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="agents-benchmarks" -->

## The Agent Benchmarks You Will Hear About

<div class="bench-table dense">
<table>
<thead><tr><th>Benchmark</th><th>Environment</th><th>How it is scored</th><th>Saturated</th></tr></thead>
<tbody>
<tr><td><strong>SWE-bench</strong> (2023)</td><td>A real GitHub repo and issue</td><td>Apply the patch, run the tests. <strong>Verified</strong> is the human-checked subset.</td><td>2025</td></tr>
<tr><td><strong>GAIA</strong></td><td>Browsing, files, tools</td><td>Assistant questions with short checkable answers.</td><td>Near the human baseline, 2026</td></tr>
<tr><td><strong>WebArena, OSWorld</strong></td><td>A browser or desktop</td><td>Check the end state: was the file saved?</td><td>OSWorld passed humans, early 2026</td></tr>
<tr><td><strong>tau-bench</strong></td><td>Customer service with tools</td><td>Did it follow policy and reach the right state?</td><td>2026</td></tr>
</tbody>
</table>
</div>

---

<!-- .slide: id="agents-open" -->

## Agent Benchmarks That Are Still Open

<div class="bench-table">
<table>
<thead><tr><th>Benchmark</th><th>Year</th><th>What it measures</th><th>Best score, late 2026</th></tr></thead>
<tbody>
<tr><td><strong>SWE-bench Pro</strong></td><td>2025</td><td>Harder issues in several languages. About 30% of tasks later found broken.</td><td>~62% on Scale's public set</td></tr>
<tr><td><strong>Terminal-Bench</strong></td><td>2025</td><td>Long multi-step tasks in a real shell.</td><td>~60% on version 4.0</td></tr>
<tr><td><strong>GDPval</strong></td><td>2025</td><td>Real work from 44 occupations, judged blind against experts' work.</td><td>Reported as a win-or-tie rate</td></tr>
</tbody>
</table>
</div>

---

<!-- .slide: id="agents-hard" -->

## Why Agent Evaluation Is Hard

<div class="card-grid cols-4">
<div class="card"><h4>Expensive</h4><p>Long runs cost hours and money.</p></div>
<div class="card"><h4>Flaky</h4><p>Environments drift; reruns differ.</p></div>
<div class="card"><h4>Partial progress</h4><p>80% done usually scores as a fail.</p></div>
<div class="card warn"><h4>Wrong reason</h4><p>The agent edits the tests. The grader says pass.</p></div>
</div>

The grader is a program the agent can reach. <!-- .element: class="text-lg" -->
