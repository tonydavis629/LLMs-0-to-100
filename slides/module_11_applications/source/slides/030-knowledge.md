:::divider id="divider-knowledge" title="The Knowledge Problem"
:::

---

<!-- .slide: id="knowledge-gaps" -->

## What the Weights Cannot Contain

<div class="card-grid cols-3">
<div class="card"><h4>Time</h4><p>Knowledge stops at the <strong>training cutoff</strong>. News, API changes, and current prices are missing.</p></div>
<div class="card"><h4>Privacy</h4><p>The model has never seen <strong>your data</strong>: internal documents, customer records, the codebase.</p></div>
<div class="card"><h4>Capacity</h4><p>The context window is <strong>finite</strong>, and every token in it is paid for on every call. A whole company wiki does not fit.</p></div>
</div>

Most LLM applications run into at least one of these right away. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="knowledge-hallucination" -->

## Hallucination

Ask a model about something it never saw and it still produces a plausible completion.

<div class="metric-box">
<p>This is <strong>next-token prediction working as trained</strong> (Module 5), with nothing true to condition on. In the training data, a confident and well-formatted answer follows a question far more often than "I have no information about that" does.</p>
</div>

A better prompt does not fix this. Putting true text in the context does, and that is what retrieval is for. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="knowledge-options" -->

## Three Ways to Close the Gap

<div class="compare-table">
<table>
<thead><tr><th>Option</th><th>How</th><th>Where it wins</th><th>Where it loses</th></tr></thead>
<tbody>
<tr><td><strong>Finetune</strong> (Module 6)</td><td>Train the facts into the weights</td><td>Form, style, tool syntax, domain vocabulary</td><td>Individual facts are hard to locate or edit in the weights, and every update is a training run</td></tr>
<tr><td><strong>Long context</strong></td><td>Paste everything into the prompt</td><td>Small, stable document sets</td><td>Every token is paid for on every call, in KV cache memory and prefill compute (Module 10)</td></tr>
<tr><td><strong>Retrieve</strong></td><td>Store documents outside the model and fetch only what the question needs</td><td>Large or changing corpora, since an update is a re-index</td><td>You now have a search problem, and the search can miss</td></tr>
</tbody>
</table>
</div>

---

<!-- .slide: id="knowledge-rule" -->

## Finetune or Retrieve?

The usual rule: **finetune for behavior, retrieve for knowledge.** <!-- .element: class="text-xl" -->

:::columns cols="2" gap="34px"
**Behavior goes in the weights**

- Tone, format, tool-calling syntax, refusal policy
- These should hold on every request and rarely change
+++
**Knowledge goes in the context**

- Facts change, and they often belong to one customer or one team
- Keep them in a store you can update, search, and audit
- Load only what each question needs
:::
