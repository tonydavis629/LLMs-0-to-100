:::divider id="divider-knowledge" title="The Knowledge Problem" sub="Three gaps a frozen model cannot close on its own"
:::

---

<!-- .slide: id="knowledge-gaps" -->

## What the Weights Cannot Contain

<div class="card-grid cols-3">
<div class="card"><h4>Time</h4><p>Knowledge stops at the <strong>training cutoff</strong>. News, API changes, current pricing: not in the weights.</p></div>
<div class="card"><h4>Privacy</h4><p>The model has never seen <strong>your data</strong>: internal documents, customer records, the codebase.</p></div>
<div class="card"><h4>Capacity</h4><p>The context window is <strong>finite</strong>, and every token is paid for on every call. The whole wiki does not fit.</p></div>
</div>

Every serious LLM application hits at least one of these on day one. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="knowledge-hallucination" -->

## What the Model Does When Asked Anyway

It completes plausibly. That is not a malfunction:

<div class="metric-box">
<p>Hallucination is <strong>next-token prediction doing what Module 5 trained it to do</strong>, with nothing true to condition on. A confident, well-formatted, wrong answer is far more probable in the training distribution than "I have no information about that."</p>
</div>

A better prompt cannot fix this. The fix: put something true in the context. That is retrieval. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="knowledge-options" -->

## Three Ways to Close the Gap

<div class="compare-table">
<table>
<thead><tr><th>Option</th><th>How</th><th>Where it wins</th><th>Where it loses</th></tr></thead>
<tbody>
<tr><td><strong>Finetune</strong> (Module 6)</td><td>Train the facts into the weights</td><td>Form, style, tool syntax, domain vocabulary</td><td>Facts land diffusely; updates need retraining; yesterday's finetune is stale today</td></tr>
<tr><td><strong>Long context</strong></td><td>Paste everything into the prompt</td><td>Small, stable document sets</td><td>Every token paid for on every call: KV cache growth and prefill compute (Module 10)</td></tr>
<tr><td><strong>Retrieve</strong></td><td>Store documents outside the model; fetch only what the question needs</td><td>Large, changing corpora; freshness is one index update away</td><td>Now you have a search problem; the search can miss</td></tr>
</tbody>
</table>
</div>

---

<!-- .slide: id="knowledge-rule" -->

## The Rule of Thumb the Field Converged On

**Finetune for behavior. Retrieve for knowledge.** <!-- .element: class="text-xl" -->

:::columns cols="2" gap="34px"
**Behavior belongs in weights**

- Tone, format, tool-calling syntax, refusal policy
- Stable properties wanted on every request, learned once
+++
**Knowledge belongs in context**

- Facts change, multiply, and belong to someone
- Keep them in a store you can update, search, and audit
- Load only what each question needs
:::

