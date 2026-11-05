:::divider id="divider-protocol" title="Why the Same Model Gets Different Scores" sub="A benchmark number belongs to a model AND a protocol"
:::

---

<!-- .slide: id="protocol-knobs" -->

## Same Weights, Different Number

The same checkpoint can move several points on MMLU from choices unrelated to the weights:

<div class="card-grid cols-3">
<div class="card"><h4>Prompt template</h4><p>Chat formatting, system prompt, whether the options are labeled A&ndash;D or 1&ndash;4.</p></div>
<div class="card"><h4>Shot count</h4><p>Zero-shot versus few-shot, and how many examples, and which ones.</p></div>
<div class="card"><h4>Scoring shape</h4><p>Free generation versus option-likelihood, and whether likelihoods are length-normalized.</p></div>
<div class="card"><h4>Answer extraction</h4><p>The regex that pulls the answer out of the generated text.</p></div>
<div class="card"><h4>Normalization</h4><p>Lowercasing, punctuation, articles, unit handling.</p></div>
<div class="card"><h4>Decoding</h4><p>Greedy versus sampled, temperature, seed, number of samples.</p></div>
</div>

**A benchmark number describes a model and a protocol together.** Reproducible reports log the model revision, dataset version, prompt, decoding settings, and harness version. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="sq-chat-templates" -->

## <span class="side-quest-badge">Side Quest</span> Chat Templates Break Evaluations

Every model family wraps messages in its own special tokens, and an instruct model only works well with the one it was finetuned on:

:::columns cols="2" gap="34px"
**ChatML** (SmolLM2, Qwen 2)

```text
<|im_start|>user
What's the weather?<|im_end|>
<|im_start|>assistant
```
+++
**Mistral**

```text
<s>[INST] What's the weather? [/INST]
```
:::

A harness that uses the wrong template, or leaves it out, scores the model lower with no change to the weights. More in the <a href="https://huggingface.co/learn/llm-course/chapter11/2" target="_blank">Hugging Face chat templates chapter</a>. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="protocol-contamination" -->

## Contamination

If test items (or near-duplicates) appear in the training data, a high score measures **recall**, not generalization. Web-scale training makes this impossible to rule out completely. Partial defenses:

<div class="card-grid cols-3">
<div class="card"><h4>N-gram overlap checks</h4><p>Search the training corpus for the test items. Finds exact and near-exact copies; misses paraphrases.</p></div>
<div class="card"><h4>Time-based splits</h4><p>Test on material published <strong>after</strong> the training cutoff. Expensive, and only works once per benchmark.</p></div>
<div class="card"><h4>Private held-out sets</h4><p>Never publish the test answers. The approach GLUE, SuperGLUE, and ARC-AGI all take.</p></div>
</div>

---

<!-- .slide: id="sq-in-training" -->

## <span class="side-quest-badge">Side Quest</span> Was the Test in the Training Set?

The exercise bundles the finetuning data and the evaluation cases, so you can **look**.

<div class="card-grid cols-2">
<div class="card"><h4>What you will find</h4><p>The <code>uppercase</code>, <code>repeat</code>, and <code>reverse</code> cases use words that appear nowhere in training. All eight <code>qa</code> facts appear <strong>verbatim</strong>.</p></div>
<div class="card warn"><h4>The uncomfortable part</h4><p><code>qa</code> is the task the instruct model scores 100% on. That number is <strong>recall of memorized facts</strong>, and the report does not say so.</p></div>
</div>

This check took one `grep` on 1,500 examples. What could it mean for a model trained on much of the public internet? <!-- .element: class="text-lg" -->

---

<!-- .slide: id="protocol-saturation-chart" -->

## How Long a Benchmark Stays Useful

<div class="sat-chart"><div class="sat-row fragment"><div class="sat-name">MNIST</div><div class="sat-track"><div class="sat-span" style="left: 0%; width: 50.0%;"></div><div class="sat-years" style="left: 51.5%;">1998 &rarr; 2012 &middot; 14 years</div></div></div><div class="sat-row fragment"><div class="sat-name">ImageNet</div><div class="sat-track"><div class="sat-span" style="left: 39.3%; width: 21.4%;"></div><div class="sat-years" style="left: 62.2%;">2009 &rarr; 2015 &middot; 6 years</div></div></div><div class="sat-row fragment"><div class="sat-name">SQuAD</div><div class="sat-track"><div class="sat-span fast" style="left: 64.3%; width: 7.1%;"></div><div class="sat-years" style="left: 72.9%;">2016 &rarr; 2018 &middot; 2 years</div></div></div><div class="sat-row fragment"><div class="sat-name">GLUE</div><div class="sat-track"><div class="sat-span fast" style="left: 71.4%; width: 3.6%;"></div><div class="sat-years" style="left: 76.5%;">2018 &rarr; 2019 &middot; 1 year</div></div></div><div class="sat-row fragment"><div class="sat-name">SuperGLUE</div><div class="sat-track"><div class="sat-span fast" style="left: 75.0%; width: 7.1%;"></div><div class="sat-years" style="left: 83.6%;">2019 &rarr; 2021 &middot; 2 years</div></div></div><div class="sat-row fragment"><div class="sat-name">ARC-AGI-1</div><div class="sat-track"><div class="sat-span" style="left: 75.0%; width: 17.9%;"></div><div class="sat-years before" style="right: 26.6%;">2019 &rarr; 2024 &middot; 5 years</div></div></div><div class="sat-row fragment"><div class="sat-name">MMLU</div><div class="sat-track"><div class="sat-span" style="left: 78.6%; width: 14.3%;"></div><div class="sat-years before" style="right: 23%;">2020 &rarr; 2024 &middot; 4 years</div></div></div><div class="sat-row fragment"><div class="sat-name">GSM8K</div><div class="sat-track"><div class="sat-span fast" style="left: 82.1%; width: 7.1%;"></div><div class="sat-years before" style="right: 19.5%;">2021 &rarr; 2023 &middot; 2 years</div></div></div><div class="sat-row fragment"><div class="sat-name">HumanEval</div><div class="sat-track"><div class="sat-span" style="left: 82.1%; width: 10.7%;"></div><div class="sat-years before" style="right: 19.5%;">2021 &rarr; 2024 &middot; 3 years</div></div></div><div class="sat-row fragment"><div class="sat-name">GPQA Diamond</div><div class="sat-track"><div class="sat-span fast" style="left: 89.3%; width: 7.1%;"></div><div class="sat-years before" style="right: 12.5%;">2023 &rarr; 2025 &middot; 2 years</div></div></div><div class="sat-row fragment"><div class="sat-name">SWE-bench Verified</div><div class="sat-track"><div class="sat-span fast" style="left: 92.9%; width: 3.6%;"></div><div class="sat-years before" style="right: 8.7%;">2024 &rarr; 2025 &middot; 1 year</div></div></div><div class="sat-axis"><div class="sat-tick" style="left: 7.1%;">2000</div><div class="sat-tick" style="left: 25.0%;">2005</div><div class="sat-tick" style="left: 42.9%;">2010</div><div class="sat-tick" style="left: 60.7%;">2015</div><div class="sat-tick" style="left: 78.6%;">2020</div><div class="sat-tick" style="left: 96.4%;">2025</div></div></div>

Each bar runs from **publication** to the year the best systems reached the human or expert ceiling. The axis is to scale. <!-- .element: class="text-lg" style="margin-top: 6px;" -->

---

<!-- .slide: id="protocol-saturation-meaning" -->

## What Saturation Costs You

Once the best models sit near the ceiling, a benchmark stops separating them. Remaining gains are mostly **noise and overfitting**.

<div class="card-grid cols-2">
<div class="card"><h4>Why the field keeps building successors</h4><p>GLUE to SuperGLUE, MMLU to MMLU-Pro and GPQA, GSM8K to AIME, ARC to ARC-AGI. Each one exists because its predecessor stopped discriminating, and AIME and ARC-AGI-1 have since saturated too.</p></div>
<div class="card"><h4>Why the window keeps shrinking</h4><p>Fourteen years for MNIST. Roughly two for the recent ones. A benchmark published today should be assumed to have a <strong>short useful life</strong>.</p></div>
</div>

**Public benchmarks compare models in general. A small private test set from your own task is the only one nobody can train on.** <!-- .element: class="text-lg" -->

---

<!-- .slide: id="protocol-measurement-gap" -->

## What the Benchmarks Leave Out

Labs optimize what they can measure. After enough rounds of that, the measurable skills are strong, and whatever is still weak sits where no benchmark looks.

<div class="card-grid cols-2">
<div class="card"><h4>Measured, so optimized</h4><p>Math with a checkable answer, code with unit tests, multiple-choice knowledge, formats a parser can verify. Scores here climb until the benchmark saturates.</p></div>
<div class="card warn"><h4>Hard to measure, so left behind</h4><p>Noticing that the question itself is wrong, judging what matters in a long document, writing that reads like a person meant it. Nothing pushes on these, and they lag.</p></div>
</div>

The result is a **spiky** model: expert-level on a benchmark task and oddly off on the task next to it, in a way that is hard to describe and harder to score. Criticizing a model that has a table of high numbers can feel unscientific. Do it anyway. The numbers tell you about the benchmarks, and the weak spots are likely in the places the benchmarks skip. <!-- .element: class="text-lg" -->
