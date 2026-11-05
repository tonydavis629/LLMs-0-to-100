:::divider id="divider-protocol" title="Why the Same Model Gets Different Scores" sub="A benchmark number belongs to a model AND a protocol"
:::

---

<!-- .slide: id="protocol-knobs" -->

## Same Weights, Different Number

Each of these can move an MMLU score by several points:

<div class="card-grid cols-3">
<div class="card"><h4>Prompt template</h4><p>Chat format, system prompt, A&ndash;D or 1&ndash;4.</p></div>
<div class="card"><h4>Shot count</h4><p>How many examples, and which.</p></div>
<div class="card"><h4>Scoring shape</h4><p>Generation or likelihood; length normalization.</p></div>
<div class="card"><h4>Answer extraction</h4><p>The regex that finds the answer.</p></div>
<div class="card"><h4>Normalization</h4><p>Case, punctuation, units.</p></div>
<div class="card"><h4>Decoding</h4><p>Greedy or sampled, temperature, seed.</p></div>
</div>

**A benchmark number describes a model and a protocol together.** <!-- .element: class="text-lg" -->

---

<!-- .slide: id="sq-chat-templates" -->

## <span class="side-quest-badge">Side Quest</span> Chat Templates Break Evaluations

Each model family wraps messages in its own special tokens:

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

Use the wrong one, or none, and the score drops with the same weights. See the <a href="https://huggingface.co/learn/llm-course/chapter11/2" target="_blank">Hugging Face chat templates chapter</a>. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="protocol-contamination" -->

## Contamination

If test items leaked into training data, the score measures **recall**, not ability. At web scale you cannot fully rule it out.

<div class="card-grid cols-3">
<div class="card"><h4>Overlap checks</h4><p>Search the training data for the test items. Misses paraphrases.</p></div>
<div class="card"><h4>Time-based splits</h4><p>Test only on material from <strong>after</strong> the training cutoff.</p></div>
<div class="card"><h4>Private test sets</h4><p>Never publish the answers, as ARC-AGI does.</p></div>
</div>

---

<!-- .slide: id="protocol-saturation-chart" -->

## How Long a Benchmark Stays Useful

<div class="sat-chart"><div class="sat-row fragment"><div class="sat-name">MNIST</div><div class="sat-track"><div class="sat-span" style="left: 0%; width: 50.0%;"></div><div class="sat-years" style="left: 51.5%;">1998 &rarr; 2012 &middot; 14 years</div></div></div><div class="sat-row fragment"><div class="sat-name">ImageNet</div><div class="sat-track"><div class="sat-span" style="left: 39.3%; width: 21.4%;"></div><div class="sat-years" style="left: 62.2%;">2009 &rarr; 2015 &middot; 6 years</div></div></div><div class="sat-row fragment"><div class="sat-name">SQuAD</div><div class="sat-track"><div class="sat-span fast" style="left: 64.3%; width: 7.1%;"></div><div class="sat-years" style="left: 72.9%;">2016 &rarr; 2018 &middot; 2 years</div></div></div><div class="sat-row fragment"><div class="sat-name">GLUE</div><div class="sat-track"><div class="sat-span fast" style="left: 71.4%; width: 3.6%;"></div><div class="sat-years" style="left: 76.5%;">2018 &rarr; 2019 &middot; 1 year</div></div></div><div class="sat-row fragment"><div class="sat-name">SuperGLUE</div><div class="sat-track"><div class="sat-span fast" style="left: 75.0%; width: 7.1%;"></div><div class="sat-years" style="left: 83.6%;">2019 &rarr; 2021 &middot; 2 years</div></div></div><div class="sat-row fragment"><div class="sat-name">ARC-AGI-1</div><div class="sat-track"><div class="sat-span" style="left: 75.0%; width: 17.9%;"></div><div class="sat-years before" style="right: 26.6%;">2019 &rarr; 2024 &middot; 5 years</div></div></div><div class="sat-row fragment"><div class="sat-name">MMLU</div><div class="sat-track"><div class="sat-span" style="left: 78.6%; width: 14.3%;"></div><div class="sat-years before" style="right: 23%;">2020 &rarr; 2024 &middot; 4 years</div></div></div><div class="sat-row fragment"><div class="sat-name">GSM8K</div><div class="sat-track"><div class="sat-span fast" style="left: 82.1%; width: 7.1%;"></div><div class="sat-years before" style="right: 19.5%;">2021 &rarr; 2023 &middot; 2 years</div></div></div><div class="sat-row fragment"><div class="sat-name">HumanEval</div><div class="sat-track"><div class="sat-span" style="left: 82.1%; width: 10.7%;"></div><div class="sat-years before" style="right: 19.5%;">2021 &rarr; 2024 &middot; 3 years</div></div></div><div class="sat-row fragment"><div class="sat-name">GPQA Diamond</div><div class="sat-track"><div class="sat-span fast" style="left: 89.3%; width: 7.1%;"></div><div class="sat-years before" style="right: 12.5%;">2023 &rarr; 2025 &middot; 2 years</div></div></div><div class="sat-row fragment"><div class="sat-name">SWE-bench Verified</div><div class="sat-track"><div class="sat-span fast" style="left: 92.9%; width: 3.6%;"></div><div class="sat-years before" style="right: 8.7%;">2024 &rarr; 2025 &middot; 1 year</div></div></div><div class="sat-axis"><div class="sat-tick" style="left: 7.1%;">2000</div><div class="sat-tick" style="left: 25.0%;">2005</div><div class="sat-tick" style="left: 42.9%;">2010</div><div class="sat-tick" style="left: 60.7%;">2015</div><div class="sat-tick" style="left: 78.6%;">2020</div><div class="sat-tick" style="left: 96.4%;">2025</div></div></div>

Each bar runs from **publication** to the year top models reached the human or expert ceiling. To scale. <!-- .element: class="text-lg" style="margin-top: 6px;" -->

---

<!-- .slide: id="protocol-saturation-meaning" -->

## What Saturation Costs You

Near the ceiling, a benchmark stops separating models. What is left is mostly **noise and overfitting**.

<div class="card-grid cols-2">
<div class="card"><h4>So the field builds successors</h4><p>GLUE to SuperGLUE, MMLU to MMLU-Pro, GSM8K to AIME. Then AIME saturated too.</p></div>
<div class="card"><h4>And the window keeps shrinking</h4><p>Fourteen years for MNIST, about two for recent benchmarks.</p></div>
</div>

**A small private test set from your own task is the only one nobody can train on.** <!-- .element: class="text-lg" -->

---

<!-- .slide: id="protocol-measurement-gap" -->

## What the Benchmarks Leave Out

Labs optimize what they can measure, so the weak spots end up where no benchmark looks.

<div class="card-grid cols-2">
<div class="card"><h4>Measured, so optimized</h4><p>Checkable math, tested code, multiple-choice knowledge.</p></div>
<div class="card warn"><h4>Hard to measure, so left behind</h4><p>Spotting a bad question, judging what matters, writing like a person meant it.</p></div>
</div>

The result is a **spiky** model: expert on the benchmark, oddly off on the task next to it. Saying so can feel unscientific next to a table of high scores. Say it anyway. <!-- .element: class="text-lg" -->
