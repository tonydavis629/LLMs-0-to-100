:::divider id="divider-multimodal" title="Evaluating Multimodal Models" sub="The answer depends on something a string cannot see"
:::

---

<!-- .slide: id="mm-why-harder" -->

## Why Multimodal Scoring Is Harder

<div class="card-grid cols-2">
<div class="card"><h4>The evidence is in the image</h4><p>Correctness depends on <strong>visual detail</strong> that no string comparison can check.</p></div>
<div class="card"><h4>Free-form answers vary</h4><p>"two" and "there are two people" match. Suites lean on <strong>multiple choice and LLM judges</strong>.</p></div>
</div>

---

<!-- .slide: id="mm-understanding" -->

## Understanding Benchmarks

<div class="bench-table dense">
<table>
<thead><tr><th>Benchmark</th><th>Year</th><th>What it stresses</th><th>Saturated</th></tr></thead>
<tbody>
<tr><td><strong>VQAv2</strong></td><td>2017</td><td>Short questions about photos.</td><td>~2022</td></tr>
<tr><td><strong>TextVQA, DocVQA, ChartQA, AI2D</strong></td><td>2019&ndash;2022</td><td>Reading text in images, documents, and charts.</td><td>Near ceiling</td></tr>
<tr><td><strong>MMMU</strong></td><td>2023</td><td>College-level questions with figures.</td><td>Past the median expert, 2025</td></tr>
<tr><td><strong>MathVista</strong></td><td>2023</td><td>Math over visual inputs.</td><td>2024</td></tr>
<tr><td><strong>MMBench, MME</strong></td><td>2023</td><td>Broad suites, scored per ability.</td><td>Near ceiling</td></tr>
<tr><td><strong>Video-MME</strong></td><td>2024</td><td>Reasoning over long videos.</td><td>Near ceiling</td></tr>
<tr><td><strong>POPE</strong></td><td>2023</td><td>"Is there a &lt;object&gt; in the image?" for objects that <strong>are not there</strong>.</td><td>Near ceiling</td></tr>
</tbody>
</table>
</div>

---

<!-- .slide: id="mm-open" -->

## Multimodal Benchmarks That Are Still Open

<div class="bench-table">
<table>
<thead><tr><th>Benchmark</th><th>Year</th><th>What it measures</th><th>Best score, late 2026</th></tr></thead>
<tbody>
<tr><td><strong>MMMU-Pro</strong></td><td>2024</td><td>MMMU with ten options and the question inside the image.</td><td>Near the best human experts</td></tr>
<tr><td><strong>Video-MME-v2</strong></td><td>2026</td><td>Harder questions over long videos.</td><td>~49%, against ~91% for experts</td></tr>
</tbody>
</table>
</div>

---

<!-- .slide: id="mm-generation" -->

## Captioning and Generation Need Different Machinery

<div class="card-grid cols-4">
<div class="card"><h4>CIDEr, SPICE</h4><p>Captions, compared to reference captions.</p></div>
<div class="card"><h4>CLIPScore</h4><p>Image-caption agreement from a CLIP model. <strong>No reference needed.</strong></p></div>
<div class="card"><h4>FID</h4><p>How close generated images are, as a set, to real ones.</p></div>
<div class="card"><h4>Word error rate</h4><p>Speech recognition.</p></div>
</div>

Same pattern as text: compare to a reference if you have one, otherwise use a judge. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="sq-no-image" -->

## <span class="side-quest-badge">Side Quest</span> Can It Be Answered Without the Image?

"What color is the banana?" can be answered without looking. Run a visual benchmark with the **image removed** and see what score survives.

<div class="card-grid cols-2">
<div class="card"><h4>What people find</h4><p>Text-only models score far above chance on several popular suites.</p></div>
<div class="card"><h4>The fix (Module 8)</h4><p>Balance the dataset so the only way to be right is to look.</p></div>
</div>
