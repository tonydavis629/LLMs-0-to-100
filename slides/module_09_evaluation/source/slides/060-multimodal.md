:::divider id="divider-multimodal" title="Evaluating Multimodal Models" sub="The answer depends on something a string cannot see"
:::

---

<!-- .slide: id="mm-why-harder" -->

## Why Multimodal Scoring Is Harder

Module 8 built models that take images. Grading them is harder for two reasons:

<div class="card-grid cols-2">
<div class="card"><h4>The evidence is not in the text</h4><p>Correctness depends on <strong>visual detail</strong> (spatial relations, small print, a chart's axis) that no string comparison can inspect.</p></div>
<div class="card"><h4>Free-form visual answers break exact match</h4><p>"two" and "there are two people" are the same answer. Multimodal suites lean on <strong>multiple choice and LLM judges</strong> more than text suites do.</p></div>
</div>

---

<!-- .slide: id="mm-understanding" -->

## Understanding Benchmarks

<div class="bench-table dense">
<table>
<thead><tr><th>Benchmark</th><th>Year</th><th>What it stresses</th></tr></thead>
<tbody>
<tr><td><strong>VQAv2</strong></td><td>2017</td><td>Short-answer questions about natural images, scored against ten human answers. Defined visual question answering.</td></tr>
<tr><td><strong>TextVQA, DocVQA, ChartQA, AI2D</strong></td><td>2019&ndash;2022</td><td>Reading text in images, documents, charts, and diagrams. Where OCR ability shows up, and what most business use of vision models needs.</td></tr>
<tr><td><strong>MMMU</strong></td><td>2023</td><td>College-level multi-discipline questions with figures and diagrams. The current headline multimodal number.</td></tr>
<tr><td><strong>MathVista</strong></td><td>2023</td><td>Mathematical reasoning over visual inputs.</td></tr>
<tr><td><strong>MMBench, MME</strong></td><td>2023</td><td>Broad capability suites with per-ability breakdowns: recognition, spatial relations, counting, OCR.</td></tr>
<tr><td><strong>Video-MME</strong></td><td>2024</td><td>Video understanding: temporal reasoning over much longer inputs.</td></tr>
<tr><td><strong>POPE</strong></td><td>2023</td><td>Hallucination probe. Ask about objects that are <strong>not in the image</strong> and see whether the model agrees they are.</td></tr>
</tbody>
</table>
</div>

---

<!-- .slide: id="mm-generation" -->

## Captioning and Generation Need Different Machinery

<div class="card-grid cols-4">
<div class="card"><h4>CIDEr, SPICE</h4><p>Captioning on MS COCO, scored against several reference captions.</p></div>
<div class="card"><h4>CLIPScore</h4><p>Uses a contrastive image-text model (Module 8) to measure image-caption agreement. <strong>No reference caption is needed.</strong></p></div>
<div class="card"><h4>FID</h4><p>Image generation: distributional similarity to real images. Human preference is still the standard for quality.</p></div>
<div class="card"><h4>Word error rate</h4><p>Speech recognition. One of the oldest and cleanest automatic metrics in the field.</p></div>
</div>

The instruction-tuned pattern repeats: with a reference, compare to it; without one, ask a judge to compare two outputs. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="sq-no-image" -->

## Side Quest: Can It Be Answered Without the Image?

Many visual benchmarks can be **partly answered from language priors alone**. "What color is the banana?" has a good guess that never looks at the picture.

<div class="card-grid cols-2">
<div class="card"><h4>The ablation</h4><p>Run the benchmark with the <strong>image removed</strong>, leaving only the question and options. Whatever score survives came from the text.</p></div>
<div class="card warn"><h4>What people find</h4><p>On several popular suites, a text-only model scores far above chance. The reported number is partly a <strong>language</strong> benchmark.</p></div>
</div>

Module 8 fixed this shortcut by balancing the dataset: every answer equally likely a priori, so the only way to be right is to look. <!-- .element: class="text-lg" -->
