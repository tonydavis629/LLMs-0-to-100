:::divider id="divider-beyond" title="Beyond Still Images" sub="Audio, speech, video, and sensors"
:::

---

<!-- .slide: id="beyond-audio" -->

## Audio Begins as a Waveform

A waveform is a **sequence of amplitude values**, thousands per second. Too raw and too long to feed directly, so we re-represent it:

<div class="card-grid cols-3">
<div class="card"><h4>Spectrograms</h4><p>Time-frequency images the model can patch like a picture</p></div>
<div class="card"><h4>Learned embeddings</h4><p>Continuous features from an audio encoder</p></div>
<div class="card"><h4>Codec tokens</h4><p>Discrete codebook entries &mdash; audio as a token sequence</p></div>
</div>

Same pattern as images: **compress the high-bandwidth signal into a sequence**, discrete or continuous. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="beyond-speech" -->

## Speech, Several Ways

<div class="taxonomy-table">
<table>
<thead><tr><th>Framing</th><th>Maps</th></tr></thead>
<tbody>
<tr><td><strong>ASR</strong> (recognition)</td><td>audio &rarr; text</td></tr>
<tr><td><strong>TTS</strong> (synthesis)</td><td>text &rarr; audio</td></tr>
<tr><td><strong>Speech-language model</strong></td><td>speech &rarr; text or speech</td></tr>
<tr><td><strong>Spoken assistant</strong></td><td>handles content <em>and</em> paralinguistics: timing, tone, speaker turns</td></tr>
</tbody>
</table>
</div>

A spoken assistant models more than the words: **how** something is said carries meaning too. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="fig-whisper" -->

:::figure img="images/radford.jpg" name="Alec Radford and the Whisper team" kicker="Robust Speech Recognition via Large-Scale Weak Supervision (Whisper, OpenAI, 2022)"
- Trained speech recognition on 680,000 hours of noisy, multilingual web audio
- Reached robust transcription without a small, clean, hand-labeled corpus
- The same lesson as CLIP: weak, abundant supervision at scale wins
:::

---

<!-- .slide: id="beyond-video" -->

## Video Adds a Time Axis

- **Sample frames**, often sparsely
- Encode each with an image encoder
- Add **temporal** position information
- Feed the features into the language model

The catch: tokens grow with **frames &times; resolution &times; patch density**. A short clip can dwarf a long text document. <!-- .element: class="text-lg" style="margin-top: 8px;" -->

---

:::manim id="beyond-budget-anim" scene="videobudget"
:::

---

<!-- .slide: id="beyond-budget-math" -->

## Brutal Token Cost

A 10-second clip at 30 fps is **300 frames**. At $224 \times 224$ with $16 \times 16$ patches, each frame is 196 tokens:

$$300 \times 196 = \textcolor{#f5a623}{58{,}800} \text{ visual tokens}$$

**More than half** a 100k context window, before the prompt, answer, or audio. Double the resolution to $448 \times 448$ and each frame is 784 tokens:

$$300 \times 784 = \textcolor{#e05252}{235{,}200} \text{ visual tokens}$$

Now it **does not fit** in 100k at all. Attention cost scales with the **square** of sequence length. <!-- .element: class="text-lg" style="margin-top: 8px;" -->

---

<!-- .slide: id="beyond-compress" -->

## So Video Models Compress Aggressively

<div class="card-grid cols-3">
<div class="card"><h4>Sample frames</h4><p>Keep a sparse subset, not every frame</p></div>
<div class="card"><h4>Pool patches</h4><p>Merge neighboring patch tokens</p></div>
<div class="card"><h4>Learned queries</h4><p>A few tokens summarize many (Q-Former style)</p></div>
<div class="card"><h4>Temporal encoders</h4><p>Compress across time before the LLM</p></div>
<div class="card"><h4>Retrieve segments</h4><p>Feed only the relevant part of the clip</p></div>
<div class="card"><h4>Longer context</h4><p>Gemini-scale windows push the ceiling up</p></div>
</div>

Context length is only half the problem. The quadratic attention cost comes with it. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="beyond-temporal" -->

## The Hard Part Is Time, Not Objects

Seeing objects frame by frame is easy. Many questions require **temporal reasoning**:

- What **changed** between these moments?
- Which event **caused** another?
- What happened **before** the person opened the door?
- Did the action **complete** successfully?

:::columns cols="2" gap="30px"
**Rate mismatch**

- Audio-video models align streams at different rates
- A frame every ~33 ms; audio samples thousands of times per second
+++
**Sensors generalize the idea**

- Depth, thermal, radar/RF, lidar, robot joint states, medical images, time series
- The representation changes; the **bridge problem stays the same**
:::
