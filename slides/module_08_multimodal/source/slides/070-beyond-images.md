:::divider id="divider-beyond" title="Beyond Still Images" sub="Audio, speech, video, and sensors"
:::

---

<!-- .slide: id="beyond-audio" -->

## Audio Begins as a Waveform

A waveform is a **sequence of amplitude values**, thousands per second. Too raw and too long to feed directly, so we re-represent it:

<div class="card-grid cols-3">
<div class="card"><h4>Spectrograms</h4><p>Time-frequency images the model can patch like a picture</p></div>
<div class="card"><h4>Learned embeddings</h4><p>Continuous features from an audio encoder</p></div>
<div class="card"><h4>Codec tokens</h4><p>Discrete codebook ids, so audio becomes a token sequence</p></div>
</div>

Same pattern as images: **compress the high-bandwidth signal into a sequence**, discrete or continuous. <!-- .element: class="text-lg" -->

---

:::manim id="beyond-audio-anim" scene="audiotokens"
:::

---

<!-- .slide: id="fig-whisper" -->

:::figure img="images/radford.jpg" name="Alec Radford and the Whisper team" kicker="Robust Speech Recognition via Large-Scale Weak Supervision (Whisper, OpenAI, 2022)"
- Trained speech recognition on 680,000 hours of noisy, multilingual web audio
- Reached robust transcription without a small, clean, hand-labeled corpus
- Same trade as CLIP: many noisy labels instead of a few clean ones
:::

---

<!-- .slide: id="beyond-audio-continuous" -->

## Audio as Continuous Vectors

<div class="img-figure"><img src="images/arch_audio_enc.svg" alt="The carried diagram for audio: waveform to a Whisper-style encoder to pooling and projector to the language model" style="max-height: 330px;"></div>

- Audio becomes continuous vectors, so it needs a projector to reach the LLM's width, just as image patches did
- Whisper's encoder turns 30 s into 1,500 vectors; pooling halves that before the LLM
- Good for **understanding** speech, but the LLM can only answer in text

---

<!-- .slide: id="beyond-audio-codec" -->

## Audio as Discrete Codec Tokens

<div class="img-figure"><img src="images/arch_audio_codec.svg" alt="The carried diagram for codec audio: a neural codec turns audio into codebook ids that share the embedding table with text; the model can output audio tokens that a codec decoder turns back into sound" style="max-height: 330px;"></div>

- A neural codec quantizes audio the way a VQ-VAE quantizes images; the ids join the vocabulary, so the model can **speak** by predicting audio tokens
- The tokens keep tone, timing, and speaker identity, which a transcript throws away
- Spoken assistants such as Moshi (Kyutai, 2024) work this way
- The two audio slides differ in **representation**, not training recipe. Moshi adds codec tokens to a pretrained text LLM; it is not trained from scratch like Chameleon

---

<!-- .slide: id="beyond-video" -->

## Video Adds a Time Axis

<div class="img-figure"><img src="images/arch_video.svg" alt="The carried diagram for video: sample frames, encode each with an image encoder, pool over time, project into the language model" style="max-height: 320px;"></div>

Sample frames, encode each with an image encoder, add **temporal** positions, pool, project. Tokens grow with **frames &times; resolution &times; patch density**, so a short clip can outweigh a long document. <!-- .element: class="text-lg" -->

---

:::manim id="beyond-budget-anim" scene="videobudget"
:::

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

Context length is only half the problem: attention cost grows with the square of the sequence length. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="beyond-temporal" -->

## Why Time Is Hard for a Sequence Model

LLMs handle sequences well when each token carries meaning. Video frames do not:

<div class="img-figure"><img src="images/frame_sampling.svg" alt="To-scale ten second timeline: frames sampled once per second; a half-second door-opening event falls between the 4 s and 5 s samples" style="max-height: 230px;"></div>

- Neighboring frames are nearly identical. The information is in the small **differences**, and sampling and pooling discard those first
- Each frame is encoded on its own, so motion must be inferred from a few compressed snapshots
- Training rarely demands more: many video benchmarks can be answered from **one frame** (Lei et al., 2022; Buch et al., 2022)

---

<!-- .slide: id="beyond-sensors" -->

## Other Streams, Same Bridge

:::columns cols="2" gap="30px"
**Rate mismatch**

- Audio-video models align streams that arrive at different rates
- A frame every ~33 ms; audio samples thousands of times per second
+++
**Sensors generalize the idea**

- Depth, thermal, radar/RF, lidar, robot joint states, medical images, time series
- The representation changes; the **bridge problem stays the same**: an encoder, a connector, and the same language model
:::
