:::divider id="divider-units" title="What Should a Model Predict?" sub="Tokens, bytes, or whole sentences"
:::

---

<!-- .slide: id="units-tokens" -->

## Every Model in This Course Predicted Tokens

Module 4 called this the **unit problem**. Bytes give tiny vocabularies and very long sequences. Words give huge vocabularies and break on typos. BPE was the compromise.

:::columns cols="2" gap="34px"
**What BPE fixed**

- A vocabulary of tens of thousands of subwords
- Any string stays representable
- Common words cost one token
+++
**What it left behind**

- The vocabulary is frozen before training starts
- Letters hide inside tokens, so spelling and counting characters are hard
- Every token gets one prediction step, whether it is "the" or the key word of a proof
:::

Research now pushes the unit both ways: smaller than a token, and much larger. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="units-bytes" -->

## Byte Models Drop the Tokenizer

<div class="bench-table dense">
<table>
<thead><tr><th>Model</th><th>How it groups bytes</th><th>Result</th></tr></thead>
<tbody>
<tr><td><strong>ByT5</strong> (Google, 2021)</td><td>No grouping. The T5 encoder-decoder runs on raw UTF-8 bytes, with a deeper encoder</td><td>Competitive with token-based mT5 and more robust to typos, but sequences are several times longer and slower to run</td></tr>
<tr><td><strong>MEGABYTE</strong> (Meta, 2023)</td><td>Fixed-size patches of bytes</td><td>Models sequences of over a million bytes</td></tr>
<tr><td><strong>BLT</strong> (Meta, 2024)</td><td>Variable patches, cut where the next byte is hard to predict</td><td>Matches Llama 3 at equal training compute, up to 8B parameters</td></tr>
<tr><td><strong>H-Net</strong> (2025)</td><td>Chunk boundaries learned end to end with the model</td><td>Reports beating a BPE transformer at equal compute</td></tr>
</tbody>
</table>
</div>

Each step keeps raw bytes as input but groups them before the large model sees them, so the sequence stays short. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="units-megabyte" -->

## MEGABYTE Splits Bytes Into Patches

<div class="svg-figure">
<svg viewBox="0 0 1000 285" role="img" aria-label="The bytes of the text 'the cat sat down' are split into four fixed patches of four bytes. Each patch becomes one position for a global transformer. For each patch, the global output feeds a small local model that predicts the bytes of the following patch one at a time.">
  <text x="20" y="257" fill="#f5a623" font-size="18" font-family="Inter, sans-serif">bytes</text>
  <g fill="none" stroke="#f5a623" stroke-width="2"><rect x="100" y="232" width="200" height="36" /><rect x="300" y="232" width="200" height="36" /><rect x="500" y="232" width="200" height="36" /><rect x="700" y="232" width="200" height="36" /></g>
  <g stroke="#2a3450" stroke-width="1.5"><line x1="150" y1="232" x2="150" y2="268" /><line x1="200" y1="232" x2="200" y2="268" /><line x1="250" y1="232" x2="250" y2="268" /><line x1="350" y1="232" x2="350" y2="268" /><line x1="400" y1="232" x2="400" y2="268" /><line x1="450" y1="232" x2="450" y2="268" /><line x1="550" y1="232" x2="550" y2="268" /><line x1="600" y1="232" x2="600" y2="268" /><line x1="650" y1="232" x2="650" y2="268" /><line x1="750" y1="232" x2="750" y2="268" /><line x1="800" y1="232" x2="800" y2="268" /><line x1="850" y1="232" x2="850" y2="268" /></g>
  <g fill="#e8eaf0" font-size="19" font-family="Inter, sans-serif" text-anchor="middle"><text x="125" y="257">t</text><text x="175" y="257">h</text><text x="225" y="257">e</text><text x="275" y="257">␣</text><text x="325" y="257">c</text><text x="375" y="257">a</text><text x="425" y="257">t</text><text x="475" y="257">␣</text><text x="525" y="257">s</text><text x="575" y="257">a</text><text x="625" y="257">t</text><text x="675" y="257">␣</text><text x="725" y="257">d</text><text x="775" y="257">o</text><text x="825" y="257">w</text><text x="875" y="257">n</text></g>
  <text x="920" y="257" fill="#8892a4" font-size="15" font-family="Inter, sans-serif">4 patches</text>
  <g stroke="#8892a4" stroke-width="2" fill="none"><path d="M 200 232 L 200 200" /><path d="M 400 232 L 400 200" /><path d="M 600 232 L 600 200" /><path d="M 800 232 L 800 200" /><path d="M 200 150 L 200 104" /><path d="M 400 150 L 400 104" /><path d="M 600 150 L 600 104" /><path d="M 800 150 L 800 104" /></g>
  <g fill="#8892a4"><path d="M 200 192 l -6 10 l 12 0 z" /><path d="M 400 192 l -6 10 l 12 0 z" /><path d="M 600 192 l -6 10 l 12 0 z" /><path d="M 800 192 l -6 10 l 12 0 z" /><path d="M 200 96 l -6 10 l 12 0 z" /><path d="M 400 96 l -6 10 l 12 0 z" /><path d="M 600 96 l -6 10 l 12 0 z" /><path d="M 800 96 l -6 10 l 12 0 z" /></g>
  <rect x="100" y="150" width="800" height="42" rx="6" fill="none" stroke="#4a9eff" stroke-width="3" />
  <text x="500" y="177" fill="#4a9eff" font-size="18" font-family="Inter, sans-serif" text-anchor="middle">global transformer: one position per patch</text>
  <g fill="none" stroke="#50c878" stroke-width="2"><rect x="115" y="40" width="170" height="54" rx="6" /><rect x="315" y="40" width="170" height="54" rx="6" /><rect x="515" y="40" width="170" height="54" rx="6" /><rect x="715" y="40" width="170" height="54" rx="6" /></g>
  <g fill="#50c878" font-size="16" font-family="Inter, sans-serif" text-anchor="middle"><text x="200" y="64">local model</text><text x="400" y="64">local model</text><text x="600" y="64">local model</text><text x="800" y="64">local model</text><text x="200" y="84" fill="#8892a4" font-size="14">bytes of patch 2</text><text x="400" y="84" fill="#8892a4" font-size="14">bytes of patch 3</text><text x="600" y="84" fill="#8892a4" font-size="14">bytes of patch 4</text><text x="800" y="84" fill="#8892a4" font-size="14">bytes of patch 5</text></g>
</svg>
</div>

- With patches of $P$ bytes, the global model's sequence is $P$ times shorter, so its attention costs $P^2$ times less
- Each small local model predicts the next patch one byte at a time
- Fixed patches ignore the text and can cut a word in half. BLT and H-Net place the boundaries by content

---

<!-- .slide: id="units-concepts" -->

## Concept Models Predict Bigger Units Than Tokens

<div class="svg-figure">
<svg viewBox="0 0 1000 290" role="img" aria-label="The same sentence split five ways, from finest to coarsest. Bytes: one unit per character. Tokens: about one unit per word. Byte Latent Transformer patches: variable-length groups of bytes, short where the text is hard to predict. CALM: one vector per four tokens. Large Concept Model: one vector for the whole sentence.">
  <g font-family="Inter, sans-serif" font-size="18">
    <text x="20" y="44" fill="#8892a4">bytes</text>
    <text x="20" y="98" fill="#4a9eff">tokens</text>
    <text x="20" y="152" fill="#a06bd4">BLT patches</text>
    <text x="20" y="206" fill="#f5a623">CALM vectors</text>
    <text x="20" y="260" fill="#50c878">LCM concept</text>
  </g>
  <g fill="none" stroke="#8892a4" stroke-width="1.5">
    <rect x="200" y="24" width="24" height="30" /><rect x="226" y="24" width="24" height="30" /><rect x="252" y="24" width="24" height="30" /><rect x="278" y="24" width="24" height="30" /><rect x="304" y="24" width="24" height="30" /><rect x="330" y="24" width="24" height="30" /><rect x="356" y="24" width="24" height="30" /><rect x="382" y="24" width="24" height="30" /><rect x="408" y="24" width="24" height="30" /><rect x="434" y="24" width="24" height="30" /><rect x="460" y="24" width="24" height="30" /><rect x="486" y="24" width="24" height="30" /><rect x="512" y="24" width="24" height="30" /><rect x="538" y="24" width="24" height="30" /><rect x="564" y="24" width="24" height="30" /><rect x="590" y="24" width="24" height="30" /><rect x="616" y="24" width="24" height="30" /><rect x="642" y="24" width="24" height="30" /><rect x="668" y="24" width="24" height="30" /><rect x="694" y="24" width="24" height="30" /><rect x="720" y="24" width="24" height="30" /><rect x="746" y="24" width="24" height="30" /><rect x="772" y="24" width="24" height="30" /><rect x="798" y="24" width="24" height="30" /><rect x="824" y="24" width="24" height="30" /><rect x="850" y="24" width="24" height="30" /><rect x="876" y="24" width="24" height="30" /><rect x="902" y="24" width="24" height="30" /><rect x="928" y="24" width="24" height="30" /><rect x="954" y="24" width="24" height="30" />
  </g>
  <g fill="none" stroke="#4a9eff" stroke-width="2">
    <rect x="200" y="78" width="102" height="30" /><rect x="304" y="78" width="128" height="30" /><rect x="434" y="78" width="76" height="30" /><rect x="512" y="78" width="154" height="30" /><rect x="668" y="78" width="102" height="30" /><rect x="772" y="78" width="128" height="30" /><rect x="902" y="78" width="76" height="30" />
  </g>
  <g fill="none" stroke="#a06bd4" stroke-width="2">
    <rect x="200" y="132" width="232" height="30" /><rect x="434" y="132" width="50" height="30" /><rect x="486" y="132" width="24" height="30" /><rect x="512" y="132" width="258" height="30" /><rect x="772" y="132" width="206" height="30" />
  </g>
  <g fill="none" stroke="#f5a623" stroke-width="2">
    <rect x="200" y="186" width="438" height="30" /><rect x="642" y="186" width="336" height="30" />
  </g>
  <rect x="200" y="240" width="778" height="30" fill="none" stroke="#50c878" stroke-width="3" />
  <text x="589" y="261" fill="#50c878" font-size="17" font-family="Inter, sans-serif" text-anchor="middle">one 1024-number vector for the whole sentence</text>
</svg>
</div>

- **BLT** (Meta, 2024) groups bytes into patches, making a patch shorter wherever the next byte is hard to predict
- **CALM** (2025) compresses every 4 tokens into one vector, cutting the number of generation steps by 4
- **Large Concept Models** (Meta, 2024) predict the embedding of the **next sentence**

---

<!-- .slide: id="units-lcm" -->

## Concept Models Plan by the Sentence

<div class="svg-figure">
<svg viewBox="0 0 1000 230" role="img" aria-label="Sentences are encoded by a frozen SONAR encoder into one vector each. A transformer reads the sequence of sentence vectors and predicts the next one. A frozen SONAR decoder turns the predicted vector back into text in any of its supported languages.">
  <rect x="20" y="70" width="160" height="70" fill="none" stroke="#8892a4" stroke-width="2" />
  <text x="100" y="100" fill="#e8eaf0" font-size="18" font-family="Inter, sans-serif" text-anchor="middle">sentences</text>
  <text x="100" y="124" fill="#8892a4" font-size="15" font-family="Inter, sans-serif" text-anchor="middle">text or speech</text>
  <path d="M 180 105 L 212 105" stroke="#8892a4" stroke-width="2" fill="none" />
  <path d="M 220 105 l -12 -6 l 0 12 z" fill="#8892a4" />
  <rect x="220" y="70" width="170" height="70" fill="none" stroke="#4a9eff" stroke-width="2" />
  <text x="305" y="100" fill="#4a9eff" font-size="18" font-family="Inter, sans-serif" text-anchor="middle">SONAR encoder</text>
  <text x="305" y="124" fill="#8892a4" font-size="15" font-family="Inter, sans-serif" text-anchor="middle">frozen</text>
  <path d="M 390 105 L 422 105" stroke="#8892a4" stroke-width="2" fill="none" />
  <path d="M 430 105 l -12 -6 l 0 12 z" fill="#8892a4" />
  <rect x="430" y="60" width="200" height="90" fill="none" stroke="#50c878" stroke-width="3" />
  <text x="530" y="98" fill="#50c878" font-size="19" font-family="Inter, sans-serif" text-anchor="middle">Concept model</text>
  <text x="530" y="124" fill="#8892a4" font-size="15" font-family="Inter, sans-serif" text-anchor="middle">predicts the next vector</text>
  <path d="M 630 105 L 662 105" stroke="#8892a4" stroke-width="2" fill="none" />
  <path d="M 670 105 l -12 -6 l 0 12 z" fill="#8892a4" />
  <rect x="670" y="70" width="170" height="70" fill="none" stroke="#4a9eff" stroke-width="2" />
  <text x="755" y="100" fill="#4a9eff" font-size="18" font-family="Inter, sans-serif" text-anchor="middle">SONAR decoder</text>
  <text x="755" y="124" fill="#8892a4" font-size="15" font-family="Inter, sans-serif" text-anchor="middle">frozen</text>
  <path d="M 840 105 L 872 105" stroke="#8892a4" stroke-width="2" fill="none" />
  <path d="M 880 105 l -12 -6 l 0 12 z" fill="#8892a4" />
  <text x="886" y="111" fill="#e8eaf0" font-size="17" font-family="Inter, sans-serif">next sentence</text>
  <text x="500" y="200" fill="#8892a4" font-size="17" font-family="Inter, sans-serif" text-anchor="middle">The model never sees a token. Text appears only at the decoder.</text>
</svg>
</div>

:::columns cols="2" gap="34px"
**Why it might help**

- A ten-sentence plan is ten predictions, not thousands
- One concept space covers SONAR's 200 languages
- Close to LeCun's argument for predicting in representation space
+++
**What is still missing**

- There is no exact likelihood, so training uses regression or diffusion on vectors
- Most experiments use 1.6B-parameter models, small by current standards
- SONAR-LLM (2025) adds a token-level loss back through the decoder
:::
