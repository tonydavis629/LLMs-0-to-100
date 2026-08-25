:::divider id="divider-attention" title="Is Attention the Right Primitive?" sub="What replaces quadratic attention"
:::

---

<!-- .slide: id="attention-bill" -->

## The Cost of Attention

Two costs, both paid on every request.

:::columns cols="2" gap="34px"
**Compute: $O(n^2)$**

- Score matrix is $n \times n$
- Double the context, quadruple the work
- FlashAttention shrank the constant, not the exponent
+++
**Memory: the KV cache**

- Stores a key and value per generated token
- Grows linearly with sequence length
- Dominates serving memory at long context
:::

Removing these costs requires giving something up. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="attention-drop-softmax" -->

## Linear Attention: Removing the Softmax

Replace the exponential with any positive feature map $\phi$. The same computation now has **two forms.**

$$\mathbf y_t = \frac{\phi(\mathbf q_t)^\top \sum_{i \le t} \phi(\mathbf k_i) \mathbf v_i^\top}{\phi(\mathbf q_t)^\top \sum_{i \le t} \phi(\mathbf k_i)}$$

:::columns cols="2" gap="34px"
**As a matrix**

- Build the $n \times n$ score matrix, mask, normalize, multiply by values
- Standard attention minus the softmax
- Cost: $O(n^2)$
+++
**As a running sum**

- Keep $\mathbf S_t = \mathbf S_{t-1} + \phi(\mathbf k_t)\mathbf v_t^\top$
- An **RNN** with a matrix hidden state
- Cost: $O(n)$ time, $O(1)$ memory
:::

Both produce the same numbers. Katharopoulos et al. (2020) titled the paper "Transformers are RNNs": the equivalence is a theorem. The exercise has you check it. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="attention-grouping" -->

## Associativity Determines the Cost

<div class="svg-figure">
<svg viewBox="0 0 1000 400" role="img" aria-label="Two ways to group the same matrix product. Grouping left builds an n by n intermediate matrix, 268 megabytes at 8192 tokens. Grouping right builds a d by d intermediate state, 16 kilobytes regardless of sequence length.">
  <text x="20" y="42" fill="#8892a4" font-size="20" font-family="Inter, sans-serif">Group to the left</text>
  <text x="20" y="86" fill="#e8eaf0" font-size="30" font-family="Inter, sans-serif">(  &#966;(Q) &#966;(K)&#7488;  ) V</text>
  <path d="M 300 76 L 372 76" stroke="#8892a4" stroke-width="2" fill="none" />
  <path d="M 372 76 l -10 -5 l 0 10 z" fill="#8892a4" />
  <rect x="392" y="12" width="130" height="130" fill="none" stroke="#f5a623" stroke-width="3" />
  <text x="457" y="84" fill="#f5a623" font-size="26" font-family="Inter, sans-serif" text-anchor="middle">n &#215; n</text>
  <text x="556" y="60" fill="#e8eaf0" font-size="22" font-family="Inter, sans-serif">8192 &#215; 8192 at 8k context</text>
  <text x="556" y="94" fill="#f5a623" font-size="22" font-family="Inter, sans-serif">268 MB, and it grows with n&#178;</text>
  <line x1="20" y1="196" x2="980" y2="196" stroke="#2a3450" stroke-width="2" />
  <text x="20" y="248" fill="#8892a4" font-size="20" font-family="Inter, sans-serif">Group to the right</text>
  <text x="20" y="292" fill="#e8eaf0" font-size="30" font-family="Inter, sans-serif">&#966;(Q) (  &#966;(K)&#7488; V  )</text>
  <path d="M 300 282 L 372 282" stroke="#8892a4" stroke-width="2" fill="none" />
  <path d="M 372 282 l -10 -5 l 0 10 z" fill="#8892a4" />
  <rect x="392" y="270" width="26" height="26" fill="none" stroke="#50c878" stroke-width="3" />
  <text x="440" y="292" fill="#50c878" font-size="26" font-family="Inter, sans-serif">d &#215; d</text>
  <text x="556" y="266" fill="#e8eaf0" font-size="22" font-family="Inter, sans-serif">64 &#215; 64, at every context length</text>
  <text x="556" y="300" fill="#50c878" font-size="22" font-family="Inter, sans-serif">16 KB, and it never grows</text>
</svg>
</div>

Matrix multiplication is associative, so both orders give the same answer. Only one of them ever builds the big object. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="attention-softmax-price" -->

## What the Softmax Provides

Dropping it costs two things.

:::columns cols="2" gap="34px"
**Sharpness**

- The exponential lets **one key dominate**
- Attention can select one token out of thousands
- A linear kernel spreads weight across many keys
+++
**Unbounded memory**

- All knowledge of the past must fit in $\mathbf S$, a **fixed-size** matrix
- A KV cache reproduces any earlier token exactly
- A fixed state cannot
:::

This is the RNN bottleneck, reintroduced deliberately. Open question: is a fixed-size summary good enough? <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

:::figure img="images/albert_gu.jpg" name="Albert Gu" kicker="S4 (2021), Mamba (2023)" alt="Albert Gu"
Built the state-space line, the most serious architectural challenger to the transformer since 2017. S4 derived a sequence model from a continuous-time dynamical system, an idea from control theory, not NLP. Mamba, with Tri Dao, made state transitions depend on the input and became competitive at scale.

Co-author Tri Dao wrote FlashAttention, the work that made exact attention fast on real hardware. The person who made attention efficient is also building its replacement. This is an engineering argument, not a fight between camps.
:::

---

<!-- .slide: id="attention-ssm" -->

## State-Space Models

S4 derives the recurrence from a continuous-time system. **Mamba** adds selectivity: state transitions that depend on the input.

<div class="svg-figure">
<svg viewBox="0 0 1000 340" role="img" aria-label="Two recurrences compared. S4 applies the same fixed update for every token. Mamba's update depends on the token: some tokens are stored into the state with a strong update, others are mostly forgotten.">
  <text x="20" y="34" fill="#8892a4" font-size="21" font-family="Inter, sans-serif">S4: fixed recurrence, the same update for every token</text>
  <g fill="none" stroke="#4a9eff" stroke-width="2">
    <rect x="40" y="66" width="110" height="48" /><rect x="290" y="66" width="110" height="48" /><rect x="540" y="66" width="110" height="48" /><rect x="790" y="66" width="110" height="48" />
  </g>
  <g fill="#e8eaf0" font-size="22" font-family="Inter, sans-serif" text-anchor="middle">
    <text x="95" y="97">h&#8320;</text><text x="345" y="97">h&#8321;</text><text x="595" y="97">h&#8322;</text><text x="845" y="97">h&#8323;</text>
  </g>
  <g stroke="#4a9eff" stroke-width="3" fill="none">
    <path d="M 150 90 L 280 90" /><path d="M 400 90 L 530 90" /><path d="M 650 90 L 780 90" />
  </g>
  <g fill="#4a9eff">
    <path d="M 280 90 l -12 -6 l 0 12 z" /><path d="M 530 90 l -12 -6 l 0 12 z" /><path d="M 780 90 l -12 -6 l 0 12 z" />
  </g>
  <g fill="#8892a4" font-size="19" font-family="Inter, sans-serif" text-anchor="middle">
    <text x="215" y="60">x&#8321;</text><text x="465" y="60">x&#8322;</text><text x="715" y="60">x&#8323;</text>
  </g>
  <line x1="20" y1="160" x2="980" y2="160" stroke="#2a3450" stroke-width="2" />
  <text x="20" y="204" fill="#8892a4" font-size="21" font-family="Inter, sans-serif">Mamba: selective recurrence, the update depends on the token</text>
  <g fill="none" stroke="#50c878" stroke-width="2">
    <rect x="40" y="236" width="110" height="48" /><rect x="290" y="236" width="110" height="48" /><rect x="540" y="236" width="110" height="48" /><rect x="790" y="236" width="110" height="48" />
  </g>
  <g fill="#e8eaf0" font-size="22" font-family="Inter, sans-serif" text-anchor="middle">
    <text x="95" y="267">h&#8320;</text><text x="345" y="267">h&#8321;</text><text x="595" y="267">h&#8322;</text><text x="845" y="267">h&#8323;</text>
  </g>
  <path d="M 150 260 L 280 260" stroke="#50c878" stroke-width="5" fill="none" />
  <path d="M 280 260 l -12 -7 l 0 14 z" fill="#50c878" />
  <path d="M 400 260 L 530 260" stroke="#8892a4" stroke-width="2" stroke-dasharray="7 6" fill="none" />
  <path d="M 530 260 l -12 -6 l 0 12 z" fill="#8892a4" />
  <path d="M 650 260 L 780 260" stroke="#50c878" stroke-width="5" fill="none" />
  <path d="M 780 260 l -12 -7 l 0 14 z" fill="#50c878" />
  <g font-size="19" font-family="Inter, sans-serif" text-anchor="middle">
    <text x="215" y="230" fill="#50c878">x&#8321; stored</text><text x="465" y="230" fill="#8892a4">x&#8322; forgotten</text><text x="715" y="230" fill="#50c878">x&#8323; stored</text>
  </g>
</svg>
</div>

Selectivity is the LSTM's gating idea. Mamba keeps training parallel with an associative scan. RWKV reached a similar design independently. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="attention-verdict" -->

## The Empirical Results

Where pure recurrent models stand against transformers, by task type.

<div class="bench-table">
<table>
<thead><tr><th>Task type</th><th>Pure recurrent models</th><th>Why</th></tr></thead>
<tbody>
<tr><td>Language modeling perplexity</td><td><strong>Competitive</strong> at modest scale</td><td>Most next-token prediction needs recent context, which a state summarizes fine</td></tr>
<tr><td>Exact recall of an earlier token</td><td><strong>Behind</strong></td><td>The token was compressed into a fixed state; it cannot be reconstructed</td></tr>
<tr><td>Copying, needle-in-a-haystack</td><td><strong>Behind</strong></td><td>Same reason. Random access is what was traded away</td></tr>
</tbody>
</table>
</div>

A fixed-size state predicts exactly this failure pattern, and the benchmarks confirm it. <!-- .element: class="text-lg" style="margin-top: 14px;" -->

---

<!-- .slide: id="attention-hybrids" -->

## Production Models Use Hybrids

**Jamba** and **Griffin** interleave a few attention layers among many recurrent ones.

<div class="svg-figure">
<svg viewBox="0 0 1000 360" role="img" aria-label="A hybrid model's layer stack: eight layers, six recurrent and two attention. The attention layers provide random access to any earlier token at quadratic cost; the recurrent layers provide constant cost per token but no exact recall.">
  <g fill="none" stroke="#50c878" stroke-width="2">
    <rect x="120" y="304" width="260" height="30" /><rect x="120" y="266" width="260" height="30" /><rect x="120" y="228" width="260" height="30" /><rect x="120" y="152" width="260" height="30" /><rect x="120" y="114" width="260" height="30" /><rect x="120" y="76" width="260" height="30" />
  </g>
  <g fill="none" stroke="#4a9eff" stroke-width="3">
    <rect x="120" y="190" width="260" height="30" /><rect x="120" y="38" width="260" height="30" />
  </g>
  <g fill="#8892a4" font-size="18" font-family="Inter, sans-serif" text-anchor="middle">
    <text x="250" y="324">recurrent</text><text x="250" y="286">recurrent</text><text x="250" y="248">recurrent</text><text x="250" y="172">recurrent</text><text x="250" y="134">recurrent</text><text x="250" y="96">recurrent</text>
  </g>
  <g fill="#4a9eff" font-size="18" font-family="Inter, sans-serif" text-anchor="middle">
    <text x="250" y="210">attention</text><text x="250" y="58">attention</text>
  </g>
  <path d="M 400 205 L 480 205" stroke="#4a9eff" stroke-width="2" fill="none" />
  <text x="500" y="196" fill="#4a9eff" font-size="21" font-family="Inter, sans-serif">attention: random access to any earlier token</text>
  <text x="500" y="226" fill="#8892a4" font-size="19" font-family="Inter, sans-serif">cost O(n&#178;), KV cache grows with context</text>
  <path d="M 400 281 L 480 281" stroke="#50c878" stroke-width="2" fill="none" />
  <text x="500" y="272" fill="#50c878" font-size="21" font-family="Inter, sans-serif">recurrence: constant cost per token</text>
  <text x="500" y="302" fill="#8892a4" font-size="19" font-family="Inter, sans-serif">fixed-size state, no exact recall</text>
</svg>
</div>

Attention only where random access is needed. The same pattern appears in the next section. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="sidequest-bitter" -->

## Side Quest: The Bitter Lesson, Applied Here

Sutton's essay: for seventy years, general methods that leverage computation beat methods built on human insight. Discussion questions:

- Is Mamba's **selectivity** human-designed structure the lesson eliminates, or a better way to spend compute, which the lesson rewards?
- Hybrid layer allocations are **hand-designed**. Somebody picked "one attention layer in eight." Does that survive?
- Was **attention itself** the human-designed structure? The bitter lesson already displaced recurrent models once.

No answer key. "Does this scale" and "is this clever" are different questions. Only the first has predicted the winner. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="attention-transition" -->

## Three Shared Assumptions

Every model in this section, transformer and state-space alike, still does all three:

<div class="card-grid cols-3">
<div class="card"><h4>Generates left to right</h4><p>One token at a time, never revising.</p></div>
<div class="card"><h4>Spends fixed compute per token</h4><p>The same layers, every time, regardless of difficulty.</p></div>
<div class="card"><h4>Freezes its weights</h4><p>Training ends, and learning ends with it.</p></div>
</div>

The next three sections question those. <!-- .element: class="text-lg" style="margin-top: 14px;" -->
