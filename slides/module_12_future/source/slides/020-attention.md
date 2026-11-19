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

## Linear Attention Removes the Softmax

$$\text{softmax attention:} \quad \mathbf y_t = \frac{\sum_{i \le t} \textcolor{#f5a623}{\exp(\mathbf q_t^\top \mathbf k_i)} \mathbf v_i}{\sum_{i \le t} \textcolor{#f5a623}{\exp(\mathbf q_t^\top \mathbf k_i)}}$$

$$\text{linear attention:} \quad \mathbf y_t = \frac{\sum_{i \le t} \textcolor{#f5a623}{\phi(\mathbf q_t)^\top \phi(\mathbf k_i)} \mathbf v_i}{\sum_{i \le t} \textcolor{#f5a623}{\phi(\mathbf q_t)^\top \phi(\mathbf k_i)}}$$

:::columns cols="2" gap="34px"
**The terms**

- $\mathbf q_t$: the query of the current token $t$. $\mathbf k_i, \mathbf v_i$: the key and value of an earlier token $i$
- $\phi$: a function applied to each entry that keeps it positive. A common choice equals $x + 1$ for $x > 0$ and $e^x$ for $x \le 0$
- Top: earlier values, weighted by score. Bottom: the total weight, so the weights add to 1
+++
**Why the change matters**

- $\exp(\mathbf q_t^\top \mathbf k_i)$ cannot be split into a query part and a key part
- In the new score, $\phi(\mathbf q_t)$ is the same for every $i$, so it factors out of the sum
- The query still sets the output. It is applied once, after the sum, instead of once per earlier token
:::

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

Matrix multiplication is associative, so both orders give the same answer. Only one of them ever builds the big object. Katharopoulos et al. (2020) called the paper "Transformers are RNNs" for this reason. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="attention-kv-memory" -->

## The State Works Like a Lookup Table

<div class="svg-figure">
<svg viewBox="0 0 1000 310" role="img" aria-label="Three tokens are written into the state. Each has a one-hot key: city is (1,0,0) with value Paris, color is (0,1,0) with value blue, time is (0,0,1) with value 3pm. Adding k times v transpose puts each value into the row its key points at, so S becomes a table with rows Paris, blue, and 3pm. Reading with the query (0,1,0) multiplies out to row two, which returns blue.">
  <text x="20" y="30" fill="#8892a4" font-size="19" font-family="Inter, sans-serif">Write each token: S += k vᵀ</text>
  <g font-family="Inter, sans-serif" font-size="19">
    <text x="20" y="85" fill="#f5a623">city</text><text x="100" y="85" fill="#e8eaf0">k = (1, 0, 0)</text><text x="250" y="85" fill="#e8eaf0">v = Paris</text>
    <text x="20" y="135" fill="#4a9eff">color</text><text x="100" y="135" fill="#e8eaf0">k = (0, 1, 0)</text><text x="250" y="135" fill="#e8eaf0">v = blue</text>
    <text x="20" y="185" fill="#50c878">time</text><text x="100" y="185" fill="#e8eaf0">k = (0, 0, 1)</text><text x="250" y="185" fill="#e8eaf0">v = 3pm</text>
  </g>
  <path d="M 370 130 L 432 130" stroke="#8892a4" stroke-width="2" fill="none" />
  <path d="M 440 130 l -12 -6 l 0 12 z" fill="#8892a4" />
  <text x="560" y="44" fill="#e8eaf0" font-size="20" font-family="Inter, sans-serif" text-anchor="middle">S</text>
  <rect x="450" y="60" width="220" height="40" fill="#f5a623" fill-opacity="0.2" stroke="#f5a623" stroke-width="2" />
  <rect x="450" y="110" width="220" height="40" fill="#4a9eff" fill-opacity="0.2" stroke="#4a9eff" stroke-width="3" />
  <rect x="450" y="160" width="220" height="40" fill="#50c878" fill-opacity="0.2" stroke="#50c878" stroke-width="2" />
  <g font-family="Inter, sans-serif" font-size="18" text-anchor="middle">
    <text x="560" y="86" fill="#f5a623">row 1: Paris</text><text x="560" y="136" fill="#4a9eff">row 2: blue</text><text x="560" y="186" fill="#50c878">row 3: 3pm</text>
  </g>
  <text x="840" y="30" fill="#8892a4" font-size="19" font-family="Inter, sans-serif" text-anchor="middle">Read: qᵀ S</text>
  <text x="840" y="72" fill="#a06bd4" font-size="19" font-family="Inter, sans-serif" text-anchor="middle">q = (0, 1, 0)</text>
  <path d="M 670 130 L 802 130" stroke="#a06bd4" stroke-width="2" fill="none" />
  <path d="M 810 130 l -12 -6 l 0 12 z" fill="#a06bd4" />
  <rect x="815" y="110" width="150" height="40" fill="none" stroke="#4a9eff" stroke-width="3" />
  <text x="890" y="136" fill="#4a9eff" font-size="19" font-family="Inter, sans-serif" text-anchor="middle">blue</text>
  <line x1="20" y1="228" x2="980" y2="228" stroke="#2a3450" stroke-width="2" />
  <text x="20" y="262" fill="#e8eaf0" font-size="20" font-family="Inter, sans-serif">With one-hot keys, writing fills the row a key points at, and reading picks a row.</text>
  <text x="20" y="296" fill="#f5a623" font-size="20" font-family="Inter, sans-serif">Real keys overlap, so rows mix and reads blend. A d × d state holds about d clean pairs.</text>
</svg>
</div>

This $\mathbf S_t = \sum_{i \le t} \phi(\mathbf k_i) \mathbf v_i^\top$ is the linear-attention state. Every variant since 2020 changes one thing: the rule for writing to it. <!-- .element: class="text-lg" -->

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

<!-- .slide: id="attention-write-rules" -->

## Five Rules for Writing to the State

Plain linear attention only adds. A full memory needs a way to **forget** and a way to **overwrite**.

<div class="bench-table dense">
<table>
<thead><tr><th>Model</th><th>Update rule</th><th>What changed</th></tr></thead>
<tbody>
<tr><td><strong>Linear attention</strong> (2020)</td><td>$\mathbf S_t = \mathbf S_{t-1} + \mathbf k_t \mathbf v_t^\top$</td><td>Adds forever and never forgets</td></tr>
<tr><td><strong>RetNet</strong> (2023)</td><td>$\mathbf S_t = \gamma \mathbf S_{t-1} + \mathbf k_t \mathbf v_t^\top$</td><td>Old memories fade at a fixed rate $\gamma$</td></tr>
<tr><td><strong>Mamba-2</strong>, <strong>GLA</strong> (2024)</td><td>$\mathbf S_t = \alpha_t \mathbf S_{t-1} + \mathbf k_t \mathbf v_t^\top$</td><td>The fade rate $\alpha_t$ is computed from the current token</td></tr>
<tr><td><strong>DeltaNet</strong> (2021, 2024)</td><td>$\mathbf S_t = \mathbf S_{t-1} + \beta_t \mathbf k_t (\mathbf v_t - \mathbf S_{t-1}^\top \mathbf k_t)^\top$</td><td>Writes only the error, so a new value replaces the old one</td></tr>
<tr><td><strong>Gated DeltaNet</strong> (2025)</td><td>$\alpha_t$ gate plus the delta write</td><td>Forgets and overwrites. Used in Qwen3-Next and Kimi Linear</td></tr>
</tbody>
</table>
</div>

In this table, $\mathbf k_t$ stands for $\phi(\mathbf k_t)$. Every rule still updates one fixed-size $\mathbf S$ per token, so the cost grows linearly with sequence length. <!-- .element: class="text-lg" style="margin-top: 10px;" -->

---

<!-- .slide: id="attention-delta-overwrite" -->

## The Delta Rule Overwrites a Row

<div class="svg-figure">
<svg viewBox="0 0 1000 350" role="img" aria-label="Earlier the sentence I live in Paris wrote Paris into the city row. A new sentence, Actually, I moved to Rome, arrives; its token Rome has key city and value Rome. An additive write adds Rome to the row, which then holds Paris plus Rome, and a read returns a blend. A delta write first reads the row and gets Paris, computes the error Rome minus Paris, and adds that error, so the row holds Rome.">
  <text x="20" y="30" fill="#8892a4" font-size="18" font-family="Inter, sans-serif">Earlier in the chat:</text><text x="200" y="30" fill="#e8eaf0" font-size="18" font-family="Inter, sans-serif">"I live in Paris." The token "Paris" wrote key city, value Paris.</text><text x="20" y="62" fill="#8892a4" font-size="18" font-family="Inter, sans-serif">New sentence:</text><text x="200" y="62" fill="#e8eaf0" font-size="18" font-family="Inter, sans-serif">"Actually, I moved to Rome." The token "Rome" writes key city, value Rome.</text>
  <text x="20" y="138" fill="#4a9eff" font-size="20" font-family="Inter, sans-serif">Additive</text>
  <rect x="170" y="112" width="200" height="40" fill="none" stroke="#8892a4" stroke-width="2" />
  <text x="270" y="138" fill="#e8eaf0" font-size="19" font-family="Inter, sans-serif" text-anchor="middle">row: Paris</text>
  <path d="M 370 132 L 452 132" stroke="#8892a4" stroke-width="2" fill="none" />
  <path d="M 460 132 l -12 -6 l 0 12 z" fill="#8892a4" />
  <text x="415" y="122" fill="#8892a4" font-size="16" font-family="Inter, sans-serif" text-anchor="middle">+ Rome</text>
  <rect x="460" y="112" width="250" height="40" fill="none" stroke="#e06c75" stroke-width="2" />
  <text x="585" y="138" fill="#e06c75" font-size="19" font-family="Inter, sans-serif" text-anchor="middle">row: Paris + Rome</text>
  <text x="730" y="138" fill="#e06c75" font-size="18" font-family="Inter, sans-serif">reads return a blend</text>
  <line x1="20" y1="182" x2="980" y2="182" stroke="#2a3450" stroke-width="2" />
  <text x="20" y="250" fill="#50c878" font-size="20" font-family="Inter, sans-serif">Delta</text>
  <g fill="none" stroke-width="2">
    <rect x="110" y="210" width="190" height="64" stroke="#8892a4" /><rect x="340" y="210" width="190" height="64" stroke="#4a9eff" /><rect x="570" y="210" width="190" height="64" stroke="#f5a623" /><rect x="800" y="210" width="180" height="64" stroke="#50c878" stroke-width="3" />
  </g>
  <g font-family="Inter, sans-serif" text-anchor="middle">
    <text x="205" y="238" fill="#e8eaf0" font-size="18">row: Paris</text><text x="205" y="262" fill="#8892a4" font-size="15">before</text>
    <text x="435" y="238" fill="#4a9eff" font-size="18">1. read: Paris</text><text x="435" y="262" fill="#8892a4" font-size="15">read with key city</text>
    <text x="665" y="238" fill="#f5a623" font-size="18">2. Rome − Paris</text><text x="665" y="262" fill="#8892a4" font-size="15">the error</text>
    <text x="890" y="238" fill="#50c878" font-size="18">3. row: Rome</text><text x="890" y="262" fill="#8892a4" font-size="15">Paris + error</text>
  </g>
  <g stroke="#8892a4" stroke-width="2" fill="none"><path d="M 300 242 L 332 242" /><path d="M 530 242 L 562 242" /><path d="M 760 242 L 792 242" /></g>
  <g fill="#8892a4"><path d="M 340 242 l -10 -5 l 0 10 z" /><path d="M 570 242 l -10 -5 l 0 10 z" /><path d="M 800 242 l -10 -5 l 0 10 z" /></g>
  <text x="20" y="326" fill="#8892a4" font-size="18" font-family="Inter, sans-serif">Writing the error instead of the value is the whole change. Adding the error to Paris lands exactly on Rome.</text>
</svg>
</div>

- DeltaNet's rule: $\mathbf S_t = \mathbf S_{t-1} + \beta_t \mathbf k_t (\mathbf v_t - \mathbf S_{t-1}^\top \mathbf k_t)^\top$. The term in parentheses is the error
- $\beta_t$ sets how much of the error to write: $\beta_t = 1$ replaces the value, $\beta_t = 0.5$ moves it halfway
- If the key is new, the read returns about zero, so the error is just $\mathbf v_t$ and the write is an ordinary add
- The model learns during training to give both sentences' tokens a similar key, so the second write lands in the same row

---

<!-- .slide: id="attention-delta" -->

## The Delta Rule Is a Gradient Step

Treat $\mathbf S$ as a tiny linear model. Each token is one training example: input $\mathbf k_t$, target $\mathbf v_t$.

1. Prediction, what the memory returns now: $\hat{\mathbf v} = \mathbf S^\top \mathbf k_t$
2. Loss: $\ell = \frac{1}{2} \lVert \hat{\mathbf v} - \mathbf v_t \rVert^2$
3. Gradient: $\nabla_{\mathbf S} \ell = \mathbf k_t (\hat{\mathbf v} - \mathbf v_t)^\top$
4. One step with learning rate $\beta_t$: $\mathbf S \leftarrow \mathbf S + \beta_t \mathbf k_t (\mathbf v_t - \hat{\mathbf v})^\top$, which is the delta rule

<div class="bench-table">
<table>
<thead><tr><th>Loss on one key-value pair</th><th>One gradient step gives</th></tr></thead>
<tbody>
<tr><td>Dot product $-\langle \mathbf S^\top \mathbf k_t, \mathbf v_t \rangle$: push the read toward $\mathbf v_t$ without limit</td><td>$\mathbf S + \beta_t \mathbf k_t \mathbf v_t^\top$, plain linear attention</td></tr>
<tr><td>Squared error $\frac{1}{2} \lVert \mathbf S^\top \mathbf k_t - \mathbf v_t \rVert^2$: make the read equal $\mathbf v_t$</td><td>$\mathbf S + \beta_t \mathbf k_t (\mathbf v_t - \mathbf S^\top \mathbf k_t)^\top$, DeltaNet</td></tr>
</tbody>
</table>
</div>

The state is a model trained while it reads, one step per token. The section on continual learning builds on this. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

:::figure img="images/albert_gu.jpg" name="Albert Gu" kicker="S4 (2021), Mamba (2023)" alt="Albert Gu"
Built the state-space line, the most serious architectural challenger to the transformer since 2017. S4 derived a sequence model from a continuous-time dynamical system, an idea from control theory, not NLP. Mamba, with Tri Dao, made state transitions depend on the input and became competitive at scale.

Co-author Tri Dao wrote FlashAttention, the work that made exact attention fast on real hardware. The person who made attention efficient is also building its replacement. This is an engineering argument, not a fight between camps.
:::

---

<!-- .slide: id="attention-s4-ode" -->

## S4 Starts From a Differential Equation

S4 stands for **Structured State Space Sequence** model (Gu, Goel, and Ré, 2021).

$$h'(t) = A h(t) + B x(t) \qquad y(t) = C h(t)$$

<div class="svg-figure">
<svg viewBox="0 0 1000 156" role="img" aria-label="Block diagram. The input x of t is scaled by B and added to A times the state, giving the rate of change h prime of t, which integrates into the state h of t. The state is read out by C to give y of t, and the same state signal feeds back through A. The junction adds; the sign comes from A, whose entries are chosen so the state decays toward zero when there is no input.">
  <text x="60" y="71" fill="#f5a623" font-size="20" font-family="Inter, sans-serif" text-anchor="middle">x(t)</text>
  <path d="M 90 65 L 142 65" stroke="#8892a4" stroke-width="2" fill="none" /><path d="M 150 65 l -10 -5 l 0 10 z" fill="#8892a4" />
  <rect x="150" y="40" width="80" height="50" fill="none" stroke="#f5a623" stroke-width="2" />
  <text x="190" y="72" fill="#f5a623" font-size="20" font-family="Inter, sans-serif" text-anchor="middle">B</text>
  <path d="M 230 65 L 295 65" stroke="#8892a4" stroke-width="2" fill="none" /><path d="M 303 65 l -10 -5 l 0 10 z" fill="#8892a4" />
  <circle cx="325" cy="65" r="22" fill="none" stroke="#e8eaf0" stroke-width="2" />
  <text x="325" y="73" fill="#e8eaf0" font-size="24" font-family="Inter, sans-serif" text-anchor="middle">+</text>
  <path d="M 347 65 L 412 65" stroke="#8892a4" stroke-width="2" fill="none" /><path d="M 420 65 l -10 -5 l 0 10 z" fill="#8892a4" /><text x="383" y="55" fill="#8892a4" font-size="15" font-family="Inter, sans-serif" text-anchor="middle">h′(t)</text>
  <rect x="420" y="35" width="180" height="60" fill="none" stroke="#4a9eff" stroke-width="3" />
  <text x="510" y="62" fill="#4a9eff" font-size="20" font-family="Inter, sans-serif" text-anchor="middle">state h(t)</text>
  <text x="510" y="84" fill="#8892a4" font-size="15" font-family="Inter, sans-serif" text-anchor="middle">integrates h′(t)</text>
  <path d="M 600 65 L 682 65" stroke="#8892a4" stroke-width="2" fill="none" /><path d="M 690 65 l -10 -5 l 0 10 z" fill="#8892a4" />
  <rect x="690" y="40" width="80" height="50" fill="none" stroke="#50c878" stroke-width="2" />
  <text x="730" y="72" fill="#50c878" font-size="20" font-family="Inter, sans-serif" text-anchor="middle">C</text>
  <path d="M 770 65 L 832 65" stroke="#8892a4" stroke-width="2" fill="none" /><path d="M 840 65 l -10 -5 l 0 10 z" fill="#8892a4" />
  <text x="875" y="71" fill="#50c878" font-size="20" font-family="Inter, sans-serif" text-anchor="middle">y(t)</text>
  <circle cx="640" cy="65" r="5" fill="#8892a4" /><path d="M 640 65 L 640 128 L 528 128" stroke="#a06bd4" stroke-width="2" fill="none" /><path d="M 520 128 l 10 -5 l 0 10 z" fill="#a06bd4" /><rect x="440" y="106" width="80" height="44" fill="none" stroke="#a06bd4" stroke-width="2" /><text x="480" y="135" fill="#a06bd4" font-size="20" font-family="Inter, sans-serif" text-anchor="middle">A</text><path d="M 440 128 L 325 128 L 325 97" stroke="#a06bd4" stroke-width="2" fill="none" /><path d="M 325 88 l -5 10 l 10 0 z" fill="#a06bd4" />
  
</svg>
</div>

:::columns cols="2" gap="34px"
**The pieces**

- $x$: the input, one number per channel
- $h$: the state, a running summary of the input so far
- $A$ feeds the state back into its own rate of change. $B$ writes the input in. $C$ reads the output
+++
**What each word of the name means**

- **State space**: the model is this equation
- **Structured**: $A$ has a special form (**HiPPO**) so $h$ summarizes the whole history and is fast to compute
- **Sequence**: built for long inputs like audio and text
:::

---

<!-- .slide: id="attention-s4-discrete" -->

## Sampling the Equation Gives an RNN

Tokens are discrete, so S4 steps the equation forward by a fixed time $\Delta$ per token.

<div class="svg-figure">
<svg viewBox="0 0 1000 336" style="max-height: 200px;" role="img" aria-label="A continuous input signal is sampled at intervals of width delta, giving one value per token. Each sample updates the state: the old state is multiplied by A bar, the input is multiplied by B bar and added, and C reads out the prediction. This is a linear RNN.">
  <text x="20" y="30" fill="#8892a4" font-size="20" font-family="Inter, sans-serif">1. Sample the continuous signal every &#916; time units: one sample per token</text>
  <path d="M 40 90.0 L 50 86.0 L 60 82.1 L 70 78.2 L 80 74.5 L 90 70.9 L 100 67.6 L 110 64.5 L 120 61.7 L 130 59.1 L 140 57.0 L 150 55.2 L 160 53.8 L 170 52.8 L 180 52.2 L 190 52.0 L 200 52.2 L 210 52.9 L 220 54.0 L 230 55.4 L 240 57.3 L 250 59.5 L 260 62.1 L 270 64.9 L 280 68.1 L 290 71.4 L 300 75.0 L 310 78.8 L 320 82.7 L 330 86.6 L 340 90.6 L 350 94.6 L 360 98.5 L 370 102.4 L 380 106.1 L 390 109.6 L 400 112.9 L 410 116.0 L 420 118.8 L 430 121.2 L 440 123.3 L 450 125.1 L 460 126.4 L 470 127.3 L 480 127.9 L 490 128.0 L 500 127.7 L 510 127.0 L 520 125.8 L 530 124.3 L 540 122.4 L 550 120.1 L 560 117.5 L 570 114.6 L 580 111.4 L 590 108.0 L 600 104.4 L 610 100.6 L 620 96.7 L 630 92.8 L 640 88.8 L 650 84.8 L 660 80.9 L 670 77.0 L 680 73.3 L 690 69.8 L 700 66.6 L 710 63.6 L 720 60.8 L 730 58.4 L 740 56.4 L 750 54.7 L 760 53.4 L 770 52.5 L 780 52.1 L 790 52.0 L 800 52.4 L 810 53.2 L 820 54.4 L 830 56.0 L 840 57.9 L 850 60.3 L 860 62.9 L 870 65.9 L 880 69.1 L 890 72.5 L 900 76.2 L 910 80.0 L 920 83.9 L 930 87.9 L 940 91.9 L 950 95.8 L 960 99.7" stroke="#8892a4" stroke-width="2" fill="none" />
  <g fill="#f5a623">
    <circle cx="120" cy="61.7" r="7" /> <circle cx="280" cy="68.1" r="7" /> <circle cx="440" cy="123.3" r="7" /> <circle cx="600" cy="104.4" r="7" /> <circle cx="760" cy="53.4" r="7" /> <circle cx="920" cy="83.9" r="7" />
  </g>
  <path d="M 120 140 L 280 140" stroke="#f5a623" stroke-width="2" fill="none" />
  <path d="M 120 134 L 120 146 M 280 134 L 280 146" stroke="#f5a623" stroke-width="2" />
  <text x="200" y="164" fill="#f5a623" font-size="19" font-family="Inter, sans-serif" text-anchor="middle">&#916;</text>
  <line x1="20" y1="186" x2="980" y2="186" stroke="#2a3450" stroke-width="2" />
  <text x="20" y="220" fill="#8892a4" font-size="20" font-family="Inter, sans-serif">2. Each sample updates the state</text>
  <rect x="40" y="240" width="110" height="48" fill="none" stroke="#4a9eff" stroke-width="2" />
  <text x="95" y="271" fill="#e8eaf0" font-size="22" font-family="Inter, sans-serif" text-anchor="middle">h&#8348;&#8331;&#8321;</text>
  <path d="M 150 264 L 300 264" stroke="#4a9eff" stroke-width="2" fill="none" />
  <path d="M 300 264 l -10 -5 l 0 10 z" fill="#4a9eff" />
  <text x="225" y="254" fill="#4a9eff" font-size="19" font-family="Inter, sans-serif" text-anchor="middle">&#215; &#256;</text>
  <circle cx="330" cy="264" r="22" fill="none" stroke="#e8eaf0" stroke-width="2" />
  <text x="330" y="272" fill="#e8eaf0" font-size="24" font-family="Inter, sans-serif" text-anchor="middle">+</text>
  <path d="M 330 306 L 330 292" stroke="#f5a623" stroke-width="2" fill="none" />
  <path d="M 330 287 l -5 10 l 10 0 z" fill="#f5a623" />
  <text x="330" y="326" fill="#f5a623" font-size="19" font-family="Inter, sans-serif" text-anchor="middle">x&#8348; &#215; B&#772;</text>
  <path d="M 352 264 L 470 264" stroke="#4a9eff" stroke-width="2" fill="none" />
  <path d="M 470 264 l -10 -5 l 0 10 z" fill="#4a9eff" />
  <rect x="480" y="240" width="110" height="48" fill="none" stroke="#4a9eff" stroke-width="2" />
  <text x="535" y="271" fill="#e8eaf0" font-size="22" font-family="Inter, sans-serif" text-anchor="middle">h&#8348;</text>
  <path d="M 590 264 L 700 264" stroke="#50c878" stroke-width="2" fill="none" />
  <path d="M 700 264 l -10 -5 l 0 10 z" fill="#50c878" />
  <text x="645" y="254" fill="#50c878" font-size="19" font-family="Inter, sans-serif" text-anchor="middle">&#215; C</text>
  <text x="715" y="272" fill="#50c878" font-size="22" font-family="Inter, sans-serif">y&#8348;</text>
  <text x="780" y="262" fill="#8892a4" font-size="18" font-family="Inter, sans-serif">linear: no tanh,</text>
  <text x="780" y="286" fill="#8892a4" font-size="18" font-family="Inter, sans-serif">no gate</text>
</svg>
</div>

:::columns cols="2" gap="34px"
**One step forward**

- $h$ changes by about $\Delta \cdot h'$ per step: $h_t \approx h_{t-1} + \Delta (A h_{t-1} + B x_t)$
- Group terms: $h_t = \bar A h_{t-1} + \bar B x_t$, with $\bar A \approx I + \Delta A$ and $\bar B \approx \Delta B$
- S4 uses the exact form $\bar A = \exp(\Delta A)$
+++
**A one-number example**

- $A = -1$ and $\Delta = 0.1$ give $h_t = 0.9 h_{t-1} + 0.1 x_t$
- Each token keeps 90% of the state and mixes in the input, the same shape as RetNet's update
- A bigger $\Delta$ forgets faster. Mamba turns that into a per-token knob
:::

---

<!-- .slide: id="attention-s4-conv" -->

## S4 Trains as One Long Convolution

Unroll the recurrence. The output is a weighted sum of past inputs, and the list of weights is called the **kernel** $K$.

$$y_t = K_0 x_t + K_1 x_{t-1} + K_2 x_{t-2} + \cdots \qquad K_j = C \bar A^j \bar B$$

<div class="svg-figure">
<svg viewBox="0 0 1000 220" style="max-height: 160px;" role="img" aria-label="A row of eight inputs. Beneath it, a kernel of eight weights that shrink with distance into the past, K zero through K seven. The output at the last position is the sum of each input times its kernel weight. The same kernel slides to every position.">
  <g fill="none" stroke="#f5a623" stroke-width="2">
    <rect x="160" y="20" width="90" height="40" /><rect x="260" y="20" width="90" height="40" /><rect x="360" y="20" width="90" height="40" /><rect x="460" y="20" width="90" height="40" /><rect x="560" y="20" width="90" height="40" /><rect x="660" y="20" width="90" height="40" /><rect x="760" y="20" width="90" height="40" /><rect x="860" y="20" width="90" height="40" />
  </g>
  <g fill="#e8eaf0" font-size="19" font-family="Inter, sans-serif" text-anchor="middle">
    <text x="205" y="47">x&#8321;</text><text x="305" y="47">x&#8322;</text><text x="405" y="47">x&#8323;</text><text x="505" y="47">x&#8324;</text><text x="605" y="47">x&#8325;</text><text x="705" y="47">x&#8326;</text><text x="805" y="47">x&#8327;</text><text x="905" y="47">x&#8328;</text>
  </g>
  <text x="20" y="47" fill="#f5a623" font-size="19" font-family="Inter, sans-serif">inputs</text>
  <text x="20" y="150" fill="#4a9eff" font-size="19" font-family="Inter, sans-serif">kernel K</text>
  <g fill="#4a9eff" fill-opacity="0.45" stroke="#4a9eff" stroke-width="1">
    <rect x="185" y="152" width="40" height="8" /><rect x="285" y="148" width="40" height="12" /><rect x="385" y="142" width="40" height="18" /><rect x="485" y="134" width="40" height="26" /><rect x="585" y="124" width="40" height="36" /><rect x="685" y="110" width="40" height="50" /><rect x="785" y="94" width="40" height="66" /><rect x="885" y="76" width="40" height="84" />
  </g>
  <g fill="#8892a4" font-size="16" font-family="Inter, sans-serif" text-anchor="middle">
    <text x="205" y="182">K&#8327;</text><text x="305" y="182">K&#8326;</text><text x="405" y="182">K&#8325;</text><text x="505" y="182">K&#8324;</text><text x="605" y="182">K&#8323;</text><text x="705" y="182">K&#8322;</text><text x="805" y="182">K&#8321;</text><text x="905" y="182">K&#8320;</text>
  </g>
  <text x="560" y="212" fill="#50c878" font-size="19" font-family="Inter, sans-serif" text-anchor="middle">y&#8328; = sum of each input times its weight. Older inputs get smaller weights.</text>
</svg>
</div>

:::columns cols="2" gap="34px"
**What $K_j$ means**

- How much an input from $j$ tokens ago still affects the output now
- $\bar B$ writes it in, $\bar A$ shrinks it $j$ times, $C$ reads it out
- One-number example: $K_j = 0.1 \times 0.9^j$, so $0.1, 0.09, 0.081, \dots$
+++
**Why it matters**

- $K$ is the **same at every position**, so one kernel slides along the sequence
- Training: apply $K$ to the whole sequence at once with an FFT
- Inference: run the recurrence token by token
:::

---

<!-- .slide: id="attention-mamba-delta" -->

## Mamba Makes the Step Size Depend on the Token

S4's convolution trick needs $\bar A$, $\bar B$, $C$ fixed for every token. Mamba gives that up: $\Delta_t$, $B_t$, and $C_t$ are computed from $x_t$.

<div class="svg-figure">
<svg viewBox="0 0 1000 260" role="img" aria-label="With a large step size, A bar is near zero: the old state is mostly erased and the current token is written in strongly. With a small step size, A bar is near one: the old state is kept and the current token barely enters.">
  <text x="240" y="30" fill="#50c878" font-size="22" font-family="Inter, sans-serif" text-anchor="middle">Large &#916;&#8348;: "this token matters"</text>
  <rect x="40" y="70" width="120" height="50" fill="none" stroke="#8892a4" stroke-width="2" stroke-dasharray="6 5" />
  <text x="100" y="101" fill="#8892a4" font-size="20" font-family="Inter, sans-serif" text-anchor="middle">old state</text>
  <path d="M 160 95 L 300 95" stroke="#8892a4" stroke-width="2" stroke-dasharray="6 5" fill="none" />
  <text x="230" y="85" fill="#8892a4" font-size="18" font-family="Inter, sans-serif" text-anchor="middle">&#256; &#8776; 0</text>
  <rect x="310" y="70" width="120" height="50" fill="none" stroke="#50c878" stroke-width="3" />
  <text x="370" y="101" fill="#50c878" font-size="20" font-family="Inter, sans-serif" text-anchor="middle">new state</text>
  <path d="M 370 190 L 370 130" stroke="#50c878" stroke-width="5" fill="none" />
  <path d="M 370 124 l -8 14 l 16 0 z" fill="#50c878" />
  <text x="370" y="214" fill="#50c878" font-size="20" font-family="Inter, sans-serif" text-anchor="middle">x&#8348; written in strongly</text>
  <text x="240" y="250" fill="#8892a4" font-size="18" font-family="Inter, sans-serif" text-anchor="middle">reset and focus: a name, a new topic</text>
  <line x1="500" y1="20" x2="500" y2="250" stroke="#2a3450" stroke-width="2" />
  <text x="760" y="30" fill="#4a9eff" font-size="22" font-family="Inter, sans-serif" text-anchor="middle">Small &#916;&#8348;: "skip this token"</text>
  <rect x="560" y="70" width="120" height="50" fill="none" stroke="#4a9eff" stroke-width="2" />
  <text x="620" y="101" fill="#e8eaf0" font-size="20" font-family="Inter, sans-serif" text-anchor="middle">old state</text>
  <path d="M 680 95 L 820 95" stroke="#4a9eff" stroke-width="5" fill="none" />
  <path d="M 820 95 l -12 -7 l 0 14 z" fill="#4a9eff" />
  <text x="750" y="85" fill="#4a9eff" font-size="18" font-family="Inter, sans-serif" text-anchor="middle">&#256; &#8776; 1</text>
  <rect x="830" y="70" width="120" height="50" fill="none" stroke="#4a9eff" stroke-width="2" />
  <text x="890" y="101" fill="#e8eaf0" font-size="20" font-family="Inter, sans-serif" text-anchor="middle">new state</text>
  <path d="M 890 190 L 890 130" stroke="#8892a4" stroke-width="2" stroke-dasharray="6 5" fill="none" />
  <text x="890" y="214" fill="#8892a4" font-size="20" font-family="Inter, sans-serif" text-anchor="middle">x&#8348; barely enters</text>
  <text x="760" y="250" fill="#8892a4" font-size="18" font-family="Inter, sans-serif" text-anchor="middle">hold: filler, punctuation, padding</text>
</svg>
</div>

- $\bar A_t = \exp(\Delta_t A)$ with $A$ negative, so one number $\Delta_t$ controls both forgetting and writing
- The kernel now changes at every position, so the FFT convolution is gone. Mamba trains with a **parallel scan** kept in fast GPU memory instead

---

<!-- .slide: id="attention-ssm" -->

## Fixed and Selective Recurrence Side by Side

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

Selectivity is the LSTM's gating idea, rebuilt so training still runs in parallel. RWKV reached a similar design independently. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="attention-ssd" -->

## Mamba-2 Turns Out to Be Linear Attention

Dao and Gu (2024) restricted $\bar A_t$ to a single number $a_t$ per step. The selective SSM then equals masked linear attention.

<div class="svg-figure">
<svg viewBox="0 0 1000 330" role="img" aria-label="Two six by six lower-triangular masks. Linear attention weights every past token equally. Mamba-2 multiplies each past token by a decay that shrinks with distance, so the cells fade away from the diagonal.">
  <text x="220" y="34" fill="#4a9eff" font-size="21" font-family="Inter, sans-serif" text-anchor="middle">Linear attention: causal mask</text>
  <g>
    <rect x="118" y="56" width="34" height="34" fill="#4a9eff" fill-opacity="0.8" stroke="#2a3450" stroke-width="1" />
    <rect x="152" y="56" width="34" height="34" fill="none" stroke="#2a3450" stroke-width="1" />
    <rect x="186" y="56" width="34" height="34" fill="none" stroke="#2a3450" stroke-width="1" />
    <rect x="220" y="56" width="34" height="34" fill="none" stroke="#2a3450" stroke-width="1" />
    <rect x="254" y="56" width="34" height="34" fill="none" stroke="#2a3450" stroke-width="1" />
    <rect x="288" y="56" width="34" height="34" fill="none" stroke="#2a3450" stroke-width="1" />
    <rect x="118" y="90" width="34" height="34" fill="#4a9eff" fill-opacity="0.8" stroke="#2a3450" stroke-width="1" />
    <rect x="152" y="90" width="34" height="34" fill="#4a9eff" fill-opacity="0.8" stroke="#2a3450" stroke-width="1" />
    <rect x="186" y="90" width="34" height="34" fill="none" stroke="#2a3450" stroke-width="1" />
    <rect x="220" y="90" width="34" height="34" fill="none" stroke="#2a3450" stroke-width="1" />
    <rect x="254" y="90" width="34" height="34" fill="none" stroke="#2a3450" stroke-width="1" />
    <rect x="288" y="90" width="34" height="34" fill="none" stroke="#2a3450" stroke-width="1" />
    <rect x="118" y="124" width="34" height="34" fill="#4a9eff" fill-opacity="0.8" stroke="#2a3450" stroke-width="1" />
    <rect x="152" y="124" width="34" height="34" fill="#4a9eff" fill-opacity="0.8" stroke="#2a3450" stroke-width="1" />
    <rect x="186" y="124" width="34" height="34" fill="#4a9eff" fill-opacity="0.8" stroke="#2a3450" stroke-width="1" />
    <rect x="220" y="124" width="34" height="34" fill="none" stroke="#2a3450" stroke-width="1" />
    <rect x="254" y="124" width="34" height="34" fill="none" stroke="#2a3450" stroke-width="1" />
    <rect x="288" y="124" width="34" height="34" fill="none" stroke="#2a3450" stroke-width="1" />
    <rect x="118" y="158" width="34" height="34" fill="#4a9eff" fill-opacity="0.8" stroke="#2a3450" stroke-width="1" />
    <rect x="152" y="158" width="34" height="34" fill="#4a9eff" fill-opacity="0.8" stroke="#2a3450" stroke-width="1" />
    <rect x="186" y="158" width="34" height="34" fill="#4a9eff" fill-opacity="0.8" stroke="#2a3450" stroke-width="1" />
    <rect x="220" y="158" width="34" height="34" fill="#4a9eff" fill-opacity="0.8" stroke="#2a3450" stroke-width="1" />
    <rect x="254" y="158" width="34" height="34" fill="none" stroke="#2a3450" stroke-width="1" />
    <rect x="288" y="158" width="34" height="34" fill="none" stroke="#2a3450" stroke-width="1" />
    <rect x="118" y="192" width="34" height="34" fill="#4a9eff" fill-opacity="0.8" stroke="#2a3450" stroke-width="1" />
    <rect x="152" y="192" width="34" height="34" fill="#4a9eff" fill-opacity="0.8" stroke="#2a3450" stroke-width="1" />
    <rect x="186" y="192" width="34" height="34" fill="#4a9eff" fill-opacity="0.8" stroke="#2a3450" stroke-width="1" />
    <rect x="220" y="192" width="34" height="34" fill="#4a9eff" fill-opacity="0.8" stroke="#2a3450" stroke-width="1" />
    <rect x="254" y="192" width="34" height="34" fill="#4a9eff" fill-opacity="0.8" stroke="#2a3450" stroke-width="1" />
    <rect x="288" y="192" width="34" height="34" fill="none" stroke="#2a3450" stroke-width="1" />
    <rect x="118" y="226" width="34" height="34" fill="#4a9eff" fill-opacity="0.8" stroke="#2a3450" stroke-width="1" />
    <rect x="152" y="226" width="34" height="34" fill="#4a9eff" fill-opacity="0.8" stroke="#2a3450" stroke-width="1" />
    <rect x="186" y="226" width="34" height="34" fill="#4a9eff" fill-opacity="0.8" stroke="#2a3450" stroke-width="1" />
    <rect x="220" y="226" width="34" height="34" fill="#4a9eff" fill-opacity="0.8" stroke="#2a3450" stroke-width="1" />
    <rect x="254" y="226" width="34" height="34" fill="#4a9eff" fill-opacity="0.8" stroke="#2a3450" stroke-width="1" />
    <rect x="288" y="226" width="34" height="34" fill="#4a9eff" fill-opacity="0.8" stroke="#2a3450" stroke-width="1" />
  </g>
  <text x="220" y="290" fill="#8892a4" font-size="18" font-family="Inter, sans-serif" text-anchor="middle">every earlier token weighted 1</text>
  <line x1="500" y1="20" x2="500" y2="300" stroke="#2a3450" stroke-width="2" />
  <text x="780" y="34" fill="#50c878" font-size="21" font-family="Inter, sans-serif" text-anchor="middle">Mamba-2: decay mask</text>
  <g>
    <rect x="678" y="56" width="34" height="34" fill="#50c878" fill-opacity="0.8" stroke="#2a3450" stroke-width="1" />
    <rect x="712" y="56" width="34" height="34" fill="none" stroke="#2a3450" stroke-width="1" />
    <rect x="746" y="56" width="34" height="34" fill="none" stroke="#2a3450" stroke-width="1" />
    <rect x="780" y="56" width="34" height="34" fill="none" stroke="#2a3450" stroke-width="1" />
    <rect x="814" y="56" width="34" height="34" fill="none" stroke="#2a3450" stroke-width="1" />
    <rect x="848" y="56" width="34" height="34" fill="none" stroke="#2a3450" stroke-width="1" />
    <rect x="678" y="90" width="34" height="34" fill="#50c878" fill-opacity="0.44" stroke="#2a3450" stroke-width="1" />
    <rect x="712" y="90" width="34" height="34" fill="#50c878" fill-opacity="0.8" stroke="#2a3450" stroke-width="1" />
    <rect x="746" y="90" width="34" height="34" fill="none" stroke="#2a3450" stroke-width="1" />
    <rect x="780" y="90" width="34" height="34" fill="none" stroke="#2a3450" stroke-width="1" />
    <rect x="814" y="90" width="34" height="34" fill="none" stroke="#2a3450" stroke-width="1" />
    <rect x="848" y="90" width="34" height="34" fill="none" stroke="#2a3450" stroke-width="1" />
    <rect x="678" y="124" width="34" height="34" fill="#50c878" fill-opacity="0.242" stroke="#2a3450" stroke-width="1" />
    <rect x="712" y="124" width="34" height="34" fill="#50c878" fill-opacity="0.44" stroke="#2a3450" stroke-width="1" />
    <rect x="746" y="124" width="34" height="34" fill="#50c878" fill-opacity="0.8" stroke="#2a3450" stroke-width="1" />
    <rect x="780" y="124" width="34" height="34" fill="none" stroke="#2a3450" stroke-width="1" />
    <rect x="814" y="124" width="34" height="34" fill="none" stroke="#2a3450" stroke-width="1" />
    <rect x="848" y="124" width="34" height="34" fill="none" stroke="#2a3450" stroke-width="1" />
    <rect x="678" y="158" width="34" height="34" fill="#50c878" fill-opacity="0.133" stroke="#2a3450" stroke-width="1" />
    <rect x="712" y="158" width="34" height="34" fill="#50c878" fill-opacity="0.242" stroke="#2a3450" stroke-width="1" />
    <rect x="746" y="158" width="34" height="34" fill="#50c878" fill-opacity="0.44" stroke="#2a3450" stroke-width="1" />
    <rect x="780" y="158" width="34" height="34" fill="#50c878" fill-opacity="0.8" stroke="#2a3450" stroke-width="1" />
    <rect x="814" y="158" width="34" height="34" fill="none" stroke="#2a3450" stroke-width="1" />
    <rect x="848" y="158" width="34" height="34" fill="none" stroke="#2a3450" stroke-width="1" />
    <rect x="678" y="192" width="34" height="34" fill="#50c878" fill-opacity="0.073" stroke="#2a3450" stroke-width="1" />
    <rect x="712" y="192" width="34" height="34" fill="#50c878" fill-opacity="0.133" stroke="#2a3450" stroke-width="1" />
    <rect x="746" y="192" width="34" height="34" fill="#50c878" fill-opacity="0.242" stroke="#2a3450" stroke-width="1" />
    <rect x="780" y="192" width="34" height="34" fill="#50c878" fill-opacity="0.44" stroke="#2a3450" stroke-width="1" />
    <rect x="814" y="192" width="34" height="34" fill="#50c878" fill-opacity="0.8" stroke="#2a3450" stroke-width="1" />
    <rect x="848" y="192" width="34" height="34" fill="none" stroke="#2a3450" stroke-width="1" />
    <rect x="678" y="226" width="34" height="34" fill="#50c878" fill-opacity="0.06" stroke="#2a3450" stroke-width="1" />
    <rect x="712" y="226" width="34" height="34" fill="#50c878" fill-opacity="0.073" stroke="#2a3450" stroke-width="1" />
    <rect x="746" y="226" width="34" height="34" fill="#50c878" fill-opacity="0.133" stroke="#2a3450" stroke-width="1" />
    <rect x="780" y="226" width="34" height="34" fill="#50c878" fill-opacity="0.242" stroke="#2a3450" stroke-width="1" />
    <rect x="814" y="226" width="34" height="34" fill="#50c878" fill-opacity="0.44" stroke="#2a3450" stroke-width="1" />
    <rect x="848" y="226" width="34" height="34" fill="#50c878" fill-opacity="0.8" stroke="#2a3450" stroke-width="1" />
  </g>
  <text x="780" y="290" fill="#8892a4" font-size="18" font-family="Inter, sans-serif" text-anchor="middle">entry (i, j) = a&#7522; a&#7522;&#8331;&#8321; &#8943; a&#11388;&#8330;&#8321;, fading with distance</text>
  <text x="500" y="324" fill="#8892a4" font-size="17" font-family="Inter, sans-serif" text-anchor="middle">rows are the current position, columns are earlier positions</text>
</svg>
</div>

$C_t$ plays the query, $B_t$ the key, and $x_t$ the value. Linear attention and state-space models turned out to be one model family. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="attention-verdict" -->

## Where Recurrent Models Fall Behind

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

Attention only where random access is needed. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="attention-hybrids-recent" -->

## Recent Models Settle Near Three to One

<div class="bench-table">
<table>
<thead><tr><th>Model</th><th>Recurrent layers</th><th>Full attention</th></tr></thead>
<tbody>
<tr><td><strong>Jamba</strong> (2024)</td><td>Mamba</td><td>1 layer in 8</td></tr>
<tr><td><strong>Qwen3-Next</strong> (2025)</td><td>Gated DeltaNet</td><td>1 layer in 4 (12 of 48)</td></tr>
<tr><td><strong>Kimi Linear</strong> (2025)</td><td>Kimi Delta Attention, a per-channel gate</td><td>7 of 27 layers</td></tr>
<tr><td><strong>Qwen3.5</strong> (2026)</td><td>Gated DeltaNet</td><td>15 of 60 layers</td></tr>
<tr><td><strong>MiniMax-M2</strong> (2025)</td><td>None</td><td>All 62 layers, after its predecessor used a hybrid</td></tr>
</tbody>
</table>
</div>

The delta-rule family is what ships. MiniMax is the counterexample: its team went back to full attention, so the question is still open. <!-- .element: class="text-lg" style="margin-top: 14px;" -->

---

<!-- .slide: id="sidequest-bitter" -->

## <span class="side-quest-badge">Side Quest</span> The Bitter Lesson for Architectures

In "The Bitter Lesson" (2019), Rich Sutton argued that across seventy years of AI research, general methods that scale with computation have beaten methods built on human knowledge. Try applying that to this section.

- People designed Mamba's selectivity by hand. Is that the kind of built-in structure Sutton says eventually loses, or a cheaper way to use compute, which his argument favors?
- Someone decided Qwen3-Next should use one attention layer in every four. Will hand-picked ratios like that last, or will models learn them?
- Attention replaced recurrent models once. Could attention itself turn out to be hand-built structure that something more general replaces?

Nobody knows yet. So far, how well a method scales with compute has predicted the winning architecture better than how clever the method is. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="attention-transition" -->

## Three Shared Assumptions

Transformers and state-space models alike still do all three:

<div class="card-grid cols-3">
<div class="card"><h4>Generates left to right</h4><p>One token at a time, never revising.</p></div>
<div class="card"><h4>Spends fixed compute per token</h4><p>The same layers, every time, regardless of difficulty.</p></div>
<div class="card"><h4>Freezes its weights</h4><p>Training ends, and learning ends with it.</p></div>
</div>

