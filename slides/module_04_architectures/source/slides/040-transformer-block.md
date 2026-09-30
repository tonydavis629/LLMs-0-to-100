<!-- .slide: id="positional-encoding" -->

## Step 2: Add Position

<div class="decoder-flow active-position">
  <div class="flow-node embedding">word &rarr; vector</div>
  <div class="flow-node position">add position</div>
  <div class="flow-node attention">multi-head attention</div>
  <div class="flow-node ffn">feed-forward network</div>
  <div class="flow-node repeat">repeat blocks</div>
  <div class="flow-node sampling">sampling</div>
</div>

<div class="position-visual">
  <div class="pos-column">
    <h3>Token vectors</h3>
    <div class="vector-pill">The</div>
    <div class="vector-pill">capital</div>
    <div class="vector-pill">of</div>
    <div class="vector-pill">France</div>
  </div>
  <div class="plus-column">+</div>
  <div class="pos-column">
    <h3>Position vectors</h3>
    <div class="vector-pill muted">pos 0</div>
    <div class="vector-pill muted">pos 1</div>
    <div class="vector-pill muted">pos 2</div>
    <div class="vector-pill muted">pos 3</div>
  </div>
  <div class="plus-column">=</div>
  <div class="pos-column result">
    <h3>Input to block 1</h3>
    <div class="vector-pill accent">The @ 0</div>
    <div class="vector-pill accent">capital @ 1</div>
    <div class="vector-pill accent">of @ 2</div>
    <div class="vector-pill accent">France @ 3</div>
  </div>
</div>

Without position information, attention is order-blind: the same tokens in a different order would look identical.

---

<!-- .slide: id="attention-sublayer" -->

## Step 3: Multi-Head Attention

<div class="decoder-flow active-attention">
  <div class="flow-node embedding">word &rarr; vector</div>
  <div class="flow-node position">add position</div>
  <div class="flow-node attention">multi-head attention</div>
  <div class="flow-node ffn">feed-forward network</div>
  <div class="flow-node repeat">repeat blocks</div>
  <div class="flow-node sampling">sampling</div>
</div>

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V$$

<div class="attention-detail">
  <div class="causal-matrix">
    <div></div><div>The</div><div>capital</div><div>of</div><div>France</div>
    <div>The</div><span class="on"></span><span></span><span></span><span></span>
    <div>capital</div><span class="on"></span><span class="on"></span><span></span><span></span>
    <div>of</div><span class="on"></span><span class="on"></span><span class="on"></span><span></span>
    <div>France</div><span class="on"></span><span class="on"></span><span class="on"></span><span class="on"></span>
  </div>
  <div class="attention-steps">
    <ul>
      <li>Each token is projected into a <strong>query</strong>, a <strong>key</strong>, and a <strong>value</strong></li>
      <li>Score every query against every key, then scale by the square root of the head dimension</li>
      <li>The <strong>causal mask</strong> keeps only the lower triangle, so a token never reads from the future</li>
      <li>Softmax the scores into weights, then output the <strong>weighted sum of values</strong></li>
      <li>Several <strong>heads</strong> run in parallel, each learning a different kind of lookup</li>
    </ul>
  </div>
</div>

Attention is the only sub-layer where information moves between positions.

---

<!-- .slide: id="ffn-sublayer" -->

## Step 4: Feed-Forward Network

<div class="decoder-flow active-ffn">
  <div class="flow-node embedding">word &rarr; vector</div>
  <div class="flow-node position">add position</div>
  <div class="flow-node attention">multi-head attention</div>
  <div class="flow-node ffn">feed-forward network</div>
  <div class="flow-node repeat">repeat blocks</div>
  <div class="flow-node sampling">sampling</div>
</div>

$$\text{FFN}(x) = W_2 \operatorname{ReLU}(W_1 x), \qquad d \rightarrow 4d \rightarrow d$$

- Applied to **each token independently**; never mixes positions
- Project up to $4d$, apply **ReLU**, project back down (GPT-2 uses GELU, a smooth variant)
- Holds much of the model's stored knowledge

---

:::manim id="ffn-anim" scene="ffn-expand"
:::

---

<!-- .slide: id="stacking-and-head" -->

## Step 5: Repeat the Block

<div class="decoder-flow active-repeat">
  <div class="flow-node embedding">word &rarr; vector</div>
  <div class="flow-node position">add position</div>
  <div class="flow-node attention">multi-head attention</div>
  <div class="flow-node ffn">feed-forward network</div>
  <div class="flow-node repeat">repeat blocks</div>
  <div class="flow-node sampling">sampling</div>
</div>

<div class="stack-visual">
  <div class="stack-block">block 1<br><span>attention + FFN</span></div>
  <div class="stack-block">block 2<br><span>attention + FFN</span></div>
  <div class="stack-block">block 3<br><span>attention + FFN</span></div>
  <div class="stack-ellipsis">...</div>
  <div class="stack-block accent">block N<br><span>attention + FFN</span></div>
</div>

Depth lets later blocks build on earlier features. GPT-2 small repeats this decoder block 12 times. Each block reuses the same shape but learns its own weights.

---

<!-- .slide: id="norm-and-residual" -->

## Normalization: Keep the Scale Stable

Normalization rescales each token vector to a stable range. This keeps a deep stack trainable.

<div class="norm-visual">
  <div class="norm-lane">
    <h3>LayerNorm</h3>
    <p>Subtract the mean, divide by the standard deviation, apply a learned gain and bias.</p>
  </div>
  <div class="norm-lane">
    <h3>RMSNorm</h3>
    <p>Divide by the root-mean-square only. Cheaper, works as well.</p>
  </div>
</div>

Placement: the original transformer normalized **after** each sub-layer (post-norm). Modern decoders normalize **before** (pre-norm), which trains more stably at depth.

---

:::manim id="norm-anim" scene="norm-demo"
:::

---

<!-- .slide: id="residual-stream" -->

## The Residual Stream View

Every sub-layer reads the current vector, computes an update, and **adds** it back:

<div class="formula-card">output = input + sub-layer(norm(input))</div>

- Blocks only add; the original embedding is never overwritten
- Prompt information stays available many layers later
- Gradients flow back through the addition without decaying
- This "residual stream" is the central object in mechanistic interpretability

---

:::manim id="residual-anim" scene="residual-stream"
:::

---

<!-- .slide: id="side-quest-induction-heads" -->

## Side Quest: Induction Heads

<div class="emb-svg">
<svg viewBox="0 40 1000 260" role="img" aria-label="Induction head example: at the second Mr, the head attends to Dursley, the token that followed the first Mr, and copies it as the prediction.">
<defs><marker id="iha" markerWidth="7" markerHeight="7" refX="5" refY="2.5" orient="auto"><path d="M0,0 L5,2.5 L0,5 Z" fill="#f5a623"></path></marker><marker id="ihb" markerWidth="7" markerHeight="7" refX="5" refY="2.5" orient="auto"><path d="M0,0 L5,2.5 L0,5 Z" fill="#3fb950"></path></marker><marker id="ihc" markerWidth="7" markerHeight="7" refX="5" refY="2.5" orient="auto"><path d="M0,0 L5,2.5 L0,5 Z" fill="#8892a4"></path></marker></defs>
<rect x="30" y="150" width="90" height="46" rx="8" fill="rgba(74,158,255,0.10)" stroke="#4a9eff" stroke-width="1.6"></rect>
<text x="75.0" y="179" text-anchor="middle" font-size="17" fill="#e8eaf0">Mr</text>
<rect x="134" y="150" width="90" height="46" rx="8" fill="rgba(63,185,80,0.16)" stroke="#3fb950" stroke-width="1.6"></rect>
<text x="179.0" y="179" text-anchor="middle" font-size="17" fill="#e8eaf0">Dursley</text>
<rect x="238" y="150" width="90" height="46" rx="8" fill="rgba(74,158,255,0.10)" stroke="#4a9eff" stroke-width="1.6"></rect>
<text x="283.0" y="179" text-anchor="middle" font-size="17" fill="#e8eaf0">was</text>
<rect x="342" y="150" width="90" height="46" rx="8" fill="rgba(74,158,255,0.10)" stroke="#4a9eff" stroke-width="1.6"></rect>
<text x="387.0" y="179" text-anchor="middle" font-size="17" fill="#e8eaf0">proud</text>
<rect x="446" y="150" width="90" height="46" rx="8" fill="rgba(74,158,255,0.10)" stroke="#4a9eff" stroke-width="1.6"></rect>
<text x="491.0" y="179" text-anchor="middle" font-size="17" fill="#e8eaf0">.</text>
<rect x="550" y="150" width="90" height="46" rx="8" fill="rgba(74,158,255,0.10)" stroke="#4a9eff" stroke-width="1.6"></rect>
<text x="595.0" y="179" text-anchor="middle" font-size="17" fill="#e8eaf0">Later</text>
<rect x="654" y="150" width="90" height="46" rx="8" fill="rgba(74,158,255,0.10)" stroke="#4a9eff" stroke-width="1.6"></rect>
<text x="699.0" y="179" text-anchor="middle" font-size="17" fill="#e8eaf0">,</text>
<rect x="758" y="150" width="90" height="46" rx="8" fill="rgba(245,166,35,0.18)" stroke="#f5a623" stroke-width="1.6"></rect>
<text x="803.0" y="179" text-anchor="middle" font-size="17" fill="#e8eaf0">Mr</text>
<rect x="862" y="150" width="90" height="46" rx="8" fill="rgba(136,146,164,0.08)" stroke="#8892a4" stroke-width="1.6" stroke-dasharray="5 4"></rect>
<text x="907.0" y="179" text-anchor="middle" font-size="17" fill="#8892a4">?</text>
<path d="M179.0,198 C179.0,240 75.0,240 75.0,202" fill="none" stroke="#8892a4" stroke-width="2" marker-end="url(#ihc)"></path>
<text x="40" y="262" text-anchor="start" font-size="14" fill="#8892a4">1. previous-token head:</text>
<text x="40" y="280" text-anchor="start" font-size="14" fill="#8892a4">Dursley stores "I came after Mr"</text>
<path d="M803.0,146 C803.0,55 179.0,55 179.0,142" fill="none" stroke="#f5a623" stroke-width="2.4" marker-end="url(#iha)"></path>
<text x="491.0" y="70" text-anchor="middle" font-size="14" fill="#f5a623">2. induction head at the second Mr looks for "the token that came after Mr" and finds Dursley</text>
<path d="M199.0,198 C239.0,300 907.0,300 907.0,202" fill="none" stroke="#3fb950" stroke-width="2.4" marker-end="url(#ihb)"></path>
<text x="623.0" y="290" text-anchor="middle" font-size="14" fill="#3fb950">3. copy: predict Dursley</text>
</svg>
</div>

- Two attention heads in different layers cooperate to copy `[A] [B] ... [A]` &rarr; `[B]` (Olsson et al., 2022)
- It works on name pairs never seen in training, so the copying happens entirely in context
- These heads form suddenly during training, at the same point in-context learning improves

---

<!-- .slide: id="sampling-step" -->

## Step 6: Sampling

<div class="decoder-flow active-sampling">
  <div class="flow-node embedding">word &rarr; vector</div>
  <div class="flow-node position">add position</div>
  <div class="flow-node attention">multi-head attention</div>
  <div class="flow-node ffn">feed-forward network</div>
  <div class="flow-node repeat">repeat blocks</div>
  <div class="flow-node sampling">sampling</div>
</div>

<div class="softmax-visual">
  <div class="matrix-card">last token vector<br><strong>France</strong></div>
  <div class="pipe-arrow">&rarr;</div>
  <div class="logit-bars">
    <div><span>Paris</span><b style="height: 120px;"></b></div>
    <div><span>London</span><b style="height: 40px;"></b></div>
    <div><span>city</span><b style="height: 64px;"></b></div>
    <div><span>capital</span><b style="height: 52px;"></b></div>
  </div>
  <div class="pipe-arrow">&rarr;</div>
  <div class="matrix-card accent">sample<br><strong>next token</strong></div>
</div>

- The **unembedding** (language-modeling head) maps the final vector to one score per vocabulary token; the mirror of the embedding lookup
- Softmax turns scores into the next-token distribution
- A decoding strategy (greedy, temperature, top-k, top-p) picks the token
- The token is appended and fed back in
