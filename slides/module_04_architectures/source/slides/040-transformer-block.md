:::divider id="divider-block" title="Putting It All Together" sub="Following one prompt through a decoder-only transformer"
:::

---

<!-- .slide: id="decoder-only-arch" -->

## The Decoder-Only Architecture

<div class="decoder-svg">
<svg viewBox="0 0 560 463" role="img" aria-label="Decoder-only transformer stack from raw text up to next-token probabilities, with residual connections around each sub-layer"><rect x="98" y="141" width="304" height="181" rx="10" fill="rgba(74,158,255,0.03)" stroke="rgba(74,158,255,0.5)" stroke-width="1.2" stroke-dasharray="5 4"></rect><defs><marker id="da" markerWidth="7" markerHeight="7" refX="5" refY="2.5" orient="auto"><path d="M0,0 L5,2.5 L0,5 Z" fill="#f5a623"></path></marker><marker id="da2" markerWidth="7" markerHeight="7" refX="5" refY="2.5" orient="auto"><path d="M0,0 L5,2.5 L0,5 Z" fill="#4a9eff"></path></marker></defs><rect x="120" y="12" width="260" height="28" rx="8" fill="rgba(245,166,35,0.12)" stroke="rgba(245,166,35,0.55)" stroke-width="1.3"></rect><text x="250" y="30" text-anchor="middle" font-size="14" fill="#e8eaf0">Next-token probabilities</text><line x1="250" y1="55" x2="250" y2="41" stroke="#f5a623" stroke-width="2" marker-end="url(#da)"></line><rect x="120" y="55" width="260" height="28" rx="8" fill="rgba(74,158,255,0.10)" stroke="rgba(74,158,255,0.40)" stroke-width="1.3"></rect><text x="250" y="73" text-anchor="middle" font-size="14" fill="#e8eaf0">Softmax</text><line x1="250" y1="98" x2="250" y2="84" stroke="#f5a623" stroke-width="2" marker-end="url(#da)"></line><rect x="120" y="98" width="260" height="28" rx="8" fill="rgba(74,158,255,0.10)" stroke="rgba(74,158,255,0.40)" stroke-width="1.3"></rect><text x="250" y="110" text-anchor="middle" font-size="14" fill="#e8eaf0">Unembedding</text><text x="250" y="124" text-anchor="middle" font-size="10" fill="#8892a4">linear: model dim &#8594; vocabulary</text><line x1="250" y1="141" x2="250" y2="127" stroke="#f5a623" stroke-width="2" marker-end="url(#da)"></line><rect x="120" y="153" width="260" height="28" rx="8" fill="rgba(136,146,164,0.10)" stroke="rgba(136,146,164,0.45)" stroke-width="1.3"></rect><text x="250" y="171" text-anchor="middle" font-size="12.5" fill="#e8eaf0">Add &amp; Norm</text><line x1="250" y1="196" x2="250" y2="182" stroke="#f5a623" stroke-width="2" marker-end="url(#da)"></line><rect x="120" y="196" width="260" height="28" rx="8" fill="rgba(136,146,164,0.10)" stroke="rgba(136,146,164,0.45)" stroke-width="1.3"></rect><text x="250" y="214" text-anchor="middle" font-size="12.5" fill="#e8eaf0">Feed-Forward Network</text><line x1="250" y1="239" x2="250" y2="225" stroke="#f5a623" stroke-width="2" marker-end="url(#da)"></line><rect x="120" y="239" width="260" height="28" rx="8" fill="rgba(136,146,164,0.10)" stroke="rgba(136,146,164,0.45)" stroke-width="1.3"></rect><text x="250" y="257" text-anchor="middle" font-size="12.5" fill="#e8eaf0">Add &amp; Norm</text><line x1="250" y1="282" x2="250" y2="268" stroke="#f5a623" stroke-width="2" marker-end="url(#da)"></line><rect x="120" y="282" width="260" height="28" rx="8" fill="rgba(136,146,164,0.10)" stroke="rgba(136,146,164,0.45)" stroke-width="1.3"></rect><text x="250" y="300" text-anchor="middle" font-size="12.5" fill="#e8eaf0">Masked Multi-Head Self-Attention</text><text x="406" y="231.5" text-anchor="start" font-size="15" fill="#f5a623">N&#215;</text><path d="M250 317 L104 317 L104 253 L119 253" fill="none" stroke="#4a9eff" stroke-width="1.6" marker-end="url(#da2)"></path><path d="M250 231 L104 231 L104 167 L119 167" fill="none" stroke="#4a9eff" stroke-width="1.6" marker-end="url(#da2)"></path><text x="100" y="285.5" text-anchor="end" font-size="10" fill="#4a9eff">residual</text><line x1="250" y1="337" x2="250" y2="323" stroke="#f5a623" stroke-width="2" marker-end="url(#da)"></line><rect x="120" y="337" width="260" height="28" rx="8" fill="rgba(74,158,255,0.10)" stroke="rgba(74,158,255,0.40)" stroke-width="1.3"></rect><text x="250" y="355" text-anchor="middle" font-size="14" fill="#e8eaf0">Positional encoding</text><line x1="250" y1="380" x2="250" y2="366" stroke="#f5a623" stroke-width="2" marker-end="url(#da)"></line><rect x="120" y="380" width="260" height="28" rx="8" fill="rgba(74,158,255,0.10)" stroke="rgba(74,158,255,0.40)" stroke-width="1.3"></rect><text x="250" y="392" text-anchor="middle" font-size="14" fill="#e8eaf0">Tokenize</text><text x="250" y="406" text-anchor="middle" font-size="10" fill="#8892a4">text &#8594; token vectors</text><line x1="250" y1="423" x2="250" y2="409" stroke="#f5a623" stroke-width="2" marker-end="url(#da)"></line><rect x="120" y="423" width="260" height="28" rx="8" fill="rgba(245,166,35,0.12)" stroke="rgba(245,166,35,0.55)" stroke-width="1.3"></rect><text x="250" y="441" text-anchor="middle" font-size="14" fill="#e8eaf0">The capital of France</text></svg>
</div>

Bottom to top: tokenize and embed, add position, $N$ identical blocks refine the vectors (each sub-layer wrapped in a blue **residual connection**), then the **unembedding** scores every token.

---

<!-- .slide: id="embedding-layer" -->

## Step 1: Words to Vectors

<div class="decoder-flow active-embedding">
  <div class="flow-node embedding">word &rarr; vector</div>
  <div class="flow-node position">add position</div>
  <div class="flow-node attention">multi-head attention</div>
  <div class="flow-node ffn">feed-forward network</div>
  <div class="flow-node repeat">repeat blocks</div>
  <div class="flow-node sampling">sampling</div>
</div>

<div class="embedding-visual">
  <div class="word-card">The capital of France</div>
  <div class="pipe-arrow">&rarr;</div>
  <div class="token-strip"><span>The</span><span>&nbsp;capital</span><span>&nbsp;of</span><span>&nbsp;France</span></div>
  <div class="pipe-arrow">&rarr;</div>
  <div class="id-strip"><span>464</span><span>3139</span><span>286</span><span>4881</span></div>
  <div class="pipe-arrow">&rarr;</div>
  <div class="embedding-table">
    <div class="et-title">embedding matrix<br><span>one learned vector per row</span></div>
    <div class="et-row"><span class="et-id">row 464</span><span class="et-vec">[ 0.14 -0.22 0.05 0.61 &#8230; ]</span></div>
    <div class="et-row"><span class="et-id">row 3139</span><span class="et-vec">[ -0.31 0.47 0.18 -0.09 &#8230; ]</span></div>
    <div class="et-row"><span class="et-id">row 286</span><span class="et-vec">[ 0.02 0.33 -0.27 0.40 &#8230; ]</span></div>
    <div class="et-row accent"><span class="et-id">row 4881</span><span class="et-vec">[ 0.55 -0.12 0.29 -0.63 &#8230; ]</span></div>
  </div>
</div>

Tokenize, then look up each ID in the embedding matrix. From here on, the transformer sees vectors, not text.

:::note
Row $k$ of the **embedding matrix** is the learned vector for token $k$. Lookup is pure indexing: ID 4881 ("France") selects row 4881, and that row **is** the token's vector.
:::

---

:::manim id="embedding-anim" scene="embedding-lookup"
:::

---

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

**Induction heads** (Olsson et al., 2022) implement copy-and-continue:

- See `[A] [B] ... [A]`, predict `[B]`
- The head finds an earlier occurrence of the current token and predicts what followed it, with no weight updates
- A core mechanism behind few-shot (in-context) learning at scale
