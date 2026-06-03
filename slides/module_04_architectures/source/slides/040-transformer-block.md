:::divider id="divider-block" title="Putting It All Together" sub="Following one prompt through a decoder-only transformer"
:::

---

<!-- .slide: id="decoder-only-arch" -->

## The Decoder-Only Architecture

<div class="decoder-arch">
  <div class="da-node io">Next-token probabilities</div>
  <div class="da-arrow">&uarr;</div>
  <div class="da-node">Softmax</div>
  <div class="da-arrow">&uarr;</div>
  <div class="da-node">Linear (LM head)</div>
  <div class="da-arrow">&uarr;</div>
  <div class="da-node">Final LayerNorm</div>
  <div class="da-arrow">&uarr;</div>
  <div class="da-block">
    <span class="da-block-label">N&times;</span>
    <div class="da-node sub">Add &amp; Norm</div>
    <div class="da-arrow">&uarr;</div>
    <div class="da-node sub">Feed-Forward Network</div>
    <div class="da-arrow">&uarr;</div>
    <div class="da-node sub">Add &amp; Norm</div>
    <div class="da-arrow">&uarr;</div>
    <div class="da-node sub">Masked Multi-Head Self-Attention</div>
  </div>
  <div class="da-arrow">&uarr;</div>
  <div class="da-node">Token + Positional Embedding</div>
  <div class="da-arrow">&uarr;</div>
  <div class="da-node io">Input tokens</div>
</div>

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
    <div class="et-row"><span class="et-id">row 464</span><span class="et-vec"><i></i><i></i><i></i><i></i><i></i><i></i></span></div>
    <div class="et-row"><span class="et-id">row 3139</span><span class="et-vec"><i></i><i></i><i></i><i></i><i></i><i></i></span></div>
    <div class="et-row"><span class="et-id">row 286</span><span class="et-vec"><i></i><i></i><i></i><i></i><i></i><i></i></span></div>
    <div class="et-row accent"><span class="et-id">row 4881</span><span class="et-vec"><i></i><i></i><i></i><i></i><i></i><i></i></span></div>
  </div>
</div>

Tokenization and the embedding lookup together turn each word into a learned vector. From here on, the transformer sees vectors, not text.

:::note
Each token ID is just an index into the **embedding matrix**: row $k$ holds the learned vector for token $k$. "Looking up" a token is pure indexing, with no computation, and these vectors are *learned* during training. Here the id 4881 ("France") selects its row, and that row **is** the token's vector.
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

Applied to **each token independently**, the feed-forward network projects the vector up into a much wider space (typically four times the model dimension), applies a **ReLU** nonlinearity, and projects it back down. Unlike attention, it never mixes positions. It adds learned features and holds much of the model's stored knowledge. (GPT-2 uses GELU, a smooth version of the same idea.)

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

Normalization rescales each token vector to a stable range so its values neither blow up nor vanish as it passes through many layers. This is what keeps a deep stack trainable.

<div class="norm-visual">
  <div class="norm-lane">
    <h3>LayerNorm</h3>
    <p>Recenters (subtract the mean) <strong>and</strong> rescales (divide by the standard deviation), then applies a learned gain and bias.</p>
  </div>
  <div class="norm-lane">
    <h3>RMSNorm</h3>
    <p>Drops the recentering and keeps only the rescaling (divide by the root-mean-square). Cheaper, and works just as well in practice.</p>
  </div>
</div>

Placement matters too: the original transformer normalized **after** each sub-layer (post-norm), while modern decoders normalize **before** each sub-layer (pre-norm), which trains more stably at depth.

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

The language-modeling head scores every vocabulary item; softmax turns those scores into the next-token distribution. The decoding strategy we saw earlier &mdash; greedy, temperature, top-k, or top-p &mdash; then chooses the next token, which is appended and fed back in.

---

<!-- .slide: id="residual-stream" -->

## The Residual Stream View

Every sub-layer is wrapped in a **residual connection**: it reads the current vector, computes an update, and adds that update back to the running vector.

<div class="formula-card">output = input + sub-layer(norm(input))</div>

Because each block only ever **adds** to this running vector, the original embedding is never overwritten. Information from the prompt stays available many layers later, and gradients flow back through the addition without decaying. This "residual stream" is the central object in mechanistic interpretability.

---

:::manim id="residual-anim" scene="residual-stream"
:::

---

<!-- .slide: id="side-quest-induction-heads" -->

## Side Quest: Induction Heads

**Induction heads** (Olsson et al., 2022) are attention heads that implement a simple copy-and-continue pattern:

- When the model sees a pattern like `[A] [B] ... [A]`, it predicts `[B]` as the next token
- This is a core mechanism behind in-context learning: the model copies a pattern it observed earlier in the same prompt

Induction heads are linked to the emergence of few-shot (in-context) learning at scale. Each head detects an earlier occurrence of the current token and predicts whatever followed it last time &mdash; no weight updates required.
