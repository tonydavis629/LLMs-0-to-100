:::divider id="divider-block" title="Putting It All Together" sub="Following one prompt through a decoder-only transformer"
:::

---

<!-- .slide: id="embedding-layer" -->

## Step 1: Words to Embeddings

<div class="decoder-flow active-tokenization active-embedding">
  <div class="flow-node tokenization">word &rarr; tokens</div>
  <div class="flow-node embedding">token IDs &rarr; vectors</div>
  <div class="flow-node position">add position</div>
  <div class="flow-node attention">multi-head attention</div>
  <div class="flow-node ffn">feed-forward network</div>
  <div class="flow-node repeat">repeat blocks</div>
  <div class="flow-node softmax">softmax</div>
</div>

<div class="embedding-visual">
  <div class="word-card">The capital of France</div>
  <div class="pipe-arrow">&rarr;</div>
  <div class="token-strip"><span>The</span><span>&nbsp;capital</span><span>&nbsp;of</span><span>&nbsp;France</span></div>
  <div class="pipe-arrow">&rarr;</div>
  <div class="id-strip"><span>464</span><span>3139</span><span>286</span><span>4881</span></div>
  <div class="pipe-arrow">&rarr;</div>
  <div class="embedding-table">
    <div>row 464</div><div class="vector-bar"></div>
    <div>row 3139</div><div class="vector-bar wide"></div>
    <div>row 286</div><div class="vector-bar short"></div>
    <div>row 4881</div><div class="vector-bar"></div>
  </div>
</div>

Each token ID selects one learned row from the embedding matrix. From here on, the transformer sees vectors, not text.

---

<!-- .slide: id="positional-encoding" -->

## Step 2: Add Position

<div class="decoder-flow active-position">
  <div class="flow-node tokenization">word &rarr; tokens</div>
  <div class="flow-node embedding">token IDs &rarr; vectors</div>
  <div class="flow-node position">add position</div>
  <div class="flow-node attention">multi-head attention</div>
  <div class="flow-node ffn">feed-forward network</div>
  <div class="flow-node repeat">repeat blocks</div>
  <div class="flow-node softmax">softmax</div>
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

Without position information, attention is order-blind: the same tokens in a different order would look too similar.

---

<!-- .slide: id="attention-sublayer" -->

## Step 3: Multi-Head Attention

<div class="decoder-flow active-attention">
  <div class="flow-node tokenization">word &rarr; tokens</div>
  <div class="flow-node embedding">token IDs &rarr; vectors</div>
  <div class="flow-node position">add position</div>
  <div class="flow-node attention">multi-head attention</div>
  <div class="flow-node ffn">feed-forward network</div>
  <div class="flow-node repeat">repeat blocks</div>
  <div class="flow-node softmax">softmax</div>
</div>

<div class="attention-visual">
  <div class="causal-matrix">
    <div></div><div>The</div><div>capital</div><div>of</div><div>France</div>
    <div>The</div><span class="on"></span><span></span><span></span><span></span>
    <div>capital</div><span class="on"></span><span class="on"></span><span></span><span></span>
    <div>of</div><span class="on"></span><span class="on"></span><span class="on"></span><span></span>
    <div>France</div><span class="on"></span><span class="on"></span><span class="on"></span><span class="on"></span>
  </div>
  <div class="head-stack">
    <div class="head-card">head 1<br><span>syntax</span></div>
    <div class="head-card">head 2<br><span>entities</span></div>
    <div class="head-card">head 3<br><span>position</span></div>
  </div>
</div>

Attention is the communication step. Each token reads from earlier tokens through the causal mask, and each head can learn a different kind of lookup.

---

<!-- .slide: id="ffn-sublayer" -->

## Step 4: Feed-Forward Network

<div class="decoder-flow active-ffn">
  <div class="flow-node tokenization">word &rarr; tokens</div>
  <div class="flow-node embedding">token IDs &rarr; vectors</div>
  <div class="flow-node position">add position</div>
  <div class="flow-node attention">multi-head attention</div>
  <div class="flow-node ffn">feed-forward network</div>
  <div class="flow-node repeat">repeat blocks</div>
  <div class="flow-node softmax">softmax</div>
</div>

<div class="ffn-visual">
  <div class="vector-column">
    <div class="vector-pill">The vector</div>
    <div class="vector-pill">capital vector</div>
    <div class="vector-pill">of vector</div>
    <div class="vector-pill">France vector</div>
  </div>
  <div class="ffn-layers">
    <div class="ffn-layer">linear up</div>
    <div class="ffn-layer accent">GELU</div>
    <div class="ffn-layer">linear down</div>
  </div>
  <div class="vector-column">
    <div class="vector-pill accent">updated The</div>
    <div class="vector-pill accent">updated capital</div>
    <div class="vector-pill accent">updated of</div>
    <div class="vector-pill accent">updated France</div>
  </div>
</div>

The FFN does not mix positions. It transforms each token vector independently, adding learned features and much of the model's stored knowledge.

---

<!-- .slide: id="stacking-and-head" -->

## Step 5: Repeat the Block

<div class="decoder-flow active-repeat">
  <div class="flow-node tokenization">word &rarr; tokens</div>
  <div class="flow-node embedding">token IDs &rarr; vectors</div>
  <div class="flow-node position">add position</div>
  <div class="flow-node attention">multi-head attention</div>
  <div class="flow-node ffn">feed-forward network</div>
  <div class="flow-node repeat">repeat blocks</div>
  <div class="flow-node softmax">softmax</div>
</div>

<div class="stack-visual">
  <div class="stack-block">block 1<br><span>attention + FFN</span></div>
  <div class="stack-block">block 2<br><span>attention + FFN</span></div>
  <div class="stack-block">block 3<br><span>attention + FFN</span></div>
  <div class="stack-ellipsis">...</div>
  <div class="stack-block accent">block N<br><span>attention + FFN</span></div>
</div>

Depth lets later blocks build on earlier features. GPT-2 small repeats this decoder block 12 times.

---

<!-- .slide: id="softmax-output" -->

## Step 6: Logits to Probabilities

<div class="decoder-flow active-softmax">
  <div class="flow-node tokenization">word &rarr; tokens</div>
  <div class="flow-node embedding">token IDs &rarr; vectors</div>
  <div class="flow-node position">add position</div>
  <div class="flow-node attention">multi-head attention</div>
  <div class="flow-node ffn">feed-forward network</div>
  <div class="flow-node repeat">repeat blocks</div>
  <div class="flow-node softmax">softmax</div>
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
  <div class="matrix-card accent">sample or choose<br><strong>next token</strong></div>
</div>

The language-modeling head scores every vocabulary item. Softmax turns those scores into the next-token distribution.

---

<!-- .slide: id="norm-and-residual" -->

## Normalization: Keep the Scale Stable

<div class="norm-visual">
  <div class="norm-lane">
    <h3>Original transformer</h3>
    <div class="norm-row"><span>stream</span><b>&rarr;</b><span>sub-layer</span><b>&rarr;</b><span class="accent">LayerNorm</span></div>
    <p>Post-norm is simple, but deep stacks are harder to train.</p>
  </div>
  <div class="norm-lane">
    <h3>Modern decoder blocks</h3>
    <div class="norm-row"><span>stream</span><b>&rarr;</b><span class="accent">LayerNorm / RMSNorm</span><b>&rarr;</b><span>sub-layer</span></div>
    <p>Pre-norm keeps the residual path cleaner at depth.</p>
  </div>
</div>

LayerNorm rescales each token vector. RMSNorm keeps the stabilizing scale step and drops mean-centering for speed and simplicity.

---

<!-- .slide: id="residual-stream" -->

## The Residual Stream View

<div class="residual-visual">
  <div class="residual-line"></div>
  <div class="residual-token start">embedding</div>
  <div class="residual-write top">attention writes a delta</div>
  <div class="residual-write bottom">FFN writes a delta</div>
  <div class="residual-write top second">next block writes again</div>
  <div class="residual-token end">final vector</div>
</div>

The stream is not replaced. Each sub-layer reads the current vector, computes an update, and adds that update back.

<div class="formula-card">output stream = input stream + sub-layer update</div>

This is why information from the original prompt can remain available many layers later.

---

<!-- .slide: id="side-quest-residual" -->

## Side Quest: In-Context Learning in the Stream

<div class="icl-diagram">
  <div class="icl-prompt">
    <div>English: cat &rarr; French: chat</div>
    <div>English: dog &rarr; French: chien</div>
    <div class="accent">English: bird &rarr; French:</div>
  </div>
  <div class="pipe-arrow">&rarr;</div>
  <div class="icl-pattern">
    <div>examples write task pattern</div>
    <div class="stream-line"></div>
    <div class="accent">current query reads that pattern</div>
  </div>
  <div class="pipe-arrow">&rarr;</div>
  <div class="matrix-card accent">oiseau</div>
</div>

In-context learning is a runtime effect: the prompt examples shape the residual stream, and later tokens use that temporary pattern without changing model weights.
