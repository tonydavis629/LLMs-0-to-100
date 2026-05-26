:::divider id="title" title="LLMs 0 to 100" sub="Module 3: Attention Mechanisms"
From Fixed Windows to Learned Lookups
:::

---

<!-- .slide: id="review-1" -->

## Review: Module 2

:::columns cols="2" gap="30px"
**The MLP**

A multi-layer perceptron applies repeated linear transformations with nonlinear activations between them:

$$\mathbf{h} = \sigma(W\mathbf{x} + \mathbf{b})$$

An MLP with enough hidden units can approximate any continuous function (universal approximation theorem).
+++
**Limitations for Sequences**

An MLP over a fixed window of tokens treats each position identically, but:

- The context window is fixed: token 1 cannot directly inform token 50
- Flattening token vectors destroys the idea that the same pattern can appear in many positions
- Parameter count grows with context length
- There is no direct mechanism for one token to select information from another token
:::

---

<!-- .slide: id="review-2" -->

## Review: Why the Architecture Matters

:::columns cols="2" gap="30px"
**From Module 2**

The single neuron could not solve XOR because a linear boundary cannot separate entangled classes. The fix was a hidden layer with a nonlinearity &mdash; the architecture had to match the structure of the problem.
+++
**Same Principle, Bigger Scale**

Language has structure that an MLP does not match:

- Every token can depend on any earlier token, not just its neighbors
- The same syntactic pattern ("the ___") appears at every position
- The relevant context length varies per token

**The architecture must match the structure of the data.**
:::

---

:::figure img="images/bahdanau_cho_bengio.jpg" name="Bahdanau, Cho &amp; Bengio" kicker="Made Attention a Central Mechanism"
- Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio (2014)
- "Neural Machine Translation by Jointly Learning to Align and Translate"
- Earlier sequence models compressed the entire source sentence into one fixed-size vector
- Their model learned to **align** each output word to the most relevant input words
- Attention was originally a fix for the bottleneck of fixed-size context vectors
- This paper made attention a first-class mechanism, not just a patch
:::

---

:::divider id="divider-why-attention" title="Why Attention?" sub="From compression to selective retrieval"
:::

---

<!-- .slide: id="fixed-context-bottleneck" -->

## The Fixed-Context Bottleneck

Earlier sequence models (RNNs, LSTMs) processed tokens one at a time and compressed the entire sequence into a **single hidden state vector**.

:::note
**Problem:** No matter how long the input, all information must fit through one fixed-size bottleneck. The model must decide what to keep before it knows what will be needed later.
:::

Attention solves this by giving each output token **direct access** to all input representations. No compression step. No bottleneck. Each token builds its own context by looking at whichever other tokens are relevant.

---

<!-- .slide: id="attention-as-lookup" -->

## Attention as Learned Lookup

Think of attention as a differentiable database query:

:::columns cols="3" gap="20px"
**Query**

What I am looking for

"Which tokens are relevant to me?"
+++
**Key**

What each token advertises

"I contain information about X"
+++
**Value**

What I retrieve

The actual content to aggregate
:::

The model learns its own queries, keys, and values. No manual feature engineering. The lookup pattern is discovered during training.

---

<!-- .slide: id="compare-normalize-retrieve" -->

## The Core Operation

Attention has three steps:

:::columns cols="3" gap="20px"
**1. Compare**

$$\text{scores} = QK^T$$

How well does each query match each key?
+++
**2. Normalize**

$$\text{weights} = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)$$

Convert scores to a probability distribution
+++
**3. Retrieve**

$$\text{output} = \text{weights} \cdot V$$

Take a weighted average of value vectors
:::

Compare, normalize, retrieve. That is all of attention.

---

:::figure img="images/vaswani_et_al.jpg" name="Vaswani et al." kicker="Attention Is All You Need"
- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan Gomez, Lukasz Kaiser, and Illia Polosukhin (2017)
- "Attention Is All You Need" &mdash; replaced recurrence entirely with attention
- Introduced scaled dot-product multi-head attention
- The Transformer architecture: no RNN, no convolution, just attention and feed-forward layers
- One of the most cited papers in machine learning
:::

---

:::divider id="divider-qkv" title="Queries, Keys, and Values" sub="Learned projections that define what each token looks for and offers"
:::

---

<!-- .slide: id="database-analogy" -->

## The Database Analogy

A traditional database maps a query to records using an index. Attention does the same thing, but with **soft** (differentiable) matching:

:::columns cols="2" gap="40px"
**Traditional Database**

- Exact match on a key field
- Returns one record
- Not differentiable
- Hand-designed index
+++
**Attention**

- Soft match between query and key vectors
- Returns a weighted combination of all values
- Fully differentiable
- Queries, keys, and values are all **learned**
:::

---

<!-- .slide: id="qkv-projections" -->

## Q, K, V Projections

Each token vector $\mathbf{x}_i$ is projected into three different spaces:

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

:::columns cols="2" gap="30px"
$X$: token matrix, shape $(n, d_{\text{model}})$

$W_Q, W_K, W_V$: learned projection matrices, shape $(d_{\text{model}}, d_k)$

$Q, K, V$: projected matrices, shape $(n, d_k)$
+++
**What this means:**

- The same token can "ask" different questions (via $Q$) than it "advertises" (via $K$) or "offers" (via $V$)
- The projection matrices are learned during training
- $d_k$ can be smaller than $d_{\text{model}}$ &mdash; a bottleneck that forces the model to focus
:::

---

<!-- .slide: id="qkv-shapes" -->

## Tensor Shapes for a Small Sequence

Consider 5 tokens with $d_{\text{model}} = 8$ and $d_k = 4$:

:::columns cols="2" gap="30px"
<div style="text-align: center;">

**Input**

$X$: $(5, 8)$

</div>

<div style="text-align: center;">

**Projections**

$W_Q, W_K, W_V$: $(8, 4)$

</div>
+++
<div style="text-align: center;">

**Projected**

$Q, K, V$: $(5, 4)$

</div>

<div style="text-align: center;">

**Scores**

$QK^T$: $(5, 5)$

</div>
:::

The $(5, 5)$ score matrix has one entry per pair of tokens. Row $i$ tells us how much token $i$'s query matches every other token's key.

---

<!-- .slide: id="pairwise-compatibility" -->

## Pairwise Token Compatibility

The score matrix $QK^T$ measures how well each query aligns with each key:

$$\text{score}(i, j) = \mathbf{q}_i \cdot \mathbf{k}_j$$

:::note
**Key insight:** This is a dot product &mdash; a measure of alignment. When query $i$ points in a similar direction to key $j$, the score is large. The model learns to point queries toward the keys that matter most.
:::

---

:::divider id="divider-scaled-dot-product" title="Scaled Dot-Product Attention" sub="The operation at the heart of every Transformer"
:::

---

<!-- .slide: id="scaling-problem" -->

## Why Scale by $\sqrt{d_k}$?

Dot products grow with dimension. For $d_k = 64$, the expected magnitude of $\mathbf{q} \cdot \mathbf{k}$ is about 4; for $d_k = 1024$, it is about 16.

When inputs to softmax are large, the output becomes **sharply peaked** &mdash; one entry near 1, the rest near 0. This produces tiny gradients that make training unstable.

:::note
Dividing by $\sqrt{d_k}$ keeps the variance of the dot product roughly constant regardless of dimension. This preserves gradient flow and stabilizes training.
:::

---

<!-- .slide: id="scaled-softmax" -->

## Scaled Softmax

The full attention weight computation:

$$\alpha_{ij} = \frac{\exp\!\left(\mathbf{q}_i \cdot \mathbf{k}_j / \sqrt{d_k}\right)}{\sum_{j'} \exp\!\left(\mathbf{q}_i \cdot \mathbf{k}_{j'} / \sqrt{d_k}\right)}$$

Each row of the attention matrix sums to 1. The weights form a **soft selection** over all tokens.

:::note
**Without scaling:** for $d_k = 64$, softmax saturates and gradients vanish. With scaling, the distribution stays smooth and learning proceeds. This is a small but critical engineering detail.
:::

---

<!-- .slide: id="attention-output" -->

## The Attention Output

The output is a weighted average of value vectors:

$$\text{output}_i = \sum_j \alpha_{ij} \, \mathbf{v}_j$$

:::columns cols="2" gap="30px"
**Interpretation:**

- Each output token is a mixture of all value vectors
- The mixture weights come from query-key compatibility
- If token $i$ attends strongly to token $j$, the output at position $i$ is dominated by $\mathbf{v}_j$
+++
**Properties:**

- Different tokens can attend to different subsets of the sequence
- The same token can be attended to by many others
- The output dimension matches the value dimension ($d_k$), not the sequence length
:::

---

<!-- .slide: id="full-attention-formula" -->

## The Complete Formula

Scaled dot-product attention in one line:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Compare, normalize, retrieve. Three matrix operations. No recurrence, no convolution, no fixed-size bottleneck. The model learns what to look at.

---

:::divider id="divider-masks" title="Attention Masks" sub="Controlling what each token can see"
:::

---

<!-- .slide: id="padding-masks" -->

## Padding Masks

In a batch, sequences have different lengths. Shorter sequences are padded with special tokens. A **padding mask** prevents attention to these fake tokens:

:::columns cols="2" gap="30px"
**Without mask:**

The model wastes capacity attending to padding tokens, which carry no information.
+++
**With mask:**

Set padded positions to $-\infty$ before softmax, so they receive zero weight after normalization.
:::

Padding masks are a practical necessity for batched training.

---

<!-- .slide: id="causal-masks" -->

## Causal Masks

A **causal mask** prevents each token from attending to **future** tokens. This is essential for autoregressive generation (like GPT).

:::columns cols="2" gap="30px"
**Bidirectional attention** (BERT-style)

Every token can attend to every other token. Useful for understanding the full context.

$$\begin{bmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{bmatrix}$$
+++
**Causal attention** (GPT-style)

Each token can only attend to itself and earlier tokens.

$$\begin{bmatrix} 1 & 0 & 0 \\ 1 & 1 & 0 \\ 1 & 1 & 1 \end{bmatrix}$$
:::

---

<!-- .slide: id="why-causal" -->

## Why Causal Masking?

In a language model, we predict the next token given all previous tokens:

$$P(w_t \mid w_1, w_2, \ldots, w_{t-1})$$

:::note
If token $t$ could see token $t+1$, the model would be **cheating** during training &mdash; it could copy the answer instead of learning to predict. Causal masking enforces the same information constraint during training that the model will face during inference, when future tokens do not exist yet.
:::

---

:::divider id="divider-multi-head" title="Multi-Head Attention" sub="Several lookup patterns running in parallel"
:::

---

<!-- .slide: id="why-multi-head" -->

## Why Multiple Heads?

A single attention head produces one weighted average per token. But a token might need to attend to different things for different reasons:

:::columns cols="2" gap="30px"
**Head 1** might learn syntactic attention &mdash; verbs attend to their subjects

**Head 2** might learn coreference &mdash; pronouns attend to their antecedents

**Head 3** might learn positional attention &mdash; each token attends to its immediate neighbor
+++
Multiple heads let the model learn **different ways to compare tokens** simultaneously, rather than forcing one pattern to serve all purposes.
:::

---

<!-- .slide: id="multi-head-mechanics" -->

## Multi-Head Mechanics

Each head $h$ has its own projections:

$$Q_h = XW_Q^h, \quad K_h = XW_K^h, \quad V_h = XW_V^h$$

Each head independently computes scaled dot-product attention:

$$\text{head}_h = \text{Attention}(Q_h, K_h, V_h)$$

The heads are **concatenated** and projected back:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) \, W_O$$

:::note
**Compute budget:** total projection size across all heads is typically kept equal to $d_{\text{model}}$. With 8 heads and $d_{\text{model}} = 512$, each head uses $d_k = 64$. Multi-head attention is not more expensive than single-head &mdash; it is the same compute, split into independent patterns.
:::

---

:::divider id="divider-cross-attention" title="Cross Attention" sub="When queries and keys come from different places"
:::

---

<!-- .slide: id="cross-vs-self" -->

## Cross Attention vs. Self-Attention

:::columns cols="2" gap="40px"
**Self-Attention**

$Q$, $K$, and $V$ all come from the **same** sequence.

Every token looks at every other token in the same sequence.

Used in both encoders and decoders.
+++
**Cross-Attention**

$Q$ comes from one sequence; $K$ and $V$ come from **another**.

One sequence selects information from the other.

Used in encoder-decoder models: the decoder attends to the encoder's representations.
:::

---

<!-- .slide: id="encoder-decoder" -->

## The Encoder-Decoder Pattern

The original Transformer used both self-attention and cross-attention:

:::columns cols="2" gap="30px"
**Encoder**

- Self-attention over the input sequence
- Every input token can see every other input token
- Produces a set of contextualized representations
+++
**Decoder**

- Self-attention over the output sequence (causally masked)
- Cross-attention: decoder queries attend to encoder keys and values
- Each output token pulls relevant information from the input
:::

Modern LLMs (GPT, LLaMA) use only the decoder with causal self-attention. The encoder-decoder pattern remains common in translation and speech models.

---

:::divider id="divider-positional" title="Positional Embeddings" sub="Teaching the model that order matters"
:::

---

<!-- .slide: id="permutation-equivariance" -->

## Self-Attention Is Permutation Equivariant

If you shuffle the input tokens, the attention output is the same shuffle. The operation has **no notion of order**.

:::note
**This is a problem.** "Dog bites man" and "man bites dog" have the same bag of tokens. Without position information, the model cannot distinguish them. Attention learns *what to look at* but not *where things are*.
:::

---

<!-- .slide: id="learned-vs-sinusoidal" -->

## Learned vs. Sinusoidal Positional Encodings

:::columns cols="2" gap="40px"
**Learned positional embeddings**

A trainable embedding table with one vector per position, added to the token embeddings.

$$X_{\text{pos}} = X + P$$

where $P$ is a learned matrix of shape (max_len, $d_{\text{model}}$).

Simple and effective. Used in BERT, GPT-2.
+++
**Sinusoidal positional encodings**

Fixed patterns based on sine and cosine at different frequencies:

$$PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

No parameters to learn. Extrapolates to unseen lengths. Used in the original Transformer.
:::

---

<!-- .slide: id="rope" -->

## Relative Position and RoPE

Both learned and sinusoidal encodings are **absolute** &mdash; they encode "position 5" as a fixed vector. But what often matters is **relative** position: "two tokens to the left."

**RoPE** (Rotary Position Embedding, Su et al. 2021) encodes relative position by rotating query and key vectors:

$$\mathbf{q}_m^T \mathbf{k}_n = f(m)^T f(n) = g(m - n)$$

The dot product depends only on the **relative offset** $m - n$, not the absolute positions. RoPE is used in LLaMA, Mistral, and most modern LLMs.

---

:::divider id="divider-compute" title="Memory and Compute" sub="The cost of every token seeing every other token"
:::

---

:::figure img="images/dao.jpg" name="Tri Dao" kicker="Made Exact Attention Fast"
- Tri Dao and collaborators (2022): FlashAttention
- Observed that the $n \times n$ attention matrix is the bottleneck
- FlashAttention computes exact attention without materializing the full $n \times n$ matrix in GPU HBM
- IO-aware: tiles the computation to keep intermediate results in fast SRAM
- Exact same result as standard attention, but 2-4x faster and 5-20x less memory
- FlashAttention-2 and -3 pushed further gains on newer GPU architectures
:::

---

<!-- .slide: id="n-squared-cost" -->

## The $O(n^2)$ Cost

Attention creates an $n \times n$ score matrix for $n$ tokens. Every token interacts with every other token.

:::columns cols="2" gap="30px"
**Compute:** $O(n^2 \cdot d_k)$ for the matrix multiplications

**Memory:** $O(n^2)$ to store the attention weights

For 128K tokens, the attention matrix has 16 billion entries.
+++
**Why this matters:**

- Doubling the context length quadruples the attention cost
- Long context is expensive because every token can see every other token
- The $O(n^2)$ cost is the main barrier to very long contexts
:::

---

<!-- .slide: id="mitigations" -->

## Modern Mitigations

Several approaches reduce the $O(n^2)$ cost without sacrificing too much quality:

:::columns cols="2" gap="30px"
**FlashAttention**

Computes exact attention without materializing the full matrix. Same result, less memory traffic.
+++
**Sliding-window attention**

Each token attends only to a local window of neighbors (plus a few global tokens). Reduces the effective $n$.
+++
**Multi-query attention (MQA)**

All heads share the same K and V projections. Only Q is per-head. Reduces KV cache size.
+++
**Grouped-query attention (GQA)**

A compromise: heads are divided into groups that share K and V. Used in LLaMA 2/3.
:::

---

:::divider id="divider-kv-cache" title="KV Cache" sub="Reusing work during generation"
:::

---

<!-- .slide: id="kv-cache-idea" -->

## The KV Cache

During autoregressive inference, the model generates one token at a time. Without caching, each step recomputes keys and values for **all** previous tokens.

:::columns cols="2" gap="30px"
**Without KV cache**

Each generation step recomputes K and V for every previous token. Cost per step grows linearly with sequence length. Total cost is $O(n^2)$.
+++
**With KV cache**

Previous keys and values are stored and reused. Each new token only computes its own K and V. Cost per step is constant. Total cost is $O(n)$.
:::

---

<!-- .slide: id="kv-cache-tradeoffs" -->

## KV Cache Tradeoffs

The KV cache is a classic memory-vs-compute tradeoff:

:::columns cols="2" gap="30px"
**Speed win**

Each generation step does $O(1)$ work instead of $O(n)$. Generation is dramatically faster, especially for long sequences.
+++
**Memory cost**

The cache stores $2 \times n_{\text{layers}} \times n_{\text{heads}} \times n_{\text{tokens}} \times d_k$ floats. For a 70B model with 128K context, the KV cache can be tens of gigabytes.
:::

:::note
**KV cache memory is a major limit** for long context and large-batch serving. This is why MQA and GQA exist &mdash; fewer unique K and V heads means a smaller cache.
:::

---

<!-- .slide: id="side-quest-registers" -->

## Side Quest: Register Tokens in Vision Transformers

Darcet, Oquab, Mairal, and Bojanowski (2023) observed **high-norm artifact tokens** in ViT feature maps, often corresponding to low-information background patches.

**Interpretation:** the model repurposes some patch tokens as internal scratch space for computation &mdash; they are not representing the image patch they correspond to, but storing global information.

**Register tokens** add extra learned token positions that the model can use as dedicated scratch space, so image tokens are freed to represent their actual patches.

:::note
This is a concrete example of token positions having roles beyond "word at position $i$." The model invents its own internal bookkeeping.
:::

---

<!-- .slide: id="side-quest-attention-maps" -->

## Side Quest: Attention Maps Are Not Always Explanations

Attention weights can be useful visual diagnostics &mdash; they show where the model is looking. But:

:::columns cols="2" gap="30px"
**Attention weights can suggest hypotheses**

- A pronoun attending to its antecedent is a real pattern
- Useful for debugging and building intuition
+++
**High attention does not prove causal importance**

- Attention is one of many operations in the network
- Ablation tests are needed for stronger claims
- The model may compensate through other pathways
:::

**Distinction:** visualization can suggest hypotheses, but ablation tests are needed for stronger claims.

---

:::divider id="divider-exercise" title="Exercise" sub="Attention Visualization and Positional Embedding Visualization"
:::

---

<!-- .slide: id="exercise-run" -->

## Running the Exercise

Open `module_03_attention/exercise.py` and fill in the `NotImplementedError` lines. Run after each step &mdash; unfinished functions are skipped automatically. <!-- .element: class="text-lg" -->

```bash
# Run the exercise (skips any not yet implemented)
cd exercises
uv run python module_03_attention/src/main.py
```

The exercise computes scaled dot-product attention step by step on a tiny 5-token sequence, then adds a causal mask and positional embeddings. Check the `output/` directory for plots after each run. <!-- .element: class="text-lg" style="margin-top: 15px;" -->

---

<!-- .slide: id="exercise-overview" -->

## Exercise: Attention Mechanisms

Implement scaled dot-product attention on a 5-token sequence with $d_{\text{model}} = 8$ and $d_k = 4$. Each function is mostly written for you &mdash; you fill in **one key line**. <!-- .element: class="text-lg" -->

:::columns cols="2" gap="30px"
**Unmasked attention (steps 1-5)**

Compute Q, K, V, raw scores, softmax weights, and the weighted output.
+++
**Causal mask and position (steps 6-8)**

Add a causal mask to block future tokens, then add positional embeddings and observe the change in attention patterns.
:::

---

:::step id="exercise-step2-code" title="Step 2: compute_qkv()"
```python
def compute_qkv(X: torch.Tensor, d_k: int = 4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project the token matrix X into query, key, and value matrices.

    Args:
        X: Token embeddings, shape (seq_len, d_model).
        d_k: Dimension of queries, keys, and values.

    Returns:
        (Q, K, V): Each of shape (seq_len, d_k).
    """
    torch.manual_seed(42)
    seq_len, d_model = X.shape
    W_Q = torch.randn(d_model, d_k) * 0.1
    W_K = torch.randn(d_model, d_k) * 0.1
    W_V = torch.randn(d_model, d_k) * 0.1

    # TODO: Compute Q, K, V by projecting X through W_Q, W_K, W_V
    # HINT: use matrix multiplication X @ W_Q, X @ W_K, X @ W_V
    raise NotImplementedError("TODO: compute Q, K, V projections")
```
+++
**Hint:** Use matrix multiplication `X @ W_Q`, `X @ W_K`, and `X @ W_V` to project the token embeddings.
+++
**Answer:**

```python
Q = X @ W_Q
K = X @ W_K
V = X @ W_V
return Q, K, V
```
:::

---

:::step id="exercise-step3-code" title="Step 3: raw_attention_scores()"
```python
def raw_attention_scores(Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Compute the pairwise compatibility scores: Q @ K^T.

    Args:
        Q: Query matrix, shape (seq_len, d_k).
        K: Key matrix, shape (seq_len, d_k).

    Returns:
        Scores matrix, shape (seq_len, seq_len).
    """
    # TODO: Compute the attention scores as the matrix product of Q and K^T
    # HINT: use Q @ K.T (the .T attribute transposes a 2D tensor)
    raise NotImplementedError("TODO: compute raw attention scores Q @ K^T")
```
+++
**Hint:** Use `Q @ K.T` to compute the matrix product of queries and transposed keys.
+++
**Answer:**

```python
return Q @ K.T
```
:::

---

:::step id="exercise-step4-code" title="Step 4: scaled_softmax()"
```python
def scaled_softmax(scores: torch.Tensor, d_k: int) -> torch.Tensor:
    """Scale scores by 1/sqrt(d_k) and apply softmax along the key dimension.

    Args:
        scores: Raw attention scores, shape (seq_len, seq_len).
        d_k: Dimension of keys (used for scaling).

    Returns:
        Attention weights, shape (seq_len, seq_len). Each row sums to 1.
    """
    # TODO: Scale the scores by 1/sqrt(d_k), then apply softmax along dim=-1
    # HINT: divide scores by (d_k ** 0.5), then call F.softmax(..., dim=-1)
    raise NotImplementedError("TODO: apply scaled softmax to attention scores")
```
+++
**Hint:** Divide `scores` by `(d_k ** 0.5)`, then call `F.softmax(..., dim=-1)`.
+++
**Answer:**

```python
return F.softmax(scores / (d_k ** 0.5), dim=-1)
```
:::

---

:::step id="exercise-step5-code" title="Step 5: attention_output()"
```python
def attention_output(weights: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Compute the attention output as a weighted sum of value vectors.

    Args:
        weights: Attention weights, shape (seq_len, seq_len).
        V: Value matrix, shape (seq_len, d_k).

    Returns:
        Output tensor, shape (seq_len, d_k).
    """
    # TODO: Compute the weighted sum of values using the attention weights
    # HINT: use weights @ V (matrix multiply the weight matrix by the value matrix)
    raise NotImplementedError("TODO: compute attention output as weighted sum of values")
```
+++
**Hint:** Multiply the weight matrix by the value matrix: `weights @ V`.
+++
**Answer:**

```python
return weights @ V
```
:::

---

:::terminal id="exercise-step5-output" title="Steps 1&ndash;5: Unmasked Attention Output" cmd="uv run python module_03_attention/src/main.py" caption="Unmasked attention weights are nearly uniform because the random projections are small. Each row sums to 1.0."
<span class="header">============================================================
STEP 1: Create Token Vectors
============================================================</span>
<span class="success">Shape: torch.Size([5, 8])</span>

<span class="header">============================================================
STEP 2: Compute Q, K, V Projections
============================================================</span>
<span class="success">Q shape: torch.Size([5, 4])</span>
<span class="success">K shape: torch.Size([5, 4])</span>
<span class="success">V shape: torch.Size([5, 4])</span>

<span class="header">============================================================
STEP 3: Raw Attention Scores (QK^T)
============================================================</span>
<span class="success">Shape: torch.Size([5, 5])</span>

<span class="header">============================================================
STEP 4: Scaled Softmax Attention Weights
============================================================</span>
<span class="success">Row sums (should be ~1.0): [1.0, 1.0, 1.0, 1.0, 1.0]</span>

<span class="header">============================================================
STEP 5: Attention Output (Weighted Sum of Values)
============================================================</span>
<span class="success">Shape: torch.Size([5, 4])</span>
:::

---

:::step id="exercise-step6-code" title="Step 6: causal_mask()"
```python
def causal_mask(seq_len: int) -> torch.Tensor:
    """Create a causal (lower-triangular) mask for autoregressive attention.

    A causal mask prevents each token from attending to future tokens.
    Entry (i, j) is 0 if j <= i (token i may attend to token j),
    and -inf if j > i (token i must NOT attend to token j).

    Args:
        seq_len: Length of the sequence.

    Returns:
        Mask tensor, shape (seq_len, seq_len), with 0s and -infs.
    """
    # TODO: Create a lower-triangular mask of 0s with -inf above the diagonal
    # HINT: use torch.tril(torch.ones(seq_len, seq_len)) for the lower triangle,
    #        then convert: 1 -> 0.0 and 0 -> float('-inf')
    raise NotImplementedError("TODO: create causal mask")
```
+++
**Hint:** Start with `torch.tril(torch.ones(seq_len, seq_len))`, then use `.masked_fill(mask == 0, float('-inf'))` to replace zeros with $-\infty$.
+++
**Answer:**

```python
mask = torch.tril(torch.ones(seq_len, seq_len))
mask = mask.masked_fill(mask == 0, float('-inf'))
return mask
```
:::

---

:::step id="exercise-step8-code" title="Step 8: add_positional_embeddings()"
```python
def add_positional_embeddings(X: torch.Tensor) -> torch.Tensor:
    """Add learned positional embeddings to the token representations.

    Args:
        X: Token embeddings, shape (seq_len, d_model).

    Returns:
        X_pos: Token embeddings plus positional embeddings, shape (seq_len, d_model).
    """
    torch.manual_seed(42)
    seq_len, d_model = X.shape
    P = torch.randn(seq_len, d_model) * 0.1

    # TODO: Add the positional embeddings P to the token embeddings X
    # HINT: simply add X + P (elementwise addition of the two tensors)
    raise NotImplementedError("TODO: add positional embeddings to token embeddings")
```
+++
**Hint:** Simply add the two tensors: `X + P`.
+++
**Answer:**

```python
return X + P
```
:::

---

:::terminal id="exercise-step8-output" title="Steps 6&ndash;8: Causal Mask and Positional Embeddings" cmd="uv run python module_03_attention/src/main.py" caption="The causal mask zeroes out the upper triangle, forcing each token to attend only to itself and earlier tokens."
<span class="header">============================================================
STEP 6: Causal Mask
============================================================</span>
<span class="success">Mask matrix:
[[  1. -inf -inf -inf -inf]
 [  1.   1. -inf -inf -inf]
 [  1.   1.   1. -inf -inf]
 [  1.   1.   1.   1. -inf]
 [  1.   1.   1.   1.   1.]]</span>

<span class="header">============================================================
STEP 7: Masked Attention Output
============================================================</span>
<span class="success">Masked weight matrix:
[[1.    0.    0.    0.    0.   ]
 [0.501 0.499 0.    0.    0.   ]
 [0.333 0.334 0.334 0.    0.   ]
 [0.25  0.251 0.25  0.25  0.   ]
 [0.2   0.2   0.2   0.2   0.2  ]]</span>

<span class="header">============================================================
STEP 8: Positional Embeddings
============================================================</span>
<span class="success">Shape: torch.Size([5, 8])</span>

<span class="header">============================================================
VISUALIZATIONS
============================================================</span>
<span class="success">Saved attention comparison to output/attention_comparison.png</span>
<span class="success">Saved positional effect to output/positional_effect.png</span>
:::

---

<!-- .slide: id="exercise-extra-credit" -->

## Extra Credit: KV Cache

Implement `kv_cache_step()` &mdash; simulate the KV cache used during autoregressive generation. Instead of recomputing keys and values for all tokens, cache them and only project the new token. <!-- .element: class="text-lg" -->

```python
def kv_cache_step(
    new_token: torch.Tensor,
    cached_keys: torch.Tensor,
    cached_values: torch.Tensor,
    W_Q: torch.Tensor, W_K: torch.Tensor, W_V: torch.Tensor,
    d_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # TODO: project new_token through W_Q, W_K, W_V, then append K and V to cache
    new_key = None
    new_value = None
    if new_key is None or new_value is None:
        raise NotImplementedError("Extra credit: project new_token through W_Q, W_K, W_V, then append K and V to cache")
```

The runner processes tokens one at a time and compares generation cost with and without the cache. <!-- .element: class="text-lg" style="margin-top: 15px;" -->

---

:::divider id="divider-quiz" title="Quiz" sub="Test your understanding"
:::

---

:::quiz id="quiz-q1" title="Q1: Learned Lookup"
In the attention mechanism, the query, key, and value projections are learned during training. If you froze these projections at random initialization (never trained them), what would the attention weights look like, and why?
+++
**Answer:** The attention weights would be approximately uniform &mdash; each token would attend equally to all other tokens. Random projections produce query and key vectors that are approximately orthogonal, so all dot products $QK^T$ are roughly equal, and softmax assigns nearly equal weight to every position. The learning is what makes the attention pattern **selective** rather than uniform.
:::

---

:::quiz id="quiz-q2" title="Q2: Causal Masking and Training"
A student proposes training a language model without a causal mask, arguing that "seeing future tokens during training will help the model learn faster." Explain why this approach fails at inference time.
+++
**Answer:** During inference, the model generates tokens one at a time and cannot see future tokens (they do not exist yet). If the model was trained with bidirectional attention, it learned to rely on future context that will not be available at inference time. The model's predictions would degrade because it is missing information it was trained to expect. Causal masking ensures the training and inference information constraints match.
:::

---

:::quiz id="quiz-q3" title="Q3: Positional Information"
You feed the sentence "the cat sat on the mat" into a Transformer with positional embeddings removed. You then reverse the sentence to "mat the on sat cat the" and feed it in again. How do the attention patterns compare, and what does this tell you?
+++
**Answer:** Without positional information, the attention patterns are identical up to the permutation of tokens. Self-attention is permutation equivariant: if you permute the input, the output is the same permutation of the original output. The model cannot distinguish "the cat sat" from "sat cat the" because it has no way to know which token came first. Positional embeddings break this symmetry by giving each position a unique signature.
:::

---

:::quiz id="quiz-q4" title="Q4: O(n^2) Scaling"
GPT-4 Turbo supports 128K token context. If you doubled the context to 256K tokens, by what factor does the attention computation cost increase? Why is this a bigger problem for attention than for the feed-forward layers?
+++
**Answer:** The attention cost increases by a factor of 4 (from $O(n^2)$ to $O((2n)^2) = O(4n^2)$). The feed-forward layers process each token independently, so their cost is $O(n)$ &mdash; doubling context doubles the feed-forward cost. The $O(n^2)$ attention cost is why long context is disproportionately expensive, and why techniques like FlashAttention, sliding-window attention, and sparse attention patterns are active research areas.
:::

---

:::quiz id="quiz-q5" title="Q5: KV Cache Trade-offs"
A model has 32 layers, 32 attention heads per layer, and $d_k = 128$ per head. How many floats are stored in the KV cache for a 4096-token sequence? If you switch from multi-head attention to grouped-query attention with 4 groups, what is the reduction factor?
+++
**Answer:** With multi-head attention: $2 \times 32 \times 32 \times 4096 \times 128 = 1,073,741,824$ floats (about 4 GB in float32). The "2" accounts for both K and V. With GQA and 4 groups, K and V are shared within each group of 8 heads, so the cache for K and V shrinks by a factor of 8: $2 \times 32 \times 4 \times 4096 \times 128 \approx 134$ million floats (about 512 MB). The reduction factor is 8, equal to the number of heads per group.
:::

---

<!-- .slide: id="resources" -->

## References and Further Reading

- Bahdanau, D., Cho, K., &amp; Bengio, Y. (2014). "Neural Machine Translation by Jointly Learning to Align and Translate." *arXiv:1409.0473*.
- Vaswani, A. et al. (2017). "Attention Is All You Need." *NeurIPS*. *arXiv:1706.03762*.
- Dao, T. et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." *arXiv:2205.14135*.
- Darcet, O. et al. (2023). "Vision Transformers Need Registers." *arXiv:2309.16588*.
- Su, J. et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding." *arXiv:2104.09864*.
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) &mdash; visual guide to attention and encoder-decoder
- [BertViz](https://codecut.ai/bertviz-visualize-attention-in-transformer-language-models/) &mdash; attention visualization tool

---

:::divider id="end" title="Questions?" sub="Next: Module 4 &mdash; The Transformer"
:::
