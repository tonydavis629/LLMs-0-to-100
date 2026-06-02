:::divider id="divider-block" title="Anatomy of a Transformer Block" sub="How one layer transforms its input"
:::

---

<!-- .slide: id="embedding-layer" -->

## The Embedding Layer

Each token ID indexes one row of a learned embedding matrix $W_E \in \mathbb{R}^{V \times d_{\text{model}}}$.

$$\mathbf{x}_i = W_E[t_i] + W_P[i]$$

The positional embedding $W_P[i]$ adds position information. Without this addition, a bag of attention scores has no inherent notion of order: shuffling the input tokens would leave the output unchanged.

The result is a sequence of vectors $\mathbf{x}_1, \dots, \mathbf{x}_n$ that flows into the first transformer block.

---

<!-- .slide: id="residual-stream" -->

## The Residual Stream

Picture the transformer not as a pipeline but as a **running vector** that flows through the network:

$$\mathbf{x}_{\text{out}} = \mathbf{x}_{\text{in}} + \text{sub-layer}(\text{norm}(\mathbf{x}_{\text{in}}))$$

Each sub-layer reads from the stream, computes its contribution, and writes it back. The stream is never replaced; it is only updated by addition. This means:

- Every block can still "see" the original embedding signal
- Gradients have a direct highway back to the input through the residual adds
- The entire depth of the model relies on this highway to avoid vanishing gradients

---

<!-- .slide: id="attention-sublayer" -->

## Sub-layer 1: Multi-Head Self-Attention

Queries, keys, and values are projected from the normalized residual stream. Attention computes a weighted mixture of values, then the output is projected back to $d_{\text{model}}$ and added to the stream:

$$\mathbf{x} = \mathbf{x} + \text{Attention}(\text{LayerNorm}(\mathbf{x}))$$

Attention is where tokens communicate. It is the only operation where information moves between positions in the same layer. The FFN processes each position independently.

---

<!-- .slide: id="ffn-sublayer" -->

## Sub-layer 2: The Feed-Forward Network

After attention, every position is processed by the same two-layer MLP:

$$\text{FFN}(\mathbf{x}) = W_2 \cdot \text{GELU}(W_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2$$

The hidden dimension is typically about $4 \times d_{\text{model}}$ (3,072 for GPT-2 small). This is applied independently at every position, with no mixing between tokens.

A large share of a model's parameters and stored **"knowledge"** live here. Each FFN neuron can be viewed as a key-value pair: when the input activates a neuron (via ReLU/GELU), it retrieves a stored output vector. The FFN is where facts like "Paris is the capital of France" are encoded.

---

<!-- .slide: id="norm-and-residual" -->

## Residual Connections and Normalization

Residual connections are the gradient highway that makes deep stacks trainable. Without them, gradients would decay exponentially with depth.

Normalization stabilizes the distribution of activations:

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

- **Post-norm** (original transformer): normalize after the sub-layer output
- **Pre-norm** (modern default): normalize before the sub-layer input, then add the residual. More stable to train at depth.

**RMSNorm** is the modern simplification: drop the mean-centering step and just divide by the root-mean-square.

---

<!-- .slide: id="stacking-and-head" -->

## Stacking Blocks and the Output Head

Identical blocks are stacked $N$ times to build depth:

$$\text{block}_N(\dots \text{block}_2(\text{block}_1(\mathbf{x}_{\text{embed}})) \dots) = \mathbf{x}_{\text{final}}$$

After the last block, a final layer norm and the **language-modeling head** project back to vocabulary-sized logits:

$$\text{logits} = W_{\text{head}} \cdot \text{LayerNorm}(\mathbf{x}_{\text{final}})$$

**Weight tying** reuses the embedding matrix as the output projection: $W_{\text{head}} = W_E^T$. This saves parameters and enforces that input and output live in the same semantic space.

---

<!-- .slide: id="forward-pass-and-params" -->

## One Forward Pass: Tokens to Logits

Trace the full path:

1. Token IDs $\rightarrow$ embedding lookup + position
2. Pass through $N$ transformer blocks (attention + FFN, each with pre-norm and residual)
3. Final layer norm
4. Unembedding / LM head $\rightarrow$ logits
5. Softmax $\rightarrow$ next-token probabilities

Parameter counting for GPT-2 small ($d_{\text{model}}=768$, $N=12$, $V=50257$):

| Component | Parameters | Share |
|-----------|-----------|-------|
| Embeddings | $V \cdot d$ | ~38% |
| Attention (all layers) | $N \cdot 4 d^2$ | ~29% |
| FFN (all layers) | $N \cdot 8 d^2$ | ~29% |
| Output head (tied) | 0 (shares embed) | 0% |
| **Total** | **~124M** | **100%** |

---

<!-- .slide: id="side-quest-residual" -->

## Side Quest: The Residual Stream and Induction Heads

Mechanistic interpretability reads the transformer as components reading from and writing to a shared residual stream. **Induction heads** (Olsson et al., 2022) implement a simple copy-and-continue pattern:

- When the model sees a pattern like `[A] [B] ... [A]`, it predicts `[B]` as the next token
- This is the mechanism behind in-context learning: the model copies a pattern it observed earlier in the context

Induction heads are linked to the emergence of few-shot learning at scale. They make the residual-stream metaphor concrete: each head reads the stream, detects a previous occurrence of a token, and writes a prediction back.
