# Module 3: Attention Mechanisms — Lecture Notes

Citations, math, and explanations for every claim in the presentation.

## Why an MLP Is Not Enough for Language

### Fixed Context Windows
- An MLP over a fixed-size window of tokens has four fundamental problems:
  1. **Fixed context:** token 1 cannot directly inform token 50; the window is rigid.
  2. **Position destruction:** flattening token vectors treats the same pattern at position 3 and position 30 differently, destroying weight sharing.
  3. **Parameter growth:** the input size (and therefore parameter count) scales with context length.
  4. **No selection mechanism:** there is no way for one token to selectively retrieve information from another token.
- **Connection to Module 2:** the MLP is a universal function approximator, but the architecture does not match the structure of sequences. The same lesson from XOR applies at a larger scale: the architecture must match the data.

## Why Attention?

### The Fixed-Context Bottleneck
- Earlier sequence models (RNNs, LSTMs) compressed the entire input sequence into a single fixed-size hidden state vector.
- **Problem:** no matter how long the input, all information must fit through one fixed-size bottleneck. The model must decide what to keep before it knows what will be needed later.
- Attention solves this by giving each output token direct access to all input representations, eliminating the compression step.

### Attention as Learned Lookup
- Attention can be understood as a differentiable database query:
  - **Query:** what the current token is looking for
  - **Key:** what each token advertises about itself
  - **Value:** the actual content to aggregate
- The model learns its own queries, keys, and values through training. No manual feature engineering is needed.
- **Reference:** Bahdanau, D., Cho, K., & Bengio, Y. (2014). "Neural Machine Translation by Jointly Learning to Align and Translate." *arXiv:1409.0473*.

## Queries, Keys, and Values

### The Database Analogy
- A traditional database maps a query to records using an exact key match. Attention does soft (differentiable) matching between query and key vectors.
- Key differences: attention returns a weighted combination of all values (not one record), is fully differentiable, and the queries, keys, and values are all learned.

### Q, K, V Projections
- Each token vector $\mathbf{x}_i$ is projected into three different spaces:

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

where $X$ is the token matrix of shape $(n, d_{\text{model}})$, $W_Q, W_K, W_V$ are learned projection matrices of shape $(d_{\text{model}}, d_k)$, and $Q, K, V$ are the projected matrices of shape $(n, d_k)$.

- The same token can "ask" different questions (via $Q$) than it "advertises" (via $K$) or "offers" (via $V$).
- $d_k$ can be smaller than $d_{\text{model}}$, creating a bottleneck that forces the model to focus on the most informative projections.

### Pairwise Token Compatibility
- The score matrix $QK^T$ measures how well each query aligns with each key:

$$\text{score}(i, j) = \mathbf{q}_i \cdot \mathbf{k}_j$$

- This is a dot product — a measure of alignment. When query $i$ points in a similar direction to key $j$, the score is large.

## Scaled Dot-Product Attention

### Why Scale by $\sqrt{d_k}$?
- For two random vectors of dimension $d_k$ with zero mean and unit variance, the expected magnitude of their dot product is $\sqrt{d_k}$.
- When inputs to softmax are large, the output becomes sharply peaked (one entry near 1, the rest near 0), producing tiny gradients that make training unstable.
- Dividing by $\sqrt{d_k}$ keeps the variance of the dot product roughly constant regardless of dimension, preserving gradient flow and stabilizing training.
- **Reference:** Vaswani, A. et al. (2017). "Attention Is All You Need." *NeurIPS*. *arXiv:1706.03762*.

### The Full Formula
- Scaled dot-product attention:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

- Each output token is a weighted average of value vectors, where the weights come from query-key compatibility.
- If token $i$ attends strongly to token $j$, the output at position $i$ is dominated by $\mathbf{v}_j$.
- The output dimension matches the value dimension ($d_k$), not the sequence length.

## Attention Masks

### Padding Masks
- In a batch, sequences have different lengths. Shorter sequences are padded with special tokens.
- A padding mask sets padded positions to $-\infty$ before softmax, so they receive zero weight after normalization.
- Padding masks are a practical necessity for batched training.

### Causal Masks
- A causal mask prevents each token from attending to future tokens. Entry $(i, j)$ is 0 if $j \leq i$ and $-\infty$ if $j > i$.
- **Bidirectional attention** (BERT-style): every token can attend to every other token. Useful for understanding the full context.
- **Causal attention** (GPT-style): each token can only attend to itself and earlier tokens.
- **Why causal masking is necessary:** In a language model, we predict $P(w_t \mid w_1, w_2, \ldots, w_{t-1})$. If token $t$ could see token $t+1$ during training, the model would be cheating — it could copy the answer instead of learning to predict. Causal masking enforces the same information constraint during training that the model will face during inference.
- **Reference:** Devlin, J. et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." *NAACL*. Radford, A. et al. (2018). "Improving Language Understanding by Generative Pre-Training." OpenAI.

## Multi-Head Self-Attention

### Why Multiple Heads?
- A single attention head produces one weighted average per token. But a token might need to attend to different things for different reasons (e.g., syntactic dependencies, coreference, positional proximity).
- Multiple heads let the model learn different ways to compare tokens simultaneously.

### Multi-Head Mechanics
- Each head $h$ has its own projections:

$$Q_h = XW_Q^h, \quad K_h = XW_K^h, \quad V_h = XW_V^h$$

- Each head independently computes scaled dot-product attention:

$$\text{head}_h = \text{Attention}(Q_h, K_h, V_h)$$

- The heads are concatenated and projected back:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) \, W_O$$

- **Compute budget:** total projection size across all heads is typically kept equal to $d_{\text{model}}$. With 8 heads and $d_{\text{model}} = 512$, each head uses $d_k = 64$. Multi-head attention is not more expensive than single-head — it is the same compute, split into independent patterns.

## Cross Attention

### Self-Attention vs. Cross-Attention
- **Self-attention:** $Q$, $K$, and $V$ all come from the same sequence. Every token looks at every other token in the same sequence.
- **Cross-attention:** $Q$ comes from one sequence; $K$ and $V$ come from another. One sequence selects information from the other.

### Encoder-Decoder Use Case
- The original Transformer used both self-attention and cross-attention:
  - **Encoder:** self-attention over the input sequence, producing contextualized representations.
  - **Decoder:** causal self-attention over the output sequence, plus cross-attention where decoder queries attend to encoder keys and values.
- Modern LLMs (GPT, LLaMA) use only the decoder with causal self-attention. The encoder-decoder pattern remains common in translation and speech models.

## Positional Embeddings

### Permutation Equivariance
- Self-attention without position information is permutation equivariant: if you permute the input, the output is the same permutation of the original output.
- The model cannot distinguish "dog bites man" from "man bites dog" because both have the same bag of tokens.
- **Reference:** The original Transformer paper (Vaswani et al., 2017) introduced sinusoidal positional encodings specifically to address this.

### Learned Positional Embeddings
- A trainable embedding table with one vector per position, added to the token embeddings:

$$X_{\text{pos}} = X + P$$

where $P$ is a learned matrix of shape (max_len, $d_{\text{model}}$).
- Simple and effective. Used in BERT, GPT-2.
- **Limitation:** cannot extrapolate beyond the maximum sequence length seen during training.

### Sinusoidal Positional Encodings
- Fixed patterns based on sine and cosine at different frequencies:

$$PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

- No parameters to learn. Theoretically allows extrapolation to unseen lengths (though in practice this is limited).
- The choice of $10000$ as the base creates a range of frequencies from very fast (low dimensions) to very slow (high dimensions).

### Relative Position and RoPE
- Both learned and sinusoidal encodings are absolute — they encode "position 5" as a fixed vector. What often matters is relative position: "two tokens to the left."
- **RoPE** (Rotary Position Embedding, Su et al. 2021) encodes relative position by rotating query and key vectors. The key property is:

$$\mathbf{q}_m^T \mathbf{k}_n = f(m)^T f(n) = g(m - n)$$

- The dot product depends only on the relative offset $m - n$, not the absolute positions.
- RoPE is used in LLaMA, Mistral, and most modern LLMs.
- **Reference:** Su, J. et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding." *arXiv:2104.09864*.

## Memory and Compute Tradeoffs

### The $O(n^2)$ Cost
- Attention creates an $n \times n$ score matrix for $n$ tokens.
- **Compute:** $O(n^2 \cdot d_k)$ for the matrix multiplications.
- **Memory:** $O(n^2)$ to store the attention weights.
- Doubling the context length quadruples the attention cost.
- For 128K tokens, the attention matrix has approximately 16 billion entries.

### FlashAttention
- **Reference:** Dao, T. et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." *arXiv:2205.14135*.
- FlashAttention computes exact attention without materializing the full $n \times n$ matrix in GPU HBM (High Bandwidth Memory).
- It tiles the computation to keep intermediate results in fast SRAM (Static Random-Access Memory), avoiding expensive HBM reads/writes.
- Same mathematical result as standard attention, but 2-4x faster and 5-20x less memory.
- FlashAttention-2 and -3 pushed further gains on newer GPU architectures.

### Sliding-Window Attention
- Each token attends only to a local window of neighbors (plus a few global tokens).
- Reduces the effective $n$ in the $O(n^2)$ cost.
- **Reference:** Beltagy, I., Peters, M. E., & Cohan, A. (2020). "Longformer: The Long-Document Transformer." *arXiv:2004.05150*.

### Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)
- **MQA:** All heads share the same K and V projections. Only Q is per-head. Reduces KV cache size significantly.
- **GQA:** A compromise: heads are divided into groups that share K and V. Fewer unique K/V projections than standard multi-head attention, but more than MQA.
- GQA is used in LLaMA 2/3.
- **Reference:** Shazeer, N. (2019). "Fast Transformer Decoding: One Write-Head is All You Need." *arXiv:1911.02150*. Ainslie, J. et al. (2023). "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." *EMNLP*.

## KV Cache

### The KV Cache Idea
- During autoregressive inference, the model generates one token at a time.
- **Without KV cache:** each generation step recomputes K and V for every previous token. Cost per step grows linearly with sequence length. Total cost is $O(n^2)$.
- **With KV cache:** previous keys and values are stored and reused. Each new token only computes its own K and V. Cost per step is constant ($O(1)$). Total cost is $O(n)$.

### KV Cache Tradeoffs
- The KV cache stores $2 \times n_{\text{layers}} \times n_{\text{heads}} \times n_{\text{tokens}} \times d_k$ floats.
- For a 70B model with 128K context, the KV cache can be tens of gigabytes.
- KV cache memory is a major limit for long context and large-batch serving. This is why MQA and GQA exist — fewer unique K and V heads means a smaller cache.

## Notable Figures

### Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio
- **Paper:** Bahdanau, D., Cho, K., & Bengio, Y. (2014). "Neural Machine Translation by Jointly Learning to Align and Translate." *arXiv:1409.0473*.
- Their attention mechanism was originally a fix for the bottleneck of fixed-size context vectors in encoder-decoder models.
- The paper made attention a first-class mechanism, not just a patch for RNNs.

### Vaswani et al.
- **Paper:** Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). "Attention Is All You Need." *NeurIPS*. *arXiv:1706.03762*.
- Replaced recurrence entirely with attention and feed-forward layers.
- Introduced scaled dot-product multi-head attention.
- One of the most cited papers in machine learning.

### Tri Dao
- **Paper:** Dao, T., Fu, D., Ermon, S., Rudra, A., & Ré, C. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." *arXiv:2205.14135*.
- Showed how careful memory-aware attention algorithms can make exact attention much faster on GPUs.
- FlashAttention-2 (2023) and FlashAttention-3 (2024) pushed further gains.

## Side Quests

### Register Tokens in Vision Transformers
- **Paper:** Darcet, O., Oquab, M., Mairal, J., & Bojanowski, P. (2023). "Vision Transformers Need Registers." *arXiv:2309.16588*.
- Observed high-norm artifact tokens in ViT feature maps, often in low-information background patches.
- Interpretation: the model repurposes some patch tokens as internal scratch space for computation.
- Register tokens add extra learned token positions so the model has dedicated places for internal computation.
- This is a concrete example of token positions having roles beyond "word at position $i$."

### Attention Maps Are Not Always Explanations
- Attention weights can be useful visual diagnostics — they show where the model is looking.
- However, high attention does not always prove causal importance. Attention is one of many operations in the network; the model may compensate through other pathways.
- **Useful distinction:** visualization can suggest hypotheses, but ablation tests are needed for stronger claims.
- **Reference:** Jain, S. & Wallace, B. C. (2019). "Attention is not Explanation." *NAACL*. Wiegreffe, S. & Pinter, Y. (2019). "Attention is not not Explanation." *EMNLP*.

## Exercise Notes

### Exercise Structure
- The single student-facing file is `exercises/module_03_attention/exercise.py`. Steps: (1) `make_token_vectors` (provided), (2) `compute_qkv`, (3) `raw_attention_scores`, (4) `scaled_softmax`, (5) `attention_output`, (6) `causal_mask`, (7) `masked_attention` (provided, uses steps 3-6), (8) `add_positional_embeddings`, extra credit `kv_cache_step`.

### Parameters
- `vocab_size = 10`, `d_model = 8`, `d_k = 4`, `seq_len = 5`
- Token IDs: `[2, 5, 1, 8, 3]` (fixed for reproducibility)
- Token labels: `["the", "cat", "sat", "on", "mat"]`
- Random seed: 42 for all random operations

### Expected Output
- With random seed 42 and small weight initialization (`* 0.1`), the unmasked attention weights are approximately uniform (each entry near 0.2), because the random projections produce nearly orthogonal query and key vectors.
- The causal mask creates a lower-triangular pattern: token 0 attends only to itself (weight 1.0), token 1 attends equally to tokens 0 and 1 (each near 0.5), and so on.
- Adding positional embeddings shifts the token representations, which slightly changes the attention pattern.

### Causal Mask Implementation
- The mask uses `torch.tril(torch.ones(seq_len, seq_len))` to create the lower triangle, then `masked_fill(mask == 0, float('-inf'))` to set upper-triangle entries to $-\infty$.
- After softmax, $-\infty$ entries become exactly 0, and the remaining entries are re-normalized to sum to 1.

### KV Cache Extra Credit
- The `kv_cache_step` function projects the new raw token embedding through all three weight matrices ($W_Q$, $W_K$, $W_V$), appends the new key and value to the cached tensors, and computes attention using only the new query and all cached keys and values.
- The runner compares generation cost with and without the KV cache, plotting time per token vs. sequence length.
