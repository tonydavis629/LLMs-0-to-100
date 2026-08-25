:::divider id="divider-quiz" title="Quiz" sub="Test your understanding"
:::

---

:::quiz id="quiz-q1" title="Q1: Learned Lookup"
In the attention mechanism, the query, key, and value projections are learned during training. If you froze these projections at random initialization (never trained them), what would the attention weights look like, and why?
+++
**Answer:** Approximately uniform. Random projections give roughly orthogonal query and key vectors, so all dot products are about equal and softmax spreads weight evenly. Learning is what makes attention **selective**.
:::

---

:::quiz id="quiz-q2" title="Q2: Causal Masking and Training"
A student proposes training a language model without a causal mask, arguing that "seeing future tokens during training will help the model learn faster." Explain why this approach fails at inference time.
+++
**Answer:** At inference, future tokens do not exist. A model trained with bidirectional attention learned to rely on future context it will never have, so its predictions degrade. Causal masking makes the training and inference constraints match.
:::

---

:::quiz id="quiz-q3" title="Q3: Positional Information"
You feed the sentence "the cat sat on the mat" into a self-attention layer with positional embeddings removed. You then reverse the sentence to "mat the on sat cat the" and feed it in again. How do the attention patterns compare, and what does this tell you?
+++
**Answer:** Identical up to the permutation. Self-attention is permutation equivariant: permute the input and the output permutes the same way. The model cannot tell which token came first. Positional embeddings break this symmetry.
:::

---

:::quiz id="quiz-q4" title="Q4: O(n^2) Scaling"
A model supports a 128K-token context. If you doubled the context to 256K tokens, by what factor does the attention computation cost increase? Why is this a bigger problem for per-token feed-forward layers?
+++
**Answer:** Attention cost quadruples: $O((2n)^2) = O(4n^2)$. Feed-forward layers process tokens independently, so their cost only doubles. This quadratic growth is why long context is disproportionately expensive and why FlashAttention, sliding-window, and sparse attention exist.
:::

---

:::quiz id="quiz-q5" title="Q5: KV Cache Trade-offs"
A model has 32 layers, 32 attention heads per layer, and $d_k = 128$ per head. How many floats are stored in the KV cache for a 4096-token sequence? If you switch from multi-head attention to grouped-query attention with 4 groups, what is the reduction factor?
+++
**Answer:** Multi-head: $2 \times 32 \times 32 \times 4096 \times 128 = 1,073,741,824$ floats (about 4 GB in float32); the "2" covers K and V. GQA with 4 groups shares K and V across each group of 8 heads: $2 \times 32 \times 4 \times 4096 \times 128 \approx 134$ million floats (about 512 MB). Reduction factor: 8, the number of heads per group.
:::

---

<!-- .slide: id="resources" -->

## References and Further Reading

- Bahdanau, D., Cho, K., &amp; Bengio, Y. (2014). "Neural Machine Translation by Jointly Learning to Align and Translate." *arXiv:1409.0473*.
- Dao, T. et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." *arXiv:2205.14135*.
- Su, J. et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding." *arXiv:2104.09864*.
- Xiao, G. et al. (2023). "Efficient Streaming Language Models with Attention Sinks." *arXiv:2309.17453*.
- Huang, Z. et al. (2020). "Pixel-BERT: Aligning Image Pixels with Text by Deep Multi-Modal Transformers." *arXiv:2004.00849*.
- [BertViz](https://codecut.ai/bertviz-visualize-attention-in-transformer-language-models/) &mdash; attention visualization tool
