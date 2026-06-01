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
You feed the sentence "the cat sat on the mat" into a self-attention layer with positional embeddings removed. You then reverse the sentence to "mat the on sat cat the" and feed it in again. How do the attention patterns compare, and what does this tell you?
+++
**Answer:** Without positional information, the attention patterns are identical up to the permutation of tokens. Self-attention is permutation equivariant: if you permute the input, the output is the same permutation of the original output. The model cannot distinguish "the cat sat" from "sat cat the" because it has no way to know which token came first. Positional embeddings break this symmetry by giving each position a unique signature.
:::

---

:::quiz id="quiz-q4" title="Q4: O(n^2) Scaling"
A model supports a 128K-token context. If you doubled the context to 256K tokens, by what factor does the attention computation cost increase? Why is this a bigger problem for per-token feed-forward layers?
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
- Dao, T. et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." *arXiv:2205.14135*.
- Su, J. et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding." *arXiv:2104.09864*.
- Xiao, G. et al. (2023). "Efficient Streaming Language Models with Attention Sinks." *arXiv:2309.17453*.
- Huang, Z. et al. (2020). "Pixel-BERT: Aligning Image Pixels with Text by Deep Multi-Modal Transformers." *arXiv:2004.00849*.
- [BertViz](https://codecut.ai/bertviz-visualize-attention-in-transformer-language-models/) &mdash; attention visualization tool
