:::divider id="divider-quiz" title="Quiz"
:::

---

:::quiz id="quiz-ffn" title="Why Does the Feed-Forward Network Matter?"
The attention sub-layer mixes information between positions, but the feed-forward network (FFN) processes each position independently. Why is the FFN still essential?
+++
The FFN holds most of the parameters and stored "knowledge."

- Each neuron acts like a key-value pair: input activates it, it retrieves a learned output vector
- Without it, the model only mixes existing embeddings and cannot store or recall facts
:::

---

:::quiz id="quiz-pre-post-norm" title="Pre-Norm vs Post-Norm"
The original transformer placed LayerNorm after each sub-layer (post-norm). Modern models place it before (pre-norm). Why is pre-norm preferred?
+++
Pre-norm trains more stably at depth.

- Post-norm: gradients pass through LayerNorm before the residual add; deep stacks compound this into vanishing or exploding gradients
- Pre-norm: normalization sits outside the residual path, so the gradient highway stays clean
:::

---

:::quiz id="quiz-decoder-only" title="Why Did Decoder-Only Win for LLMs?"
Decoder-only models are not universally better. What specific advantages made them the dominant LLM architecture?
+++
- One stack: simpler, scales cleanly
- Next-token objective gives a dense training signal at every position
- Generation subsumes "understanding": classification becomes text completion
- In-context learning emerges at scale without task-specific heads
:::

---

:::quiz id="quiz-token-tax" title="The Tokenization Trade-Off"
A larger vocabulary means shorter sequences but a larger embedding matrix. Why does non-English text often cost more tokens per sentence?
+++
- English dominates training corpora, so more English words fit whole in the vocabulary
- Other scripts and low-resource languages split into more pieces
- The "token tax": same content, more tokens, higher cost and context usage
:::

---

:::quiz id="quiz-moe" title="How Does MoE Add Parameters Without Adding Compute?"
Mixture of Experts replaces one large FFN with many smaller FFNs plus a router. Why does total compute per token stay roughly constant?
+++
- The router picks a small subset per token (e.g., top-2 of 64 experts)
- Most expert parameters stay inactive each forward pass
- Parameters scale with expert count; compute scales only with the activated subset
:::

---

:::quiz id="quiz-temperature" title="The Effect of Temperature on Sampling"
Temperature $T$ scales logits before softmax. How does a very high temperature change the sample distribution, and why might that be undesirable?
+++
- High $T$ flattens the distribution: unlikely tokens become nearly as probable as likely ones
- Output turns random and ungrammatical
- Very low $T$ sharpens the peak: repetitive, deterministic text
- Typical range: $T \approx 0.7$ to $1.0$
:::
