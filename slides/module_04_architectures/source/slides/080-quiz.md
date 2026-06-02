:::divider id="divider-quiz" title="Quiz"
:::

---

:::quiz id="quiz-ffn" title="Why Does the Feed-Forward Network Matter?"
The attention sub-layer mixes information between positions, but the feed-forward network (FFN) processes each position independently. Why is the FFN still crucial?
+++
The FFN is where most of the model's parameters and stored "knowledge" live. Each neuron acts like a key-value pair: when its input activates it, the neuron retrieves a learned output vector. Without the FFN, the model would only mix existing embeddings and would have no way to store or recall facts.
:::

---

:::quiz id="quiz-pre-post-norm" title="Pre-Norm vs Post-Norm"
The original transformer placed LayerNorm after each sub-layer (post-norm). Modern models place it before (pre-norm). Why is pre-norm preferred?
+++
Pre-norm is more stable to train at depth. With post-norm, gradients must propagate through the normalization layer before reaching the residual add, and in deep stacks this compounds into vanishing or exploding gradients. Pre-norm puts the normalization outside the residual path, keeping the gradient highway clean.
:::

---

:::quiz id="quiz-decoder-only" title="Why Did Decoder-Only Win for LLMs?"
Decoder-only models are not universally better. What specific advantages made them the dominant LLM architecture?
+++
A single stack is simpler and scales cleanly. The next-token objective gives a dense training signal at every position. The generative objective subsumes "understanding" tasks: classification can be phrased as text completion. And emergent in-context learning appears at scale without task-specific heads.
:::

---

:::quiz id="quiz-token-tax" title="The Tokenization Trade-Off"
A larger vocabulary means shorter sequences but a larger embedding matrix. Why does non-English text often cost more tokens per sentence?
+++
English is overrepresented in training corpora, so English subword sequences are shorter (more words fit whole in the vocabulary). Languages with different scripts or less training data are split into more pieces. This is the "token tax": the same semantic content requires more tokens, inflating both cost and context-window usage.
:::

---

:::quiz id="quiz-moe" title="How Does MoE Add Parameters Without Adding Compute?"
Mixture of Experts replaces one large FFN with many smaller FFNs plus a router. Why does total compute per token stay roughly constant?
+++
The router selects only a small subset of experts (e.g., top-2 out of 64) for each token. Most expert parameters are inactive on any given forward pass. Parameter count scales with the number of experts, but compute scales only with the activated subset.
:::

---

:::quiz id="quiz-temperature" title="The Effect of Temperature on Sampling"
Temperature $T$ scales logits before softmax. How does a very high temperature change the sample distribution, and why might that be undesirable?
+++
High temperature flattens the distribution, making unlikely tokens nearly as probable as likely ones. The output becomes random, ungrammatical, and thematically inconsistent. Conversely, very low temperature sharpens the peak, producing repetitive, deterministic text. The sweet spot is typically $T \approx 0.7 \text{ to } 1.0$.
:::
