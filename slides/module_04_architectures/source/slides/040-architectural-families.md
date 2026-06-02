:::divider id="divider-families" title="Three Architectural Families" sub="Encoder-decoder, encoder-only, and decoder-only"
:::

---

<!-- .slide: id="encoder-decoder" -->

## Family 1: Encoder-Decoder

The original transformer (Vaswani et al., 2017). The encoder builds bidirectional representations; the decoder generates while attending back through cross-attention.

:::columns cols="2" gap="30px"
**Encoder stack**

Bidirectional self-attention (no causal mask). Each token can attend to every other token in the input. Produces a sequence of context vectors.
+++
**Decoder stack**

Causal self-attention (masked) plus cross-attention to the encoder output. Each generated token can attend to all encoder tokens and all previously generated tokens.
:::

Natural fit for sequence-to-sequence tasks: translation, summarization, where input and output are clearly distinct sequences. T5 and BART follow this design.

---

<!-- .slide: id="encoder-only" -->

## Family 2: Encoder-Only

Bidirectional attention, no causal mask, no decoder. Trained with **masked language modeling** (forward reference to Module 5): some input tokens are hidden, and the model must predict them from their context on both sides.

$$\text{output}_i = f(\mathbf{x}_1, \dots, \mathbf{x}_n) \quad \text{for all } i$$

Use cases: classification, sentence embeddings, retrieval, named-entity recognition. The model produces representations, not generations.

BERT (Devlin et al., 2018) made this architecture dominant for "understanding" tasks. RoBERTa improved the training recipe without changing the architecture.

---

:::figure img="images/devlin_chang_lee_toutanova.jpg" name="Jacob Devlin, Ming-Wei Chang, Kenton Lee & Kristina Toutanova" kicker="Introduced BERT (2018)"
- Devlin et al. introduced bidirectional pretraining with masked language modeling
- Before BERT, language models were either left-to-right or shallowly bidirectional
- BERT made encoder-only transformers the standard for understanding tasks
- Devlin et al., <https://arxiv.org/abs/1810.04805>
:::

---

<!-- .slide: id="decoder-only" -->

## Family 3: Decoder-Only

Causal masking, autoregressive next-token prediction. Each token attends only to itself and previous tokens.

$$\text{output}_t = f(\mathbf{x}_1, \dots, \mathbf{x}_t)$$

Built for generation, but capable of nearly everything through prompting. GPT (Radford et al., 2018), GPT-2 (2019), GPT-3 (2020), and the Llama lineage all share this design.

Why decoder-only won for LLMs:

- A single stack is simpler and scales cleanly
- The next-token objective gives a dense training signal at every position
- The generative objective subsumes "understanding" tasks: you can phrase classification as a completion prompt
- No task-specific heads are needed; everything becomes text generation
- In-context and few-shot learning emerge at scale

---

:::figure img="images/radford_gpt.jpg" name="Alec Radford & OpenAI Team" kicker="Introduced GPT and GPT-2 (2018-2019)"
- Radford et al. demonstrated that autoregressive pretraining on unlabeled text produces powerful general-purpose models
- GPT-2 showed that scale alone (1.5B parameters) produces coherent multi-paragraph text
- The decoder-only design became the dominant LLM paradigm
- Radford et al., "Improving Language Understanding by Generative Pre-Training" (2018); "Language Models are Unsupervised Multitask Learners" (2019)
:::

---

<!-- .slide: id="honest-tradeoffs" -->

## The Honest Trade-offs

Decoder-only did **not** win on every axis. It won on generality and scaling:

- **Encoder-only** is still more efficient for fixed-vector embeddings, retrieval, and classification. If you only need a sentence representation, BERT-style models are faster.

- **Encoder-decoder** remains strong where input and output are clearly distinct, as in machine translation. The input can be processed bidirectionally while the output is generated autoregressively.

- **Decoder-only** excels when the task is open-ended or when you want a single model to handle many tasks through prompting. The price is that every task must be phrased as text completion.

The choice depends on the problem, not on decoder-only being universally superior.
