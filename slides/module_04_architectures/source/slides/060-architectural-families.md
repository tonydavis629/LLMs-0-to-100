:::divider id="divider-families" title="Three Architectural Families" sub="Encoder-decoder, encoder-only, and decoder-only"
:::

---

<!-- .slide: id="encoder-decoder" -->

## Family 1: Encoder-Decoder

<div class="encoder-decoder-layout">
  <img src="images/transformer.webp" alt="Original transformer encoder-decoder architecture">
  <div>
    <p>The original transformer has two stacks: an encoder for the input and a decoder for the output.</p>
    <div class="family-grid">
      <div><strong>Encoder</strong><span>bidirectional self-attention over the source</span></div>
      <div><strong>Decoder</strong><span>causal self-attention plus cross-attention to encoder states</span></div>
      <div><strong>Best fit</strong><span>translation, summarization, and other input-output sequence tasks</span></div>
      <div><strong>Examples</strong><span>T5 and BART keep separate input understanding and output generation</span></div>
    </div>
  </div>
</div>

---

<!-- .slide: id="encoder-only" -->

## Family 2: Encoder-Only

Bidirectional attention, no causal mask, no decoder. Trained with **masked language modeling** (forward reference to Module 5): some input tokens are hidden, and the model must predict them from their context on both sides.

$$\text{output}_i = f(\mathbf{x}_1, \dots, \mathbf{x}_n) \quad \text{for all } i$$

Use cases: classification, sentence embeddings, retrieval, named-entity recognition. The model produces representations, not generations.

BERT (Devlin et al., 2018) made this architecture dominant for "understanding" tasks. RoBERTa improved the training recipe without changing the architecture.

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

:::figure img="images/radford.jpg" name="Alec Radford" kicker="GPT and GPT-2 (2018-2019)"
- Showed that autoregressive pretraining could produce general-purpose text models
- GPT-2 demonstrated coherent long-form generation from scale and data
- The decoder-only stack became the dominant LLM design
:::

---

<!-- .slide: id="honest-tradeoffs" -->

## The Honest Trade-offs

Decoder-only did **not** win on every axis. It won on generality and scaling:

- **Encoder-only** is still more efficient for fixed-vector embeddings, retrieval, and classification. If you only need a sentence representation, BERT-style models are faster.

- **Encoder-decoder** remains strong where input and output are clearly distinct, as in machine translation. The input can be processed bidirectionally while the output is generated autoregressively.

- **Decoder-only** excels when the task is open-ended or when you want a single model to handle many tasks through prompting. The price is that every task must be phrased as text completion.

The choice depends on the problem, not on decoder-only being universally superior.
