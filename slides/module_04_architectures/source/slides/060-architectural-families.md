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

- Bidirectional attention, no causal mask, no decoder
- Trained with **masked language modeling** (Module 5): hide tokens, predict them from both sides

$$\text{output}_i = f(\mathbf{x}_1, \dots, \mathbf{x}_n) \quad \text{for all } i$$

- Produces representations, not generations: classification, embeddings, retrieval, named-entity recognition
- BERT (Devlin et al., 2018) made this dominant for "understanding" tasks; RoBERTa improved the recipe, same architecture

---

<!-- .slide: id="decoder-only" -->

## Family 3: Decoder-Only

- Causal masking, autoregressive next-token prediction
- GPT (Radford et al., 2018), GPT-2, GPT-3, and the Llama lineage share this design

$$\text{output}_t = f(\mathbf{x}_1, \dots, \mathbf{x}_t)$$

Why it won for LLMs:

- One stack, simple, scales cleanly
- Next-token objective: dense training signal at every position
- Any task phrases as text completion; no task-specific heads
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

Decoder-only won on generality and scaling, not every axis:

- **Encoder-only:** faster for embeddings, retrieval, classification
- **Encoder-decoder:** strong when input and output are distinct (translation); bidirectional input, autoregressive output
- **Decoder-only:** best for open-ended tasks and one-model-for-everything prompting; the price is phrasing every task as completion

Pick by problem, not by fashion.
