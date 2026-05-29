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
