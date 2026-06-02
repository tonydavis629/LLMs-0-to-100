:::divider id="divider-rnn" title="Before Transformers" sub="The recurrent era and its limitations"
:::

---

<!-- .slide: id="rnn-idea" -->

## Recurrent Neural Networks

An RNN processes tokens one at a time, carrying a hidden state forward as a running summary of everything seen so far:

$$\mathbf{h}_t = f(\mathbf{h}_{t-1}, \mathbf{x}_t; W)$$

At every step the old hidden state and the new token are combined to produce a new hidden state. The same weights are reused at every time step, so the network has a fixed parameter count regardless of sequence length.

The hidden state is a fixed-size vector that must compress the entire history into a single point. Long sentences force that vector to carry information from dozens or hundreds of steps back.

---

<!-- .slide: id="rnn-problems" -->

## The Problems with Recurrence

Recurrent training is fundamentally hard for three reasons:

- **Vanishing and exploding gradients.** Gradients must survive repeated matrix multiplications as they travel backward through time. Over long sequences the product of many weight matrices either shrinks to near zero (vanishing) or blows up (exploding). This is the same backprop multiplication rule from Module 2, amplified across many recurrent steps.

- **Sequential dependency.** Token $t$ cannot be processed until tokens $1$ through $t-1$ are done. This makes training impossible to parallelize across the sequence: for a batch of $n$ tokens, the RNN takes $n$ serial steps instead of one parallel step.

- **Fixed-size bottleneck.** The hidden state has a fixed dimension. No matter how long the input, all context must be squeezed through one vector of the same size.

---

<!-- .slide: id="lstm-gru" -->

## LSTMs and GRUs: Gating Mechanisms

The Long Short-Term Memory network (1997) added **gating** to the recurrent cell so information could flow across many time steps without vanishing:

$$\mathbf{f}_t = \sigma(W_f \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t]) \quad \text{(forget gate)}$$
$$\mathbf{i}_t = \sigma(W_i \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t]) \quad \text{(input gate)}$$
$$\mathbf{o}_t = \sigma(W_o \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t]) \quad \text{(output gate)}$$

By selectively remembering and forgetting, LSTMs mitigated the gradient problem. GRUs (2014) simplified the gates to just two: update and reset.

Both kept the **sequential and bottleneck limitations.** The cell was improved, but the architecture was still fundamentally serial.

---

:::figure img="images/hochreiter_schmidhuber.jpg" name="Sepp Hochreiter & Jürgen Schmidhuber" kicker="Introduced Long Short-Term Memory (1997)"
- Schmidhuber and Hochreiter's LSTM added forget, input, and output gates to recurrent cells
- For the first time, gradients could propagate across hundreds of time steps without vanishing
- LSTMs became the default choice for sequence modeling from 1997 until the transformer era
- Hochreiter & Schmidhuber, "Long Short-Term Memory," *Neural Computation* (1997)
:::

---

<!-- .slide: id="seq2seq" -->

## Sequence-to-Sequence Learning

Encoder-decoder RNNs (Sutskever et al., 2014) split the problem into two recurrent networks:

- **Encoder:** reads the input sequence and compresses it into a single fixed-size context vector
- **Decoder:** generates the output sequence, conditioned on that vector

This was the dominant architecture for machine translation. But the fixed-vector bottleneck is severe: a long source sentence must be compressed into the same-size vector as a short one.

Bahdanau attention (Module 3) was invented specifically to fix this bottleneck, letting the decoder attend directly to encoder states instead of relying on a single compressed vector.

---

:::figure img="images/sutskever_vinyals_le.jpg" name="Ilya Sutskever, Oriol Vinyals & Quoc Le" kicker="Sequence to Sequence Learning with Neural Networks (2014)"
- Sutskever, Vinyals, and Le introduced the encoder-decoder RNN framework for neural machine translation
- The encoder compresses the entire input into one vector; the decoder generates from it
- This fixed-vector bottleneck is exactly what Bahdanau attention was invented to fix
- Sutskever et al., <https://arxiv.org/abs/1409.3215>
:::

---

<!-- .slide: id="attention-is-all-you-need" -->

## "Attention Is All You Need" (2017)

Vaswani et al. removed recurrence entirely. In its place: stacked multi-head self-attention with position-wise feed-forward networks.

Why this mattered:

- **Full parallelism across positions.** Every token can be processed at the same time during training; only the causal mask enforces order at inference time.

- **Constant path length.** In an RNN, information from token 1 must travel through every intermediate hidden state to reach token $n$. In a transformer, token 1 and token $n$ interact directly through one attention layer. The longest signal path is $O(1)$ layers, not $O(n)$ recurrent steps.

- **No fixed-size bottleneck.** Each layer accesses the full sequence through attention; nothing is compressed into a single vector.

The transformer is best understood as a direct response to RNN limitations. Every design choice trades sequential recurrence for parallel computation.
