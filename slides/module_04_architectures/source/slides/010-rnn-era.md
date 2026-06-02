:::divider id="divider-rnn" title="Before Transformers" sub="The recurrent era and its limitations"
:::

---

<!-- .slide: id="rnn-idea" -->

## Recurrent Neural Networks

<div class="rnn-flow">
  <div class="rnn-token">the</div>
  <div class="rnn-arrow">&rarr;</div>
  <div class="rnn-cell">RNN<br><span>state 1</span></div>
  <div class="rnn-arrow">&rarr;</div>
  <div class="rnn-token">cat</div>
  <div class="rnn-arrow">&rarr;</div>
  <div class="rnn-cell">same cell<br><span>state 2</span></div>
  <div class="rnn-arrow">&rarr;</div>
  <div class="rnn-token">sat</div>
  <div class="rnn-arrow">&rarr;</div>
  <div class="rnn-cell">same cell<br><span>state 3</span></div>
</div>

<div class="formula-card">new hidden state = function(previous hidden state, current token)</div>

The same cell is reused at every time step. The hidden state is a fixed-size summary, so long sequences force more and more history into one vector.

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

<div class="gate-diagram">
  <div class="gate-input">previous state<br>+ current token</div>
  <div class="gate-arrow">&rarr;</div>
  <div class="gate-stack">
    <div class="gate forget">forget gate<br><span>what to erase</span></div>
    <div class="gate input">input gate<br><span>what to write</span></div>
    <div class="gate output">output gate<br><span>what to expose</span></div>
  </div>
  <div class="gate-arrow">&rarr;</div>
  <div class="gate-memory">cell state<br><span>longer memory path</span></div>
</div>

LSTMs added gates so information could be selectively remembered, overwritten, or hidden. GRUs simplified the same idea into update and reset gates.

Both kept the **sequential and bottleneck limitations.** The cell was improved, but the architecture was still fundamentally serial.

---

<!-- .slide: id="figure-schmidhuber" class="notable-figure" -->

<div class="notable-stage">
  <img class="notable-photo notable-photo-center fragment fade-out" data-fragment-index="2" src="images/Schmidhuber.jpg" alt="Jürgen Schmidhuber">
  <h2 class="notable-name-first fragment fade-in-then-out" data-fragment-index="1">Jürgen Schmidhuber</h2>
  <div class="notable-reveal fragment fade-in" data-fragment-index="2">
    <img class="notable-photo notable-photo-side" src="images/Schmidhuber.jpg" alt="Jürgen Schmidhuber">
    <div class="notable-copy">
      <h2>Jürgen Schmidhuber</h2>
      <h3>Long Short-Term Memory (1997)</h3>
      <ul>
        <li>With Sepp Hochreiter, introduced gated recurrent cells for long-range dependencies</li>
        <li>Made gradient flow through long sequences much more reliable</li>
        <li>LSTMs became the default sequence model before transformers</li>
      </ul>
    </div>
  </div>
</div>

---

<!-- .slide: id="seq2seq" -->

## Sequence-to-Sequence Learning

Encoder-decoder RNNs (Sutskever et al., 2014) split the problem into two recurrent networks:

- **Encoder:** reads the input sequence and compresses it into a single fixed-size context vector
- **Decoder:** generates the output sequence, conditioned on that vector

This was the dominant architecture for machine translation. But the fixed-vector bottleneck is severe: a long source sentence must be compressed into the same-size vector as a short one.

Bahdanau attention (Module 3) was invented specifically to fix this bottleneck, letting the decoder attend directly to encoder states instead of relying on a single compressed vector.

---

<!-- .slide: id="figure-sutskever" class="notable-figure" -->

<div class="notable-stage">
  <img class="notable-photo notable-photo-center fragment fade-out" data-fragment-index="2" src="images/sutskever.jpg" alt="Ilya Sutskever">
  <h2 class="notable-name-first fragment fade-in-then-out" data-fragment-index="1">Ilya Sutskever</h2>
  <div class="notable-reveal fragment fade-in" data-fragment-index="2">
    <img class="notable-photo notable-photo-side" src="images/sutskever.jpg" alt="Ilya Sutskever">
    <div class="notable-copy">
      <h2>Ilya Sutskever</h2>
      <h3>Sequence to Sequence Learning (2014)</h3>
      <ul>
        <li>With Oriol Vinyals and Quoc Le, showed encoder-decoder RNNs could translate sequences</li>
        <li>The encoder compressed the source sentence into one vector</li>
        <li>That bottleneck motivated attention over encoder states</li>
      </ul>
    </div>
  </div>
</div>

---

<!-- .slide: id="figure-vaswani" class="notable-figure" -->

<div class="notable-stage">
  <img class="notable-photo notable-photo-center fragment fade-out" data-fragment-index="2" src="images/vaswani.webp" alt="Ashish Vaswani">
  <h2 class="notable-name-first fragment fade-in-then-out" data-fragment-index="1">Ashish Vaswani</h2>
  <div class="notable-reveal fragment fade-in" data-fragment-index="2">
    <img class="notable-photo notable-photo-side" src="images/vaswani.webp" alt="Ashish Vaswani">
    <div class="notable-copy">
      <h2>Ashish Vaswani</h2>
      <h3>Attention Is All You Need (2017)</h3>
      <ul>
        <li>Lead author on the original transformer paper</li>
        <li>Helped replace recurrence with stacked attention and feed-forward blocks</li>
        <li>The architecture became the foundation for modern LLMs</li>
      </ul>
    </div>
  </div>
</div>

---

<!-- .slide: id="attention-is-all-you-need" -->

## Attention Is All You Need (2017)

Vaswani et al. removed recurrence entirely. In its place: stacked multi-head self-attention with position-wise feed-forward networks.

Why this mattered:

- **Full parallelism across positions.** Every token can be processed at the same time during training; only the causal mask enforces order at inference time.

- **Constant path length.** In an RNN, information from token 1 must travel through every intermediate hidden state to reach token $n$. In a transformer, token 1 and token $n$ interact directly through one attention layer. The longest signal path is $O(1)$ layers, not $O(n)$ recurrent steps.

- **No fixed-size bottleneck.** Each layer accesses the full sequence through attention; nothing is compressed into a single vector.

The transformer is best understood as a direct response to RNN limitations. Every design choice trades sequential recurrence for parallel computation.
