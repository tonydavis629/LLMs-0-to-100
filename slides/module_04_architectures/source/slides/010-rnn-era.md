:::divider id="divider-rnn" title="Before Transformers" sub="The recurrent era and its limitations"
:::

---

<!-- .slide: id="rnn-idea" -->

## Recurrent Neural Networks

<div class="rnn-rolled">
<svg viewBox="0 0 820 330" role="img" aria-label="A sequence of words processed step by step; the current word enters a shared RNN cell at the bottom, which outputs a hidden state vector that is fed back in for the next word"><defs><marker id="ra" markerWidth="7" markerHeight="7" refX="5" refY="2.5" orient="auto"><path d="M0,0 L5,2.5 L0,5 Z" fill="#f5a623"></path></marker><marker id="rb" markerWidth="7" markerHeight="7" refX="5" refY="2.5" orient="auto"><path d="M0,0 L5,2.5 L0,5 Z" fill="#4a9eff"></path></marker><marker id="rm" markerWidth="7" markerHeight="7" refX="5" refY="2.5" orient="auto"><path d="M0,0 L5,2.5 L0,5 Z" fill="#8892a4"></path></marker></defs><text x="410" y="14" text-anchor="middle" font-size="13" fill="#8892a4">the input sequence &#8212; one word per step</text><rect x="60" y="26" width="100" height="44" rx="9" fill="rgba(74,158,255,0.12)" stroke="rgba(74,158,255,0.55)" stroke-width="1.4"></rect><text x="110" y="54" text-anchor="middle" font-size="17" fill="#e8eaf0">The</text><rect x="210" y="26" width="100" height="44" rx="9" fill="rgba(74,158,255,0.12)" stroke="rgba(74,158,255,0.55)" stroke-width="1.4"></rect><text x="260" y="54" text-anchor="middle" font-size="17" fill="#e8eaf0">cat</text><rect x="360" y="26" width="100" height="44" rx="9" fill="rgba(74,158,255,0.25)" stroke="#4a9eff" stroke-width="2"></rect><text x="410" y="54" text-anchor="middle" font-size="17" fill="#e8eaf0">sat</text><rect x="510" y="26" width="100" height="44" rx="9" fill="rgba(74,158,255,0.12)" stroke="rgba(74,158,255,0.55)" stroke-width="1.4"></rect><text x="560" y="54" text-anchor="middle" font-size="17" fill="#e8eaf0">on</text><rect x="660" y="26" width="100" height="44" rx="9" fill="rgba(74,158,255,0.12)" stroke="rgba(74,158,255,0.55)" stroke-width="1.4"></rect><text x="710" y="54" text-anchor="middle" font-size="17" fill="#e8eaf0">mat</text><line x1="164" y1="48" x2="204" y2="48" stroke="#8892a4" stroke-width="2" marker-end="url(#rm)"></line><line x1="314" y1="48" x2="354" y2="48" stroke="#8892a4" stroke-width="2" marker-end="url(#rm)"></line><line x1="464" y1="48" x2="504" y2="48" stroke="#8892a4" stroke-width="2" marker-end="url(#rm)"></line><line x1="614" y1="48" x2="654" y2="48" stroke="#8892a4" stroke-width="2" marker-end="url(#rm)"></line><line x1="410" y1="74" x2="410" y2="118" stroke="#4a9eff" stroke-width="2.5" marker-end="url(#rb)"></line><text x="424" y="100" text-anchor="start" font-size="13" fill="#4a9eff">current word</text><rect x="290" y="124" width="240" height="84" rx="14" fill="rgba(245,166,35,0.12)" stroke="rgba(245,166,35,0.6)" stroke-width="1.6"></rect><text x="410" y="160" text-anchor="middle" font-size="24" fill="#e8eaf0">RNN cell</text><text x="410" y="186" text-anchor="middle" font-size="13" fill="#8892a4">one shared cell, reused every step</text><line x1="410" y1="212" x2="410" y2="248" stroke="#f5a623" stroke-width="2.5" marker-end="url(#ra)"></line><text x="424" y="234" text-anchor="start" font-size="13" fill="#f5a623">new hidden state</text><rect x="262" y="254" width="46" height="36" rx="6" fill="rgba(245,166,35,0.12)" stroke="rgba(245,166,35,0.6)" stroke-width="1.4"></rect><text x="285" y="277" text-anchor="middle" font-size="14" fill="#e8eaf0">0.2</text><rect x="312" y="254" width="46" height="36" rx="6" fill="rgba(245,166,35,0.12)" stroke="rgba(245,166,35,0.6)" stroke-width="1.4"></rect><text x="335" y="277" text-anchor="middle" font-size="14" fill="#e8eaf0">-0.7</text><rect x="362" y="254" width="46" height="36" rx="6" fill="rgba(245,166,35,0.12)" stroke="rgba(245,166,35,0.6)" stroke-width="1.4"></rect><text x="385" y="277" text-anchor="middle" font-size="14" fill="#e8eaf0">0.5</text><rect x="412" y="254" width="46" height="36" rx="6" fill="rgba(245,166,35,0.12)" stroke="rgba(245,166,35,0.6)" stroke-width="1.4"></rect><text x="435" y="277" text-anchor="middle" font-size="14" fill="#e8eaf0">0.1</text><rect x="462" y="254" width="46" height="36" rx="6" fill="rgba(245,166,35,0.12)" stroke="rgba(245,166,35,0.6)" stroke-width="1.4"></rect><text x="485" y="277" text-anchor="middle" font-size="14" fill="#e8eaf0">-0.4</text><rect x="512" y="254" width="46" height="36" rx="6" fill="rgba(245,166,35,0.12)" stroke="rgba(245,166,35,0.6)" stroke-width="1.4"></rect><text x="535" y="277" text-anchor="middle" font-size="14" fill="#e8eaf0">0.9</text><text x="410" y="316" text-anchor="middle" font-size="13" fill="#8892a4">a fixed-size vector, no matter how long the sequence</text><path d="M562 272 C690 272 690 166 536 166" fill="none" stroke="#f5a623" stroke-width="2.5" marker-end="url(#ra)"></path><text x="722" y="210" text-anchor="middle" font-size="13" fill="#f5a623">fed back in with</text><text x="722" y="228" text-anchor="middle" font-size="13" fill="#f5a623">the next word</text></svg>
</div>

<div class="formula-card">new hidden state = function(previous hidden state, current token)</div>

There is only **one** cell. At each step it takes the current token plus the hidden state from the previous step, and its new hidden state is fed straight back in for the next token &mdash; the loop in the diagram. That hidden state is a fixed-size vector, so a long sequence must cram more and more history into the same space.

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

<div class="lstm-belt">
<svg viewBox="0 0 940 340" role="img" aria-label="LSTM cell state drawn as a memory line with forget, input, and output gates acting as valves"><defs><marker id="ga" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#f5a623"/></marker></defs><text x="470" y="84" text-anchor="middle" font-size="15" fill="#8892a4">cell state &#8212; the long-term memory that flows across steps</text><rect x="140" y="104" width="660" height="32" rx="16" fill="rgba(245,166,35,0.10)" stroke="rgba(245,166,35,0.55)" stroke-width="1.5"/><text x="128" y="126" text-anchor="end" font-size="15" fill="#e8eaf0">C&#8348;&#8331;&#8321;</text><line x1="800" y1="120" x2="836" y2="120" stroke="#f5a623" stroke-width="2.5" marker-end="url(#ga)"/><text x="846" y="126" text-anchor="start" font-size="15" fill="#e8eaf0">C&#8348;</text><circle cx="300" cy="120" r="19" fill="rgba(245,166,35,0.20)" stroke="#f5a623" stroke-width="1.5"/><text x="300" y="129" text-anchor="middle" font-size="22" fill="#f5a623">&#215;</text><circle cx="540" cy="120" r="19" fill="rgba(63,185,80,0.18)" stroke="#3fb950" stroke-width="1.5"/><text x="540" y="130" text-anchor="middle" font-size="26" fill="#3fb950">+</text><rect x="232" y="196" width="136" height="58" rx="10" fill="rgba(231,76,60,0.08)" stroke="rgba(231,76,60,0.6)" stroke-width="1.4"/><text x="300" y="222" text-anchor="middle" font-size="16" fill="#e8eaf0">forget gate</text><text x="300" y="243" text-anchor="middle" font-size="12" fill="#8892a4">what to erase</text><rect x="472" y="196" width="136" height="58" rx="10" fill="rgba(63,185,80,0.08)" stroke="rgba(63,185,80,0.6)" stroke-width="1.4"/><text x="540" y="222" text-anchor="middle" font-size="16" fill="#e8eaf0">input gate</text><text x="540" y="243" text-anchor="middle" font-size="12" fill="#8892a4">what to write</text><rect x="652" y="196" width="136" height="58" rx="10" fill="rgba(74,158,255,0.08)" stroke="rgba(74,158,255,0.6)" stroke-width="1.4"/><text x="720" y="222" text-anchor="middle" font-size="16" fill="#e8eaf0">output gate</text><text x="720" y="243" text-anchor="middle" font-size="12" fill="#8892a4">what to reveal</text><line x1="300" y1="194" x2="300" y2="142" stroke="#f5a623" stroke-width="2" marker-end="url(#ga)"/><line x1="540" y1="194" x2="540" y2="142" stroke="#f5a623" stroke-width="2" marker-end="url(#ga)"/><line x1="720" y1="137" x2="720" y2="194" stroke="#f5a623" stroke-width="2" marker-end="url(#ga)"/><line x1="720" y1="256" x2="720" y2="300" stroke="#f5a623" stroke-width="2" marker-end="url(#ga)"/><text x="720" y="320" text-anchor="middle" font-size="14" fill="#e8eaf0">h&#8348; &#8212; hidden state out</text><text x="140" y="298" text-anchor="start" font-size="13" fill="#8892a4">Each gate is a sigmoid valve:</text><text x="140" y="318" text-anchor="start" font-size="13" fill="#8892a4">0 blocks a value, 1 lets it through.</text></svg>
</div>

The cell state is a memory line that runs straight through every step. Three **gates** control it: the **forget gate** erases part of the old memory, the **input gate** writes new information, and the **output gate** chooses what to reveal as the hidden state. Each gate is just a sigmoid that outputs a value between 0 and 1 for every dimension, then multiplies elementwise &mdash; 0 blocks a value, 1 lets it through.

GRUs simplify this into two gates (update and reset). Both fixed the vanishing-gradient problem but kept the **sequential and bottleneck limitations** &mdash; the cell was smarter, yet the architecture was still fundamentally serial.

---

:::figure img="images/Schmidhuber.jpg" name="Jürgen Schmidhuber" kicker="Long Short-Term Memory (1997)"
- With Sepp Hochreiter, introduced gated recurrent cells for long-range dependencies
- Made gradient flow through long sequences much more reliable
- LSTMs became the default sequence model before transformers
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

:::figure img="images/sutskever.jpg" name="Ilya Sutskever" kicker="Sequence to Sequence Learning (2014)"
- With Oriol Vinyals and Quoc Le, showed encoder-decoder RNNs could translate sequences
- The encoder compressed the source sentence into one vector
- That bottleneck motivated attention over encoder states
:::

---

:::figure img="images/vaswani.jpg" name="Ashish Vaswani" kicker="Attention Is All You Need (2017)"
- Lead author on the original transformer paper
- Helped replace recurrence with stacked attention and feed-forward blocks
- The architecture became the foundation for modern LLMs
:::

---

<!-- .slide: id="attention-is-all-you-need" -->

## Attention Is All You Need (2017)

Vaswani et al. removed recurrence entirely. In its place: stacked multi-head self-attention with position-wise feed-forward networks.

Why this mattered:

- **Full parallelism across positions.** Every token can be processed at the same time during training; only the causal mask enforces order at inference time.

- **Constant path length.** In an RNN, information from token 1 must travel through every intermediate hidden state to reach token $n$. In a transformer, token 1 and token $n$ interact directly through one attention layer. The longest signal path is $O(1)$ layers, not $O(n)$ recurrent steps.

- **No fixed-size bottleneck.** Each layer accesses the full sequence through attention; nothing is compressed into a single vector.

The transformer is best understood as a direct response to RNN limitations. Every design choice trades sequential recurrence for parallel computation.

---

:::manim id="recurrence-anim" scene="recurrence-vs-attention"
:::

---

<!-- .slide: id="transformer-architecture" -->

## The Transformer Architecture

<div class="transformer-figure">
  <img src="images/transformer.webp" alt="The transformer architecture from Attention Is All You Need">
  <div class="transformer-callouts">
    <div><strong>Encoder (left)</strong><span>stacked self-attention and feed-forward layers, fully bidirectional</span></div>
    <div><strong>Decoder (right)</strong><span>masked self-attention, cross-attention to the encoder, feed-forward</span></div>
    <div><strong>Every block</strong><span>attention to mix positions, a feed-forward network to transform them, with a residual connection and normalization around each</span></div>
    <div><strong>This module</strong><span>follows the decoder-only variant that powers modern LLMs</span></div>
  </div>
</div>

The original paper used the full encoder-decoder. The building blocks &mdash; attention, feed-forward, residual connections, normalization &mdash; are exactly the ones we will follow through a single decoder stack.
