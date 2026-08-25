:::divider id="divider-quiz" title="Quiz"
:::

---

:::quiz id="quiz-softmax-price" title="What the Softmax Was For"
Removing the softmax lets attention be computed as a recurrence with a fixed-size state. What was the softmax buying, and which class of tasks exposes the loss?
+++
**Short answer: sharp selection. The tasks that expose it are the ones needing exact recall of a specific earlier token.**

- The exponential lets one key dominate: attention picks one token out of thousands
- A linear kernel spreads weight across many keys: a blur, not a lookup
- The fixed-size state cannot store every past token anyway
- Result: fine on perplexity, behind on copying and needle-in-a-haystack retrieval
:::

---

:::quiz id="quiz-kv-vs-state" title="Two Kinds of Memory Cost"
A serving system stores a KV cache that grows with every generated token. What is the corresponding cost for a state-space model, and what does that mean for very long conversations?
+++
**Short answer: constant. The state is a fixed-size matrix, so memory does not grow with the conversation at all.**

- KV cache: one key and value per token per layer, growing linearly forever
- Recurrent state: $\mathbf S$ and $\mathbf z$, sized by head dimension, not sequence length
- A million-token conversation costs the same memory as a ten-token one
- The catch: constant memory is *why* exact recall is impossible
- Hybrids pay the cache only on the few layers where random access earns its cost
:::

---

:::quiz id="quiz-selectivity" title="What Mamba Rebuilt"
Mamba's contribution is usually described as "selective" state transitions. Which earlier architecture's mechanism is that a rebuild of, and what did Mamba have to preserve that the original did not?
+++
**Short answer: LSTM gating. Mamba had to preserve parallel training, which gating originally destroyed.**

- LSTM forget and input gates decide what enters and leaves the state, conditioned on the input. That is "selective"
- The idea was never wrong; it was slow. Gating made training sequential, unable to parallelize across positions
- Mamba's contribution is engineering: an associative scan keeps the recurrence parallel during training, and the state stays in fast GPU memory
- The version that won is the one that fits the hardware
:::

---

:::quiz id="quiz-left-to-right" title="Where the Assumption Lives"
Left-to-right generation was called a choice rather than a fact. Which two pieces of machinery encode that choice?
+++
**Short answer: the chain-rule factorization of the joint probability, and the causal mask that enforces it in the architecture.**

- The factorization $p(x_1,\dots,x_T) = \prod_t p(x_t \mid x_{<t})$ is exactly true for *any* ordering
- Left to right makes the likelihood exactly computable in one pass, so training is cheap and stable
- The causal mask enforces it: position $t$ sees nothing after it, so one forward pass predicts every position
- A design decision for convenience, not a claim about how language works
:::

---

:::quiz id="quiz-infilling" title="Why the Model Cannot Fill in a Blank"
A diffusion language model can fill in a blank in the middle of a document. A standard autoregressive model cannot. What component is responsible for that inability?
+++
**Short answer: the causal mask.**

- Filling a blank requires conditioning on *both* sides
- The mask forbids a position from attending to anything after it
- Prompting cannot fix it: the constraint is enforced in the attention pattern, not learned
- Diffusion has no mask: every position attends to every other at every denoising step, so infilling is the default behavior
:::

---

:::quiz id="quiz-diffusion-adoption" title="Why Diffusion Adoption Is Slow"
Diffusion language models generate many tokens per forward pass, yet adoption has been slow. Explain the obstacle in terms of the serving stack rather than model quality.
+++
**Short answer: the KV cache exists because past tokens never change, and under diffusion they do.**

- Every serving optimization assumes causality: caching, continuous batching, streaming, prefix caching
- Diffusion revises the whole sequence every step, so none of those hold
- A successor must beat the incumbent's loss curve *plus* a decade of infrastructure built on assumptions it violates
- Block diffusion (autoregressive across blocks, diffusive within) exists mainly so KV caching survives
:::

---

:::quiz id="quiz-hybrids" title="A Pattern Across Two Sections"
Both the architecture section and the diffusion section ended with hybrids winning. What do the two cases have in common?
+++
**Short answer: in both, the challenger is better on one axis and strictly worse on another, so the practical answer is to pay for each property only where it earns its cost.**

- Recurrence buys constant memory, loses exact recall: hybrids keep a few attention layers for random access
- Diffusion buys parallel generation, loses KV caching: block diffusion keeps autoregression across blocks
- Neither challenger dominates; each trades
- Successors get absorbed as components where their tradeoff is favorable. Expect the next architecture to arrive as a layer type, not a replacement
:::

---

:::quiz id="quiz-fixed-depth" title="Two Ways to Think Harder"
A fixed-depth transformer spends the same computation on the word "the" as on the last step of a hard proof. Describe the two ways the lecture gave for spending more on the hard one, and what each costs.
+++
**Short answer: emit more tokens (chain of thought), or loop the layers (recurrent depth). The first costs tokens and latency; the second costs supervisability and infrastructure.**

- Chain of thought: write intermediate steps and read them back. Costs latency and billing for every word
- Looped layers: more computation without emitting anything; depth becomes a runtime dial
- Looping breaks every serving assumption about fixed work per token
- Bigger cost: you cannot reward a reasoning process you cannot read. RL needs a transcript to score
:::

---

:::quiz id="quiz-compute-equivalent" title="Reading a Result Carefully"
A paper reports that a 3.5B recurrent-depth model reaches the compute load of a 50B model. Why is "this 3.5B model matches a 50B model" the wrong summary?
+++
**Short answer: the claim is about compute-equivalence, not performance-equivalence. Matching the computational load of a 50B model is not the same as matching what a well-trained 50B dense model can do.**

- Unrolling depth means the small model performs as many operations per answer as a larger one
- Real result: the compute of a large model from a small memory footprint
- But a 50B dense model also has 50B parameters of stored knowledge. Looping cannot give a 3.5B model that
- Coverage of this field converts compute-equivalence into performance-equivalence constantly
:::

---

:::quiz id="quiz-state-gradient" title="What the State Update Really Is"
The exercise's state update is `S = S + phi(k) v^T`. What is that update actually performing, and what does it imply about linear-attention models?
+++
**Short answer: a single step of gradient descent, on a squared reconstruction loss for a linear map from keys to values.**

- The outer product is exactly the gradient for a linear model predicting $\mathbf v$ from $\phi(\mathbf k)$
- The hidden state is not a buffer. It is the weight matrix of a tiny model, one training step per token
- Every linear-attention model has been doing test-time training all along
- The generalization: bigger inner model, more steps, better optimizer. That is test-time training
:::

---

:::quiz id="quiz-nested-loops" title="Naming the Two Loops"
Test-time training puts a training loop inside the forward pass. Name the two nested loops, say what each learns, and identify which one you already know.
+++
**Short answer: the outer loop is ordinary pretraining, and the inner loop runs at inference on one sequence. The outer loop learns how to learn; the inner loop learns the document in front of it.**

- Outer loop: the one you know. Gradient descent over a corpus, slow, done once
- Its objective changes: learn weights that make the inner loop effective
- Inner loop: runs during inference, updating a small model that serves as the hidden state, trained on the context
- The proposal: make the inner loop a real optimizer, not a hand-designed rule that secretly already was one
:::

---

:::quiz id="quiz-continual-eval" title="The Cost of Never Freezing"
Continual learning on a deployed model sounds like free improvement. What failure mode does it hit, and what happens to the evaluate-then-ship workflow?
+++
**Short answer: catastrophic forgetting. And evaluation stops being something you do once before shipping.**

- Catastrophic forgetting (McCloskey and Cohen, 1989): train on new data and old capabilities degrade
- Mitigations: replay, penalize movement in weights that mattered, isolate new learning in added parameters. None solved at this scale
- Evaluate-then-ship assumes the artifact you tested is the artifact you serve
- With changing weights, every eval result has an expiry date, and replicas trained on different traffic become different models
:::
