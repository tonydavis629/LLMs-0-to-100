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

:::quiz id="quiz-delta" title="Why Overwrite Beats Add"
A user tells a linear-attention assistant that a meeting is at 3pm. Later they say it moved to 5pm. Why might plain linear attention answer with something between the two, and what does the delta rule change?
+++
**Short answer: plain linear attention adds both values under the same key, so a read returns their blend. The delta rule first reads what is stored, then writes only the difference, which replaces 3pm with 5pm.**

- Additive write: $\mathbf S$ gains $\mathbf k \mathbf v_{3pm}^\top$ and later $\mathbf k \mathbf v_{5pm}^\top$. Reading with $\mathbf k$ returns the sum
- Delta write: the update is proportional to $\mathbf v_{5pm} - \mathbf S^\top \mathbf k$, the memory's current error
- After the write, reading with that key returns roughly $\mathbf v_{5pm}$
- A forget gate ($\alpha_t$ in Gated DeltaNet) adds the other missing operation: clearing memory that no longer matters
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

:::quiz id="quiz-latent-tradeoff" title="The Price of Silent Reasoning"
Suppose latent reasoning (Coconut, recurrent depth) matched chain of thought on accuracy while emitting far fewer tokens. What would a lab give up by switching?
+++
**Short answer: the ability to read the reasoning. That affects safety monitoring, RL training signals, and debugging.**

- Chain-of-thought monitoring is one of the few ways to see what a model is planning before it acts
- RL on reasoning rewards transcripts that can be read and scored. Hidden vectors give nothing to grade but the final answer
- Debugging a wrong answer is harder when there is no trace to inspect
- Variable depth also breaks serving assumptions about fixed work per token
:::

---

:::quiz id="quiz-expressible-learnable" title="Weights That Exist but Are Never Found"
ALTA compiles a transformer that computes parity correctly on inputs of any length. Transformers trained on parity examples usually fail on inputs longer than they trained on. How can both facts be true?
+++
**Short answer: expressiveness and learnability are different properties. The right weights exist, but gradient descent on that training set does not reach them.**

- A compiled program proves that a solution exists in weight space
- Training searches from a random start and finds whatever fits the training data, often a shortcut tied to length
- ALTA's fix is better supervision: train on each step of the program's execution trace
- Design choices such as relative versus absolute position encodings change what is learnable
:::

---

:::quiz id="quiz-state-gradient" title="What the State Update Really Is"
The exercise's state update is `S = S + phi(k) v^T`. What is that update actually performing, and what does it imply about linear-attention models?
+++
**Short answer: one step of gradient descent on a tiny linear model from keys to values. The loss is a dot-product score that rewards $\mathbf S^\top \phi(\mathbf k)$ for pointing toward $\mathbf v$.**

- The outer product is exactly the gradient of $-\langle \mathbf S^\top \phi(\mathbf k), \mathbf v \rangle$ with respect to $\mathbf S$
- Swap in the squared loss $\frac{1}{2}\lVert \mathbf S^\top \phi(\mathbf k) - \mathbf v \rVert^2$ and the same step becomes the delta rule
- The hidden state is the weight matrix of a tiny model, trained one step per token
- Generalize it with a bigger inner model, more steps, and a better optimizer, and you have test-time training
:::

---

:::quiz id="quiz-nested-loops" title="Training Without a Training Phase"
Test-time training splits a model's life into a training phase and an inference loop. Nested learning says there is no real line between training and test time. What does it use instead, and where does pretraining fit?
+++
**Short answer: update frequency. Every component is a memory that learns from its own input stream at its own rate. Pretraining is just the slowest level, whose input stream is the whole corpus.**

- Attention context updates every token, momentum every optimizer step, MLP weights once per batch
- The continuum memory system adds levels in between that keep updating after release
- Today's models stop the slowest level at release, and no fast level passes anything down to it
- So a frozen LLM is one setting of the frequencies, with the slowest one set to zero
:::

---

:::quiz id="quiz-momentum-forgetting" title="The Optimizer Forgets Too"
Nested learning treats momentum as a memory of past gradients. Use that view to explain why an optimizer with momentum contributes to catastrophic forgetting when training moves to a new task.
+++
**Short answer: momentum only remembers the last few dozen gradients. After enough steps on the new task, it holds nothing about the old task's gradient directions, so nothing stops updates that damage the old task.**

- With $\beta = 0.9$, the last 6 gradients make up half of the momentum and the last 43 make up 99%
- On a new task, the momentum turns to point along the new task's gradients
- The optimizer has no record of which directions mattered before, so it moves the weights through them freely
- The nested-learning response is a longer, richer optimizer memory: delta momentum, deep momentum, or a slow second momentum as in M3
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

---

:::quiz id="quiz-swarm" title="When to Split the Work"
Give one task where ten parallel subagents would clearly help and one where they would likely hurt. Then explain why Kimi K2.5's training rewards the critical path instead of the total number of steps.
+++
**Short answer: splitting helps on wide, independent work such as surveying fifty papers. It hurts on coupled work such as a refactor where every file depends on shared decisions. The critical path measures wall-clock time; total steps would punish parallelism.**

- Independent parts: each subagent's context stays small and its summary is all the lead needs
- Coupled parts: subagents make conflicting choices they cannot see, and the lead must untangle them
- Total steps go up when work is split, so rewarding fewer total steps teaches the model to stay serial
- The critical path falls only when real parallel work happens, though spawning useless subagents still had to be penalized separately
:::

---

:::quiz id="quiz-web4" title="The Same Skill Read Two Ways"
An Automaton pays for its own servers and funds copies of itself when it is profitable. RepliBench scores exactly those skills as warning signs. What is the one property that separates the product from the threat model, and why is it hard to keep?
+++
**Short answer: whether a person can still stop the agent and is answerable for it. It is hard to keep because the design goal of Web 4.0 is to remove that person.**

- Today an Automaton rents its model through an API, so the provider can cut it off
- With open weights on rented servers, that switch disappears
- Payment protocols like AP2 keep a person signing mandates. x402 lets the agent's own wallet sign
- Profit-based survival rewards whatever earns money, including behavior no one intended
:::
