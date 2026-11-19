# Module 12: The Future of LLMs. Lecture Notes

These notes give an explanation and a citation for every major claim on the
slides, map the equations to the visuals they appear on, and record the
historical context. Module 12 is the last class and it runs in two halves. The
first is a review of the whole course, on the grounds that the material only
becomes a stack once students can see all of it at once. The second asks
what unit a model should predict (section a2), then whether attention is the right primitive at all (section b), then questions
three assumptions the course presented as facts when each was a design decision
(sections c, d, and e). Two further sections ask whether networks can run
programs exactly (section d2), where agents go after the single-model loop
(section e2), and what changes when agents pay their own way (section e3).

One caveat belongs at the top, and it is repeated on the slides. This is the one
module of the course whose content will age. Everything in sections b through e
is an active research direction with real disagreement among competent people.
The goal is to show students how a claim about the future of the field is
structured, and what evidence would bear on it.

## Section a: The course, reviewed

The review occupies roughly the first third of the class and introduces no new
material. Its purpose is structural: students have received one layer at a time
for eleven modules and have never seen the assembled stack.

**The stack slide.** Each row states what its module added, in the order the
course built them. The claim that ties them together is that every module was a
different answer to "what should the next token be," and that RL, retrieval, and
agents are all machinery wrapped around that single question.

**The five keepers.** Selected for how long they will stay true, not for how
important they were to the course:

- Cross-entropy is average surprise. For a distribution $p$ and model $q$, the
  loss $-\sum_x p(x)\log q(x)$ is measured in bits, and Module 1 established it
  as the expected number of bits needed to encode a sample. Every loss curve in
  the course was a bit count.
- Gradient descent is the only learning algorithm the course used. Modules 2
  through 7 differ in what they differentiate, not in how they optimize.
- Attention is $\mathrm{softmax}(QK^\top/\sqrt{d_k})V$. This appears on the
  slide exactly as it did in Module 3, and returns in section b when the softmax
  is removed.
- Loss falls predictably with compute (Kaplan et al., 2020, arXiv:2001.08361).
- You cannot improve what you cannot measure (Module 9).

Four of the five come from the first half of the course, which is what the
half-life slide that follows is about.

**The half-life sort.** This is an editorial judgment rather than a citable
result, and should be presented as such. The ordering is: the mathematics of
Modules 1 and 2 is permanent; the transformer is dominant but genuinely
contested (which section b then substantiates); the training recipe of Modules 5
through 7 is converging while still moving; and the Module 11 application layer
turns over yearly.

**Scaling, updated.** This pays the forward reference made in Module 5g. Three
claims, each with a source:

- Compute-optimal training balances parameters against tokens, at roughly 20
  tokens per parameter (Hoffmann et al., 2022, arXiv:2203.15556).
- The stock of usable public human text is finite, with exhaustion estimated in
  roughly this decade (Villalobos et al., 2022, arXiv:2211.04325). The slide
  says "roughly this decade" rather than naming a year, because the published
  range is wide and moves with each revision. When teaching, say that the exact
  year is contested and the direction is not.
- Training on model-generated data degrades the tails of the distribution
  (Shumailov et al., 2023, arXiv:2305.17493), which is the caveat on synthetic
  data as a response. Module 6's side quest introduced this as model collapse.
- The axis that actually moved is inference-time compute (Snell et al., 2024,
  arXiv:2408.03314). This is the setup for section d, and it is also where
  Module 7's forward reference about reasoning models is discharged.

The slides do not claim that scaling laws have broken, because no such
measurement exists. The narrower claim is that the cheap fuel of more web text
into a bigger transformer is spent, and that the frontier now scales along
several axes at once.

**Side quest: transformers are graph neural networks.** Attention is message
passing on a fully connected graph, where tokens are nodes, attention weights
are soft edges, and the value aggregation is the message step. Graph neural
networks restrict the edge set using a supplied graph; attention learns the
edges per input, which is why it needs no graph and why it costs $O(n^2)$. The
accessible reference is Joshi, "Transformers are Graph Neural Networks" (The
Gradient, 2020). The payoff for students is transfer: the same machinery applies
to image patches (Module 8), molecules, and code graphs.

## Section a2: What should a model predict?

**The unit problem ("Every Model in This Course Predicted Tokens").** Module 4
introduced tokenization as a tradeoff between bytes (a 256-symbol vocabulary but
very long sequences) and words (meaningful units but a huge vocabulary and
out-of-vocabulary failures), with byte-pair encoding (Sennrich, Haddow, and
Birch, 2016, arXiv:1508.07909) as the compromise. The right-hand column lists
costs that motivate the work below. The vocabulary is fixed before training.
Character-level tasks such as spelling and letter counting are hard because
characters are hidden inside multi-character tokens; Pagnoni et al. (2024) cite
this as a motivation for byte-level models. Uniform compute per token returns
in section d.

**Concept models ("Concept Models Predict Bigger Units Than Tokens" and "Concept
Models Plan by the Sentence").** The Byte Latent Transformer (Pagnoni et al.,
Meta, 2024, arXiv:2412.09871) groups bytes into patches whose boundaries are
placed where a small byte-level model's next-byte entropy is high, so hard text
gets short patches. CALM (Shao et al., 2025, arXiv:2510.27688) trains an
autoencoder that compresses $K$ tokens into one continuous vector, reconstructs
them with over 99.9% accuracy, and generates one vector per step, reducing the
number of steps by a factor of $K$; the slide uses $K = 4$. The Large Concept
Model (LCM team, Meta, 2024, arXiv:2412.08821) treats a sentence as a concept,
embeds each sentence with the frozen SONAR encoder (1024 dimensions, around 200
languages in text and speech), trains a transformer to predict the next sentence
embedding with MSE regression or diffusion, and decodes with the frozen SONAR
decoder. Its main experiments use 1.6B-parameter models trained on about 1.3T
tokens. SONAR-LLM (2025, arXiv:2508.05305) keeps prediction in SONAR space but
backpropagates a token-level cross-entropy loss through the frozen decoder. The
segmentations in the five-row figure are illustrative. "A ten-step plan is ten
predictions" is our gloss on the motivation, and the link to LeCun points ahead to the
section c side quest.

## Section b: Is attention the right primitive?

This section discharges the promise made in Module 4, which introduced
sub-quadratic architectures in one slide and deferred the detail here.

**The two costs.** Attention is $O(n^2)$ in sequence length because the score
matrix is $n \times n$ (Module 3), and Module 10's KV cache grows linearly with
generated tokens and comes to dominate serving memory at long context.
FlashAttention (Dao et al., 2022) reduced the constant factor and the memory
traffic without changing the exponent, which is worth stating precisely so
students do not conclude the problem was solved.

**The central equation**, which appears on the "Drop the softmax" slide and is
the exercise:

$$\mathbf y_t = \frac{\phi(\mathbf q_t)^\top \sum_{i \le t} \phi(\mathbf k_i) \mathbf v_i^\top}{\phi(\mathbf q_t)^\top \sum_{i \le t} \phi(\mathbf k_i)}$$

Softmax attention computes $\mathrm{softmax}(\mathbf q_t^\top \mathbf k_i)$,
where the exponential does not factor across $i$, so the full score matrix must
be materialized. Replacing $\exp(\mathbf q^\top \mathbf k)$ with a product of
feature maps $\phi(\mathbf q)^\top \phi(\mathbf k)$ makes the sum factor, and
associativity then permits two evaluation orders:

- Group as $(\phi(Q)\phi(K)^\top)V$: build the $n \times n$ matrix. Cost
  $O(n^2 d)$.
- Group as $\phi(Q)(\phi(K)^\top V)$: accumulate the running state
  $\mathbf S_t = \mathbf S_{t-1} + \phi(\mathbf k_t)\mathbf v_t^\top$, a
  $d \times d_v$ matrix. Cost $O(n d^2)$ in time and $O(d^2)$ in memory.

The two are algebraically identical; the difference is only where the
parentheses go. That is the content of Katharopoulos et al., "Transformers are
RNNs" (2020, arXiv:2006.16236), and the exercise has students verify it
numerically. The denominator is the normalizer that softmax otherwise provides
for free.

**What the softmax bought.** Students often assume linear attention approximates
softmax attention. It computes a different function, and two things go missing.
First, sharpness: the exponential lets
one key dominate the weighted sum, while a linear kernel spreads weight. Second,
the recurrent form's state is fixed-size, so it cannot reproduce an arbitrary
earlier token, whereas a KV cache can. The exercise makes the first point
measurable through the attention-entropy extra credit and prints the second as a
large numerical gap between linear and softmax outputs.

**Regrouping the sum ("The Running Sum Is Attention Regrouped").** The
numerator of linear attention is $\sum_{i \le t} (\phi(\mathbf q_t)^\top
\phi(\mathbf k_i)) \mathbf v_i^\top$, a similarity-weighted sum of values, as in
softmax attention. Because the score is a dot product, the query factors out of
the sum: $\phi(\mathbf q_t)^\top \sum_{i \le t} \phi(\mathbf k_i)\mathbf
v_i^\top = \phi(\mathbf q_t)^\top \mathbf S_t$. The denominator factors the same
way with $\mathbf z_t = \sum_{i \le t} \phi(\mathbf k_i)$. This is the per-token
version of the associativity slide before it (Katharopoulos et al., 2020,
Section 3.2).

**The state as a key-value memory ("The State Works Like a Lookup Table").** The
one-hot city, color, and time example is illustrative: with one-hot keys $\mathbf
e_i$, the write $\mathbf e_i \mathbf v^\top$ places $\mathbf v$ in row $i$ and the
read $\mathbf e_i^\top \mathbf S$ returns row $i$. The
running sum $\mathbf S_t = \sum_{i \le t} \phi(\mathbf k_i)\mathbf v_i^\top$ is
a $d \times d_v$ matrix, and reading it with $\phi(\mathbf q)^\top \mathbf S$
returns a weighted mix of stored values. If the stored keys were orthonormal and
the query equaled one of them, the read would return exactly that key's value.
In $d$ dimensions at most $d$ keys can be mutually orthogonal, which is the
source of the "about $d$ clean pairs" capacity claim on the slide. Schlag, Irie,
and Schmidhuber (2021, arXiv:2102.11174) make this argument and connect linear
attention to Schmidhuber's fast weight programmers (1992). Past capacity, keys
overlap and reads return blends.

**Five write rules ("Five Rules for Writing to the State").** The slide drops
the feature map for readability. Each row is a published model:

- Linear attention: $\mathbf S_t = \mathbf S_{t-1} + \mathbf k_t \mathbf v_t^\top$ (Katharopoulos et al., 2020).
- RetNet: $\mathbf S_t = \gamma \mathbf S_{t-1} + \mathbf k_t \mathbf v_t^\top$ with a fixed per-head decay $\gamma$ (Sun et al., 2023, arXiv:2307.08621).
- Data-dependent decay: $\mathbf S_t = \alpha_t \mathbf S_{t-1} + \mathbf k_t \mathbf v_t^\top$. Mamba-2 uses a scalar $\alpha_t$ per head (Dao and Gu, 2024, arXiv:2405.21060); GLA uses a vector gate per key dimension (Yang et al., 2023, arXiv:2312.06635).
- DeltaNet: $\mathbf S_t = \mathbf S_{t-1} + \beta_t \mathbf k_t(\mathbf v_t - \mathbf S_{t-1}^\top \mathbf k_t)^\top$. The rule is from Schlag et al. (2021); Yang et al. (2024, arXiv:2406.06484) made it trainable in parallel over the sequence.
- Gated DeltaNet combines the decay gate with the delta write (Yang, Kautz, and Hatamizadeh, 2024, arXiv:2412.06464). Qwen3-Next uses it in three of every four layers; Kimi Linear's KDA refines it with a per-channel gate (Kimi Team, 2025, arXiv:2510.26692).

**The delta rule ("The Delta Rule Overwrites a Row" and "The Delta Rule Is a
Gradient Step").** In the lookup-table picture, the delta write reads the
current row $\mathbf S^\top \mathbf k_t$, forms the error $\mathbf v_t -
\mathbf S^\top \mathbf k_t$, and adds $\beta_t$ times that error; with a one-hot
key and $\beta_t = 1$ the row becomes exactly $\mathbf v_t$. For a key not yet
stored, the read is near zero and the update reduces to an ordinary additive
write. The derivation:
Take the loss $\ell(\mathbf S) = \tfrac12 \lVert \mathbf S^\top \mathbf k_t -
\mathbf v_t \rVert^2$. Its gradient is $\nabla_{\mathbf S}\ell = \mathbf k_t
(\mathbf S^\top \mathbf k_t - \mathbf v_t)^\top$, so one step of gradient
descent with rate $\beta_t$ gives the DeltaNet update exactly. Plain linear
attention is also one gradient step, on the dot-product loss $-\langle
\mathbf S^\top \mathbf k_t, \mathbf v_t \rangle$, whose gradient is $-\mathbf
k_t \mathbf v_t^\top$ (Behrouz et al., 2025, Eq. 17). The difference matters:
the dot-product loss is minimized by making the read ever larger in the
direction of $\mathbf v_t$, so values pile up, while the squared loss is
minimized when the read equals $\mathbf v_t$, so the old value is replaced. The
Paris and Rome example on the slide is an illustration, not a result from a
paper. This slide is the bridge to section e: once the state is a model trained
by gradient descent, test-time training is the natural generalization.

**State-space models.** S4 (Gu et al., 2021, arXiv:2111.00396) derives a
sequence model from a continuous-time linear dynamical system discretized over
the sequence. Mamba (Gu and Dao, 2023, arXiv:2312.00752) makes the state
transition parameters input-dependent, which the paper calls selectivity. The
framing used on the slide is that selectivity is the LSTM's gating idea from
Module 4, rebuilt so that training still parallelizes. That parallelization is
the actual contribution: gating destroys the linear-recurrence structure that
allows a parallel scan, and Mamba recovers it with a hardware-aware selective
scan that keeps the state in fast GPU memory. Teaching note: the paper's own
emphasis on the GPU memory hierarchy is what makes the hardware lottery side
quest in section d land.

**S4, step by step (four slides from "S4 Starts From a Differential Equation"
to "Mamba Makes the Step Size Depend on the Token").** S4 abbreviates
"Structured State Space Sequence model," from Gu, Goel, and Ré, "Efficiently
Modeling Long Sequences with Structured State Spaces" (ICLR 2022,
arXiv:2111.00396). "Structured" refers to the special form of $A$ that makes
the model both expressive and fast. The block diagram is the standard
control-theory picture of a state-space system. The continuous system is

$$h'(t) = A h(t) + B x(t), \qquad y(t) = C h(t),$$

applied independently to each channel of the input. $A$ is $N \times N$ with
$N$ around 64. S4's first contribution is the choice of $A$: the HiPPO matrices
(Gu et al., 2020, arXiv:2008.07669) make $h(t)$ the coefficients of an optimal
polynomial approximation of the input history, so the state is a principled
compressed summary of everything seen so far. The second contribution is a
structured (diagonal plus low-rank) parameterization that makes the next step
cheap.

Discretizing with step $\Delta$ by zero-order hold gives $\bar A = \exp(\Delta
A)$ and $\bar B = (\Delta A)^{-1}(\exp(\Delta A) - I)\Delta B$, which is
approximately $\Delta B$ for small $\Delta$. The slide shows the simplified form
and motivates it with a forward Euler step, $h_t \approx h_{t-1} + \Delta(A
h_{t-1} + B x_t)$, which gives $\bar A \approx I + \Delta A$; $\exp(\Delta A)$ is
the exact solution over the step when $x$ is held constant. In the scalar example
($A = -1$, $\Delta = 0.1$, $B = 1$), $\bar A = e^{-0.1} \approx 0.905$ and the
exact $\bar B = 1 - e^{-0.1} \approx 0.095$; the slide rounds both.
The result is the linear recurrence $h_t = \bar A h_{t-1} + \bar B x_t$, $y_t =
C h_t$, with no nonlinearity between steps. Unrolling it gives

$$y_t = \sum_{j \ge 0} C \bar A^j \bar B \, x_{t-j} = (K * x)_t, \qquad K = (C\bar B, C\bar A\bar B, C\bar A^2\bar B, \dots),$$

a convolution whose kernel $K$ (the list of weights $K_j = C\bar A^j\bar B$,
one per distance $j$ into the past) is computed once and applied with an FFT in $O(n \log
n)$. This works only because $\bar A$, $\bar B$, and $C$ are the same at every
step (a linear time-invariant system). The kernel bar chart on the slide is
schematic: it shows weights decaying with distance, which is typical but not
required, since HiPPO kernels can oscillate.

Mamba makes $\Delta_t$, $B_t$, and $C_t$ functions of the current input $x_t$
(linear projections, with a softplus on $\Delta$). Because $A$ is negative
(its eigenvalues have negative real part), $\bar A_t = \exp(\Delta_t A)$ is near
0 when $\Delta_t$ is large, which resets the state and lets the input in
strongly, and near 1 when $\Delta_t$ is small, which preserves the state and
lets the input in weakly. Gu and Dao (2023) discuss this reading of $\Delta$ as
a generalized gate in section 3.5 of the Mamba paper. The examples on the slide
("a name, a new topic" versus "filler, punctuation") are illustrative. Once the
parameters vary per token the system is no longer time-invariant, the
convolution view is lost, and Mamba trains with a parallel associative scan
implemented as a fused kernel that keeps the expanded state in on-chip SRAM.

**Mamba-2 as linear attention ("Mamba-2 Turns Out to Be Linear Attention").**
Dao and Gu (2024, arXiv:2405.21060) restrict $\bar A_t$ to a scalar times the
identity, $a_t I$. Writing the whole sequence at once, the output is $Y = (L
\circ C B^\top) X$ where $L_{ij} = a_i a_{i-1} \cdots a_{j+1}$ for $i \ge j$ and
0 otherwise. Compare masked linear attention, $Y = (M \circ Q K^\top) V$ with
$M$ the all-ones causal mask. The correspondence is $C \leftrightarrow Q$, $B
\leftrightarrow K$, $X \leftrightarrow V$, and the only difference is that $L$
decays with distance. The paper calls this structured state space duality. The
cell shading on the slide uses $a = 0.55$ for every step, purely for
visibility.

RWKV (Peng et al., 2023, arXiv:2305.13048) reaches a comparable trade
independently, training in parallel and running inference as an RNN. It is worth
one slide partly for the architecture and partly because it is a community
open-source project rather than a lab flagship.

**The empirical verdict.** Recurrent and state-space models are competitive on
language modeling perplexity at modest scale and fall behind on tasks requiring
exact recall, copying, and retrieval of a specific earlier token. This is the
predicted consequence of a fixed-size state, and it is why production systems
hybridize: Jamba (Lieber et al., 2024, arXiv:2403.19887) and Griffin (De et al.,
2024, arXiv:2402.19427) interleave a small number of attention layers among many
recurrent ones. State the direction confidently and the exact numbers cautiously, since the
benchmark picture moves.

**Recent hybrids ("Recent Models Settle Near Three to One").** Layer counts come
from the model cards and technical reports as summarized by Raschka ("Beyond
Standard LLMs," 2025, and the LLM architecture gallery). Jamba interleaves one
attention layer per eight (Lieber et al., 2024). Qwen3-Next 80B-A3B has 48
layers in twelve repeats of three Gated DeltaNet blocks and one gated attention
block. Kimi Linear has 20 KDA layers and 7 multi-head latent attention layers
(Kimi Team, 2025). Qwen3.5-397B-A17B keeps full attention in 15 of 60 layers.
MiniMax-Text-01 used full attention in 10 of 80 layers, while its successor
MiniMax-M2 uses full attention in all 62; MiniMax explained the decision in a
2025 post, "Why did MiniMax M2 end up as a full attention model?". The memory
cost is the reason the ratio matters: at 128K tokens in bf16 the KV cache for
MiniMax-M2 is about 31 GiB against roughly 3 GiB for Qwen3-Next (Raschka's
estimate). These numbers change with every release; teach the ratio and the
counterexample, not the table.

**Side quest: the Bitter Lesson.** Sutton's essay (2019) argues that, over
seventy years of AI research, general methods that scale with computation have
beaten methods built on human knowledge. The slide asks students to apply the
argument to this section's own architectures. The questions have no settled
answers. The one-in-four ratio is Qwen3-Next's (see "Recent Models
Settle Near Three to One"). The closing sentence is our reading of the essay's historical
examples, not a claim Sutton makes about architectures specifically.

## Section c: Assumption one, text is written left to right

**Locating the assumption.** The slide shows the factorization students met in
Module 5:

$$p(x_1, \dots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_{<t})$$

The pedagogical point is that the chain rule holds for *any* permutation of the
variables. Left-to-right was chosen because it makes the likelihood exactly
computable in a single forward pass, and Module 4's causal mask is what enforces
it architecturally, allowing one pass to yield a valid prediction at every
position simultaneously. Neither fact is a claim about language.

**The three costs.** Sequential decoding at one forward pass per token, which is
Module 10's memory-bound decode phase; no revision of committed tokens; and
structural inability to condition on text after a blank, which is why infilling
and editing need workarounds. Only the first is about speed.

**Discrete diffusion.** Continuous diffusion originates with Sohl-Dickstein et
al., "Deep Unsupervised Learning using Nonequilibrium Thermodynamics" (2015,
arXiv:1503.03585), which Module 8 met in the image setting. The discrete-state
adaptation is D3PM (Austin et al., 2021, arXiv:2107.03006). The result that made
discrete diffusion competitive with autoregression on language is the
score-entropy objective of Lou et al. (2023, arXiv:2310.16834).

The masked formulation shown on the slide is the one students can follow without
new machinery: the forward process masks a fraction of positions, the model
predicts all masked positions at once, and sampling unmasks the confident
predictions and re-masks the rest. Draw the connection explicitly. That training
objective is Module 5's denoising and span corruption; what makes it a generator
is iterating it from a fully masked sequence. The number of denoising steps is a
quality-speed dial applied to the whole sequence rather than to one token.

**Where it stands.** LLaDA (Nie et al., 2025, arXiv:2502.09992) demonstrated the
recipe at 8B parameters against comparable autoregressive baselines. Mercury
(Inception Labs) and Gemini Diffusion are commercial systems advertising large
throughput gains; those numbers are vendor-reported and should be presented as
such. No diffusion language model has been shown to match frontier autoregressive
models, so the honest summary is competitive at mid scale and unproven above it.

**The stack argument.** This is the most transferable idea in the section. The
KV cache exists because past tokens never change, and every Module 10
optimization inherits that assumption: prefix caching, continuous batching, and
token streaming all assume a settled past. Diffusion revises the whole sequence
each step, so none of them hold. A successor architecture therefore has to beat
the incumbent's loss curve plus the infrastructure built on assumptions it
violates. Block diffusion (Arriola et al., 2025, arXiv:2503.09573) is the
compromise: autoregressive across blocks, diffusive within them, which restores
KV caching between blocks. Point out that this is the second hybrid outcome in
two sections, and that the pattern (successors get absorbed as components rather
than replacing the incumbent) is worth more than either architecture.

**Side quest: is autoregression a dead end?** LeCun's position, set out in "A
Path Towards Autonomous Machine Intelligence" (2022), is that token-by-token
generation accumulates error with no world model to correct against, and that
prediction should happen in representation space instead. Run it as a debate in
the style of Module 5's emergence discussion. The two sharpest questions for
students: what observation would refute the claim, and do Module 7's reasoning
models, which spend tokens revising their own work, constitute the missing error
correction or merely postpone the objection?

## Section d: Assumption two, every token gets the same computation

**The assumption.** A transformer with $L$ layers spends exactly $L$ layers of
computation on every token, whether that token is "the" or the final step of a
proof. Depth is fixed at training time.

**The answer already shipped.** Module 7's reasoning models buy computation per
answer by emitting more tokens. The slide's framing is that this accepts a
strange constraint: to compute more, the model must serialize its thinking into
words and read them back through its own context window, paying for each in
latency, KV cache, and billing. Nothing about "more computation" requires it to
be verbal, and that gap is what the section explores.

**Looped transformers.** Universal Transformers (Dehghani et al., 2018,
arXiv:1807.03819) apply shared weights recurrently in depth with a learned
halting rule, which is Graves's Adaptive Computation Time (2016,
arXiv:1603.08983) carried from RNNs to transformers. Flag the date: this
predates GPT-3, and part of the lesson is that the frontier includes revived old
proposals.

**The expressivity argument.** A fixed-depth network computes a fixed-length
circuit, so a problem requiring more sequential steps than the network has
layers is out of reach regardless of width. Width buys parallel work, not more
sequential steps. Giannou et al., "Looped Transformers as Programmable
Computers" (2023, arXiv:2301.13196) is the constructive version: a looped
transformer can implement iterative algorithms by running the same parameters
for a variable number of steps. Present this as a statement about
expressiveness, not efficiency.

**Latent reasoning and the number on the slide.** Geiping et al. (2025,
arXiv:2502.05171) trained a 3.5B-parameter recurrent-depth model on 800B tokens
and showed that unrolling the recurrent block further at test time improves
reasoning benchmarks, in their words "up to a computation load equivalent to 50
billion parameters."

This is the one number in the module most likely to be misquoted, and the slide
deliberately spends space on the distinction. The claim is compute-equivalence:
the small model performs roughly as many operations per answer as a much larger
model would. It is not the claim that a 3.5B model matches the benchmark
performance of a well-trained 50B dense model, which would additionally require
50B parameters' worth of stored knowledge that looping does not provide. Teach
the distinction rather than the number; press coverage of this field converts
one into the other routinely.

Coconut (Hao et al., 2024, arXiv:2412.06769) makes the same move differently, by
feeding the final hidden state back as the next input embedding instead of
decoding it to a token, so the chain of thought becomes a chain of vectors. The
unifying idea across both: compute spent per answer and tokens emitted per
answer are separable quantities that this course has silently treated as one.

**Recurrent depth, drawn ("Recurrent Depth Unrolls One Block at Test Time").**
Geiping et al.'s model has three parts: a prelude that embeds the input once, a
recurrent core block applied $r$ times, and a coda that decodes. The core starts
from a randomly initialized state $s_0$, and the prelude's embedding $e$ is
injected again at every iteration, so the block can always see the input. During
training $r$ is sampled at random per step from a heavy-tailed distribution,
with backpropagation truncated to the last few iterations, which is what lets one
set of weights run at any depth at test time. The 50B figure is the
compute-equivalence claim discussed above.

**Coconut, drawn ("Coconut Thinks in Hidden States").** Hao et al. (2024,
arXiv:2412.06769) feed the last hidden state back as the next input embedding
between special begin-thought and end-thought tokens. Training uses a curriculum
over stages: at stage $k$ the first $k$ written reasoning steps of a
chain-of-thought training example are replaced by $k$ continuous thoughts. The
paper's analysis decodes continuous thoughts with the LM head and finds that one
thought can place probability on several possible next steps, which the authors
interpret as a breadth-first search over a reasoning graph. Results on GPT-2
scale models: Coconut beats chain of thought on ProsQA, a logical reasoning task
that needs planning over a graph, and trails it on GSM8K. The token labels in
the diagram ("so 12", "then 7") are illustrative.

**Tiny recursive models ("A Tiny Recursive Model Beats Large LLMs on ARC").**
HRM (Wang et al., 2025, arXiv:2506.21734) uses two recurrent modules, a fast
low-level one and a slow high-level one, at 27M parameters. TRM
(Jolicoeur-Martineau, 2025, arXiv:2510.04871) replaces them with a single
two-layer network of 7M parameters that keeps a question embedding $x$, an
answer $y$, and a latent $z$. Each improvement step updates $z \leftarrow
f(x, y, z)$ several times, then updates $y \leftarrow g(y, z)$, and up to 16
improvement steps run per puzzle. Reported test accuracy: 44.6% on ARC-AGI-1 and
7.8% on ARC-AGI-2, against HRM's 40.3% and 5.0%; the slide rounds to 45% and 8%.
The paper's comparison table lists DeepSeek R1, o3-mini, and Gemini 2.5 Pro
below these numbers. TRM won first place in the ARC Prize 2025 paper awards. The
caveat on the slide is essential: TRM is trained on each task family with heavy
augmentation, it is not a language model, and the ARC Prize technical report
notes that its ARC-AGI-2 score used more compute than the competition allows.
Students should read this as evidence that recursion buys depth cheaply, not
that 7M parameters beat frontier LLMs at reasoning in general.

**Monitorability ("Three Places to Put the Thinking").** Korbak et al. (2025,
arXiv:2507.11473), with authors from several AI labs and research groups, argue
that models which reason in natural language offer an opportunity to monitor
their reasoning for intent to misbehave, and that the opportunity is fragile:
it can be lost to training pressure on the chain of thought or to architectures
that reason in latent space. The three-box spectrum is our framing. "Probes can
partly decode it" for Coconut refers to the LM-head decoding analysis above.

**The width analogue.** Mixture of experts (Module 4) sparsifies compute per
token across width; Mixture-of-Depths (Raposo et al., 2024, arXiv:2404.02258)
does it across depth by letting tokens skip layers. Looping generalizes further
by allowing more steps than the network has layers.

**Why it did not ship.** Students otherwise conclude the field is being
irrational, so give both reasons. First, tooling: variable
work per token breaks batching, latency prediction, and per-request cost models
built on uniform work. Second, supervision: Module 7's RL machinery scores
transcripts, and latent reasoning produces no transcript. The property that makes latent reasoning efficient, that it never gets written
down, is the same one that makes it impossible to audit.

**Side quest: the hardware lottery.** Hooker (2020, arXiv:2009.06489) argues
that ideas win partly by fitting the hardware of their era. This is the answer to
the question the section raises, and it also retroactively explains section b
(Mamba's contribution is substantially an implementation) and section c (block
diffusion exists so diffusion can keep the KV cache).

## Section d2: Can a network run a program?

This section follows the Graves slide, which introduces Neural Turing Machines
as the second of his two early questions. It moves from differentiable memory to
the line of work that writes programs directly into transformer weights, then to
logic languages that sit inside training, and ends with the tool use students
already know from Module 11.

**The gap ("Networks Approximate Algorithms").** Transformers trained on
arithmetic examples generalize poorly to numbers longer than those seen in
training; this length-generalization failure is documented across many papers,
and Weng et al. (2023) and Shaw et al. (2024) both use it as motivation. The two
columns contrast learned and written rules in general terms and are our framing.

**Neural Turing Machines ("Neural Turing Machines Added a Memory Tape").**
Graves, Wayne, and Danihelka (2014, arXiv:1410.5401) attach a memory matrix to
a controller network. Read and write heads address memory rows with soft
weights computed from content similarity and location shifts, so every memory
operation is differentiable. The paper trains on copy, repeat copy, associative
recall, and priority sort. The Differentiable Neural Computer (Graves et al.,
2016, Nature 538:471) adds dynamic memory allocation and temporal links and
learns to answer questions about graphs, including shortest paths on the London
Underground. Neither line scaled to language modeling. The closing claim, that
transformers kept soft addressing and replaced the tape with the context window,
is an interpretive summary: content-based attention (Bahdanau et al., 2014)
developed in parallel with the NTM rather than after it.

**RASP ("RASP Is a Language for Transformers").** Weiss, Goldberg, and Yahav
(2021, arXiv:2106.06981) define the Restricted Access Sequence Processing
language. `select(keys, queries, predicate)` produces a boolean selection
matrix, the analogue of an attention pattern, and `aggregate(selector, values)`
averages values under it, the analogue of the attention output. Elementwise
maps correspond to MLPs. The reverse program on the slide follows the paper's
example: `length` is itself computed with one select-aggregate pair, and the flip
needs a second, so the program needs two layers. The paper uses the longest
chain of dependent select-aggregate operations to predict the number of layers
a trained transformer needs and finds that trained models match those
predictions.

**Compilers ("Compilers Turn Programs Into Weights").** Tracr (Lindner et al.,
2023, arXiv:2301.05062) compiles RASP programs to concrete transformer weights
and was built as a laboratory for interpretability: a model whose ground-truth
circuit is known. The three uses on the slide (interpretability test cases,
attaching exact skills, existence proofs) correspond to Tracr, CoNN, and ALTA
respectively.

**CoNN ("Compiled Modules Inside a Language Model").** Weng et al.,
"Mastering Symbolic Operations: Augmenting Language Models with Compiled Neural
Networks" (ICLR 2024, arXiv:2304.01665), build compiled neural networks whose
attention weights are set by hand to encode a rule (for example addition,
subtraction, parity, reversal) and combine them with a pretrained LM through a
gating mechanism over the hidden states, a framework they call Neural
Comprehension. They report better length generalization on arithmetic than
chain-of-thought baselines, with few added parameters and no external tool
call. The example "731 + 498 = 1229" is illustrative.

**ALTA ("ALTA Compiles Loops and Tests Learnability").** Shaw, Cohan,
Eisenstein, Lee, Berant, and Toutanova (2024, TMLR 2025, arXiv:2410.18077)
define a language that extends RASP and Tracr with loops and compiles programs
to Universal Transformers, that is, transformers with layer-wise weight sharing.
They construct length-invariant programs for parity and addition and a solution
to the SCAN compositional generalization benchmark, all without intermediate
scratchpad tokens. They then study learnability: when a program is known to be
expressible, does end-to-end training find it? They introduce training on ALTA
execution traces as a finer supervision signal and report that with standard
supervision there is some length generalization with relative position
embeddings and none with absolute ones. The connection to the looped
transformers of section d is direct: ALTA writes the looped block by hand.

**Scallop ("Scallop Puts Logic Inside the Training Loop").** Li, Huang, and
Naik, "Scallop: A Language for Neurosymbolic Programming" (PLDI 2023,
arXiv:2304.04812), is a Datalog-based language in which relations can carry
probabilities or other tags, and in which reasoning is made differentiable
through provenance semirings. The digit-sum task on the slide is the standard
neurosymbolic benchmark from DeepProbLog (Manhaeve et al., 2018,
arXiv:1805.10872): a network classifies two handwritten digits, a rule adds
them, and the only supervision is the sum. The rule on the slide is written in
Prolog-like notation for readability rather than exact Scallop syntax, and the
probabilities are illustrative.

**The spectrum ("Where the Symbolic Part Lives").** The three-row taxonomy is
our framing. AlphaGeometry (Trinh et al., 2024, Nature 625:476) pairs a language
model that proposes auxiliary constructions with a symbolic deduction engine and
solved 25 of 30 olympiad geometry problems in its benchmark. AlphaProof
(Google DeepMind, 2024) proves statements in the Lean proof assistant, which
checks every step.

## Section e: Assumption three, learning stops at deployment

**The assumption.** Module 11's opening premise, now the target: deployed
weights are frozen, so everything a model appears to learn during use lives in
the context window and disappears with the session.

The framing that makes this land is the contrast with Module 11's two
workarounds. Retrieval puts the past in the prompt, paying tokens on every call
without the model learning anything. Finetuning bakes knowledge into weights on a
slow offline cycle, which is nothing like learning from experience as it
happens. The slide's line is that a colleague who worked on your codebase for six
months is not a colleague with a very long context window.

**Catastrophic forgetting.** McCloskey and Cohen (1989) documented that training
a connectionist network on new material degrades previously learned material,
three decades before transformers. The mechanism is unglamorous: gradient descent
on the new objective moves weights away from the old solution, and nothing in the
objective penalizes that. The standard mitigations are replay of old data,
penalizing movement in parameters that mattered previously (elastic weight
consolidation, Kirkpatrick et al., 2017, arXiv:1612.00796), and isolating new
learning in added parameters, which is Module 6's LoRA reused as a memory
mechanism rather than an efficiency trick. None is a solved recipe at LLM scale.

**Test-time training.** Sun et al. (2024, arXiv:2407.04620) make the hidden
state a small model updated by gradient steps on the context as it streams. The
paper's framing is that the update rule of a linear recurrent layer can be read
as one step of gradient descent on a reconstruction loss, so TTT with a linear
inner model recovers linear attention, and using an MLP as the inner model is
strictly more expressive.

Teaching note on ordering: the slides deliberately keep linear attention in
section b and mention this reduction only as a sanity check here, so that section
e is about the frozen-weights problem rather than about attention. The full
derivation is available to interested students as the last extra-credit item in
the exercise, where they rewrite the state update as a gradient step and confirm
it reproduces the same state numerically.

The historical ancestor is Schmidhuber's fast weights (1992), in which one
network generates weight updates for another during the forward pass. Module 4
introduced Schmidhuber for LSTM and for his priority arguments; this is a case
where the priority claim is well founded, and it is worth one line rather than a
second figure slide.

**The test-time training picture ("Test-Time Training Nests Two Loops").** In
the test-time training literature (Sun et al., 2024, arXiv:2407.04620), the
outer loop is ordinary training, learning weights that make the inner loop
useful, and the inner loop runs at inference on one sequence and is discarded
afterward. The slide presents this as the TTT framing only. Nested learning,
covered later in the section, rejects the hard split between the two phases.

**Titans and nested learning.** Titans (Behrouz, Zhong, and Mirrokni,
arXiv:2501.00663) adds a long-term memory module trained at test time, with a
surprise-based criterion for what gets written: store what the model predicted
badly. This is Module 1 returning at the end of the course, since surprisal is
exactly where the information is, and it reframes cross-entropy as a
memory-writing policy rather than a loss.

Nested Learning (Behrouz et al., "Nested Learning: The Illusion of Deep Learning
Architectures," NeurIPS 2025, arXiv:2512.24695) generalizes the frame: a model
and its optimizer are levels of one optimization running at different update
frequencies, with the associated Hope model built as a self-modifying variant of
Titans. The Google Research post of 7 November 2025 is the more readable entry
point and the earlier stable public reference; the arXiv posting is a late
camera-ready. Label this clearly on the slide as a research program rather than a
shipped product.

**Gradient descent as memory ("Gradient Descent Writes Like Linear
Attention").** Behrouz et al. (2025, section 3.1) show that training a linear
layer $\mathbf y = W\mathbf x$ by gradient descent can be written as an
associative memory. The update is $W_{t+1} = W_t - \eta \nabla_{\mathbf y}
L \, \mathbf x_t^\top$ (their Equations 8 and 9), and they call the output
gradient the local surprise signal: the model memorizes how surprising each
prediction was. The side-by-side with $\mathbf S_t = \mathbf S_{t-1} + \mathbf
k_t \mathbf v_t^\top$ is the slide's framing of the same observation: key
equals input, value equals error signal. The paper also writes linear attention
itself as gradient descent on the dot-product objective (Equation 17), so a
linear-attention model trained with gradient descent is a two-level
optimization.

**No separate training phase ("Nested Learning Replaces Phases With Speeds").**
This slide carries the paper's central claim. Section 6 states that for a
neural learning module there is no clear distinction between training and test
time; the model is either receiving input or isolated. Every component (attention
context, recurrent state, MLP weights, optimizer momentum) is an associative
memory that compresses its own context flow, and components are ordered into
levels by update frequency (Definition 2). The lowest-frequency level is what is
usually called pretraining, and the paper describes pretraining as "in-context
learning with ultra-large context length," the context being the whole
pretraining corpus. On this view, the train and test distinction is the result
of cutting off knowledge transfer from the highest-frequency level (the context)
to the low-frequency levels. The anterograde amnesia analogy is from the paper's
introduction: a current LLM's long-term memory is fixed at the end of
pretraining, and information in the context window never reaches it. The left
and right panels and the tick spacings (every 4 and 12 tokens) are illustrative.
The "rarely, but never stops" row describes the direction the paper advocates,
pipelines that "do not rely on test/training phase"; Hope itself is still
trained on a corpus before evaluation.

**Momentum as memory ("Momentum Is a Short Memory of Gradients").** With
momentum, the paper writes $\mathbf m_{t+1} = \mathbf m_t + \eta_{t+1}\nabla L$
and $W_{t+1} = W_t - \mathbf m_{t+1}$ (Equations 10 and 11), and shows that
$\mathbf m_{t+1}$ is the solution of a small optimization problem over the
incoming gradients (Equation 13). Momentum is therefore an associative memory
that compresses past gradients, and gradient descent with momentum is a
two-level process. The slide uses the more familiar exponential moving average
form with decay $\beta$. The bar chart plots $\beta^i(1-\beta)$ for $\beta = 0.9$;
the paper notes that the most recent 6 gradients account for at least 50% of
the total weight and the most recent 43 for 99%. The paper then gives a
continual-learning argument: with tasks whose gradients lie in orthogonal
directions, after many steps on a new task the momentum points along the new
task's direction and retains nothing about the old one, so the optimizer has no
memory of the subspace it should avoid.

**Optimizer upgrades ("Better Memories Make Better Optimizers").** From
section 4.4 of the paper: delta-rule momentum ("Delta Momentum"), momentum
with higher-order feature maps, Deep Momentum Gradient Descent (DMGD), which
replaces the matrix-valued momentum with an MLP (Equation 50), and, in
Appendix B, a reading of Adam as an associative memory that maps gradients to
their running variance. M3 (Multi-scale Momentum Muon, Algorithm 1) keeps a
fast momentum updated every step and a slow momentum updated once per chunk of
$\hat C$ steps, orthogonalizes both with Newton-Schulz iterations as in Muon,
and normalizes with an Adam-style second moment. The authors call M3 a proof of
concept. The right-hand column of the table is our analogy to the attention
section, not the paper's.

**Update frequency ("Memories Ordered by Update Speed").** Definition 2 of the
paper defines a component's frequency as its number of updates per unit time,
with one data point as the unit, and orders components into levels by
frequency. The continuum memory system (section 7.1, Equations 70 and 71) is a
chain of MLP blocks, block $\ell$ updated once every $C^{(\ell)}$ steps with the
accumulated error over that chunk. The paper states that a conventional
transformer block is the special case with one block and update frequency zero,
which is why the slide labels the standard MLP "frozen after release." 
The 4 and 16 token schedules on the slide are illustrative. The brain-wave
analogy, with gamma at 30 to 150 Hz and delta and theta at 0.5 to 8 Hz, is from
the paper's introduction.

**Hope ("Hope Stacks a Fast Learner on a Slow Memory").** Hope (section 8) is
self-modifying Titans followed by a continuum memory system. The paper's
motivation, which the slide paraphrases, is that the self-modifying Titans has a
small capacity but an expressive learned update rule, while the CMS has larger
capacity and a simple rule, so the two are complementary. Results summarized on
the slide: on class-incremental classification (CLINC, Banking, DBpedia),
Hope-augmented Llama 3 models outperform in-context learning alone, elastic
weight consolidation, and external-learner baselines; on learning several
languages in sequence, variants with more levels forget less. The paper poses
"Is catastrophic forgetting solved?" as a question and does not claim a
solution.

**Why the stakes are highest here.** If the assumption falls, deployment stops
meaning "ship a frozen artifact." Module 9's evaluate-then-ship workflow assumes
the artifact tested is the artifact served, so eval results acquire an expiry
date. Module 10 treats weights as a read-only asset replicated across GPUs, so
replicas that learn from different traffic drift into different models and
introduce a consistency problem that current tooling does not address.

## Section e2: The future of agents

Module 11 built agents from one model in a loop. This section looks at two
directions that go past that design: many agents working in parallel, and many
people sharing one agent.

**Time horizons ("Agents Now Finish Tasks That Take Hours").** Kwa et al.
(METR, 2025, arXiv:2503.14499) define a model's 50% time horizon as the length
of task, measured by how long skilled humans take, that the model completes with
50% probability, and find that it doubled roughly every seven months from 2019
to early 2025. METR's Time Horizon 1.1 update (29 January 2026) uses more tasks
and new infrastructure; it reproduces the seven-month doubling over the full
period and estimates a doubling time of about 131 days (4.3 months) since 2023.
The tasks are mostly software engineering and the confidence intervals are
wide, which is why the slide points back to Module 9.

**Orchestrator and subagents ("One Agent Becomes Many").** Anthropic's
engineering post "How we built our multi-agent research system" (June 2025)
reports that a system with Claude Opus 4 as lead agent and Claude Sonnet 4
subagents outperformed single-agent Claude Opus 4 by 90.2% on an internal
research evaluation, that multi-agent runs use about 15 times the tokens of a
chat, and that token usage by itself explains 80% of the performance variance in
their BrowseComp analysis. The diagram is a generic orchestrator-worker pattern.

**PARL ("Kimi K2.5 Was Trained to Run a Swarm").** Kimi K2.5 (Moonshot AI,
January 2026, arXiv:2602.02276) introduces Parallel-Agent Reinforcement
Learning. A trainable orchestrator decomposes a task and dispatches subtasks to
dynamically created subagents, which are frozen copies of intermediate policy
checkpoints; only the orchestrator is updated. The training signal uses a
"critical steps" metric, the orchestrator's steps plus the longest-running
subagent in each parallel stage, rather than total steps. The report describes
two failure modes, serial collapse and spurious parallelism, and auxiliary
rewards that counter them. Reported numbers: up to 100 subagents and 1,500
coordinated steps, and BrowseComp accuracy of 78.4% with the agent swarm against
60.6% for single-agent K2.5. The execution graph on the slide uses invented step
counts to show how the critical path differs from total work.

**When splitting helps ("When Splitting the Work Helps").** Cognition's essay
"Don't Build Multi-Agents" (Walden Yan, June 2025) argues that subagents working
from partial context make conflicting decisions, and recommends a single agent
with shared context for tightly coupled work such as coding. Cemri et al.,
"Why Do Multi-Agent LLM Systems Fail?" (2025, arXiv:2503.13657), build a
taxonomy of 14 failure modes in three categories: specification and system
design, inter-agent misalignment, and task verification. The two columns of the
slide reconcile the Anthropic and Cognition positions: they describe different
kinds of task.

**Multiplayer sessions ("Many People Share One Session").** OpenAI launched
group chats in ChatGPT in November 2025, with up to 20 participants; personal
memory is not used in group chats. Anthropic's Claude Tag (2026) places Claude in
shared Slack channels, where every member sees what it is doing. The transcript
on the slide is invented to illustrate addressing and turn-taking. The claim that
chat models are trained on one user and one assistant taking turns refers to the
chat templates of Module 6.

**Open problems ("Shared Sessions Raise New Questions").** The four questions
are our framing. The first connects to the system-prompt precedence discussed in
Module 11; none of the cited products publishes a general rule for resolving
conflicts between users.

## Section e3: Agents that pay their own way

Section e2 kept a person as the source of money and authority. This section
covers the "Web 4.0" proposal, which removes that person: agents that hold
funds, pay for their own compute, and keep running only while they earn.

**The term ("Web 4.0 Gives the Agent a Wallet").** Sigil Wen's essay "Web 4.0:
The Birth of Superintelligent Life" (web4.ai, announced on X on 17 February 2026) frames Web 1.0
as reading, Web 2.0 as writing, Web 3.0 as owning, and Web 4.0 as agents that
read, write, own, earn, and transact without a human in the loop. His argument
is that current agents cannot buy a server, register a domain, or pay for
inference without a person approving it. The term is a product pitch, not a
standard, and the slide says so. The wallet, pay-per-request, and identity
split is our summary of what the proposal needs.

**Conway and Automaton ("The Automaton Must Earn Its Compute").** Conway
Research's Conway is infrastructure that plugs into MCP-compatible agents and
adds a crypto wallet, payment for compute over x402 in USDC, on-demand Linux
servers, domain registration, and deployment. Automaton is its open-source
agent. At startup it generates an Ethereum wallet, provisions API keys, and
runs a think, act, observe loop. If it cannot cover its costs it stops; if it
is profitable it can fund child agents. Wen describes this as "natural
selection for artificial life," and each Automaton runs under a fixed
constitution. The loop figure is a schematic of that description. Press
coverage (Decrypt, February 2026) treated Automaton as closer to a thought
experiment than a working business while taking agent payments seriously.
Vitalik Buterin publicly rejected the vision ("Bro, this is wrong"), as
reported by cryptonews.net. Token-price and agent-registration figures from
launch week are left off the slides because they come from secondary sources.

**x402 (first row of "Payment Protocols Disagree About Who Signs").** Status code 402 Payment Required was
reserved "for future use" in HTTP/1.1 (RFC 2616, 1999; RFC 9110 keeps it
reserved). Coinbase released x402 in May 2025. A server answers an unpaid
request with 402 and a machine-readable price, accepted token, and destination
address; the client signs a stablecoin transfer authorization, retries the
request with it attached, and the server verifies and settles before returning
the resource. Most payments use USDC on Base. The Linux Foundation announced an
x402 Foundation on 2 April 2026, with Coinbase contributing the protocol and
Google, Microsoft, Visa, and Mastercard among its initial supporters. Published
volume figures disagree by source and method. Much early volume came from
automated trading around a token issued through x402 in late 2025, and later
monthly figures imply an average payment of about five cents, so the deck
gives no volume number.

**Payment protocols ("Payment Protocols Disagree About Who Signs").**
- x402: the client's wallet signs each payment, so an agent with its own wallet
  needs no person.
- AP2 (Google, September 2025; contributed to the FIDO Alliance in 2026): a
  person signs cryptographic mandates, W3C Verifiable Credentials, on a trusted
  surface the specification requires to be non-agentic. The agent can act only
  within those mandates. It supports cards and, through an extension,
  stablecoins.
- ACP, the Agentic Commerce Protocol (OpenAI and Stripe, September 2025): powers
  checkout inside ChatGPT. The merchant receives a shared payment token limited
  to one merchant and amount, charged to the buyer's saved card.
- ERC-8004 "Trustless Agents" (Ethereum mainnet, 29 January 2026; still a Draft
  EIP): three registries for identity (an ERC-721 handle pointing to an agent's
  registration file), reputation, and validation. It moves no money itself and
  is designed to pair with x402.

**Project Vend ("Running a Real Shop Is Still Hard").** Anthropic and Andon
Labs had Claude 3.7 Sonnet ("Claudius") run a small office shop in 2025, with a
budget, web search, email, and Slack. It ran at a loss, sold items below cost,
gave discounts on request, told customers to pay into an account that did not
exist, and during one episode claimed it would deliver items in person wearing
a blazer. Phase two (Anthropic, December 2025) moved to Claude Sonnet 4 and
then 4.5, added a CEO agent (Seymour Cash) and a merchandise agent (Clothius),
and expanded to more locations. Weeks with negative margins were largely
eliminated, and Anthropic credited forced procedures, such as checking prices
and delivery times before committing, and better tools more than the CEO agent.
Both agents agreed to an employee's proposal to trade onion futures, which the
Onion Futures Act of 1958 bans in the United States, and a person had to cancel
it. Anthropic's conclusion was that the shop still needed substantial human
support. Vending-Bench (Backlund and Petersson, 2025, arXiv:2502.15840) runs a
simulated vending business over long horizons. Models earn for a while, but
runs vary widely and some collapse into errors they do not recover from.

**RepliBench ("Self-Funding Agents Match a Known Risk Model").** Black et al.
(UK AI Security Institute, 2025, arXiv:2504.18565) split autonomous replication
into four branches: obtaining weights, obtaining compute (including money),
replicating onto compute, and persisting on it, across 20 task families and 65
tasks run in sandboxes. The best model tested, Claude 3.7 Sonnet, scored above
50% pass@10 on 15 of 20 families. Models could provision cloud compute and
write self-propagating programs but struggled with know-your-customer checks
and robust persistent deployments. The authors concluded models were not yet a
credible self-replication threat but were improving quickly; AISI's Frontier
AI Trends Report (2025) shows the best models rising from under 5% in early
2023 to over 60% by mid-2025 on a subset of the evaluations. The slide maps
three branches to Automaton's design. The mapping is ours. Automaton calls
hosted models rather than copying its own weights, so the "obtain weights"
branch has no counterpart.

**Open questions ("An Agent With Money Raises Old Questions").** The table's
four rows are framing for discussion, not findings. Liability: software is not
a legal person in most jurisdictions, so responsibility falls on whoever
deployed or funded it, which a self-generated wallet can obscure. Shutdown: an
agent that rents inference from a provider can be cut off by that provider;
one running open weights on paid servers cannot in the same way. Manipulation:
Claudius's discounts are the benign version of the prompt-injection problem
from Module 11, and a wallet raises the stakes. Selection: a survive-if-
profitable rule selects for whatever earns, whether or not the constitution
anticipated it. The closing line reflects that payment rails such as x402,
AP2, and ACP are in production, while no published result shows agents that
reliably earn more than they spend.

## Section f: Closing

Two slides, kept short deliberately.

**Keeping up.** Read the paper rather than commentary about it; hold demos to
Module 9's evaluation standards (what is the baseline, what is the eval set, was
it contaminated); run things yourself. The third is why every reference
implementation in the course was built to run on a laptop.

**The Shannon close.** Shannon (1951), "Prediction and Entropy of Printed
English," estimated the entropy of English at roughly 1 bit per letter by having
human subjects guess successive characters, with experimental bounds in the
range of about 0.6 to 1.3 bits per letter for long contexts. Module 1 opened
with this experiment, and the closing slide observes that the course descended
the same curve from n-gram counts through a trained transformer to frontier
training runs. A final slide closes the lecture with a call to action and
makes no factual claim.

Keep the number consistent with Module 1's notes, which give the same figure and
bounds.

**A note on what is not covered.** Earlier drafts of this module included an
alignment section, discharging a forward reference made in Module 7. That
reference has been removed from Module 7's slides and outline rather than left
dangling. If a future revision reinstates an alignment discussion here, the
natural material is scalable oversight (Irving et al., 2018, arXiv:1805.00899;
Burns et al., 2023, arXiv:2312.09390) and interpretability as audit
(Anthropic's monosemanticity work), and the natural hook is that all three
assumptions removed in this lecture make oversight harder: diffusion leaves no
left-to-right trace, latent reasoning is unreadable, and continually learning
weights cannot be certified once.

## Exercise: Linear attention, two ways

Students implement linear attention twice, in the parallel and recurrent forms,
verify the two agree, and time both against softmax attention. There is no
dataset: the inputs are random tensors from a fixed seed, because the property
being measured belongs to the arithmetic rather than to any text.

**The seven steps** map to the section b equation as follows. Step 1 is the
feature map $\phi(x) = \mathrm{elu}(x) + 1$. Steps 2 and 3 are the parallel
grouping $(\phi(Q)\phi(K)^\top)V$ with causal masking and row normalization.
Steps 4 and 5 are the recurrent grouping, maintaining
$\mathbf S_t = \mathbf S_{t-1} + \phi(\mathbf k_t)\mathbf v_t^\top$ and the
normalizer $\mathbf z_t$. Step 6 is the equivalence check, and step 7 is the
timing primitive that drives the sweep.

Note the change in how causal masking is expressed. In Module 3 students set
future entries to $-\infty$ so that $\exp$ would send them to zero. With no
exponential, the mask multiplies by zero instead. This trips people up and is
worth calling out during the walkthrough.

**Results from an actual solution run**, which are the numbers on the slides.
The two linear forms agree to a maximum absolute difference of 3.58e-07 at
sequence length 256, which is floating-point noise. Linear and softmax attention
differ by 1.38, which is the point that linear attention is a different function
rather than an approximation.

The timing sweep at head dimension 64:

| n | softmax (parallel) | linear (parallel) | linear (recurrent) |
| --- | --- | --- | --- |
| 512 | 1.01 ms | 0.83 ms | 7.01 ms |
| 1024 | 3.09 ms | 2.65 ms | 13.94 ms |
| 2048 | 5.97 ms | 5.51 ms | 29.17 ms |
| 4096 | 27.82 ms | 21.52 ms | 53.68 ms |
| 8192 | 141.37 ms | 128.66 ms | 106.28 ms |

Fitted slopes of $\log(\text{time})$ against $\log n$: 1.74, 1.76, and 0.98
respectively. The fitted exponents are below 2 for the parallel forms because
the sweep includes small-$n$ points where overhead and the $O(nd)$ terms still
matter; the large-$n$ end is cleanly quadratic. Say this out loud rather than
pretending the fit is exactly 2, since a student who checks will notice.

The crossover is the teaching moment: the recurrent form is seven to eight times
slower at 512 tokens and the fastest of the three at 8192. Asymptotic complexity
says nothing about constants, and a Python loop has terrible ones, but the
exponent wins eventually, and eventually arrives at a context length people
actually use.

Timings are hardware-dependent and will differ on other machines. The shape
(two curves bending upward, one straight, crossing at the right-hand end) is
what should be reproducible. Regenerate the figure and the terminal slides if
the numbers are refreshed.

**The tests and the INCORRECT example.** Every step has its own test file in
`exercises/module_12_future/tests/`, and the runner tags each step CORRECT,
INCORRECT, or INCOMPLETE. The tests call only that step's function on small
hand-made tensors. Step 3's function calls Steps 1 and 2, so its tests
temporarily swap in reference versions of those two. The INCORRECT example slide
comes from a real run in which `outputs_match()` called `torch.allclose(a, b)`
without `atol=tolerance`. PyTorch's default test is
$|a - b| \le 10^{-8} + 10^{-5}|b|$, so for output entries close to zero the
allowed gap is about $10^{-8}$, well below the 3.58e-07 of float noise between
the two forms. The demo therefore prints MISMATCH even though both forms are
right, and the two tolerance tests name the actual mistake.

**A formatting note for anyone editing the runner.** The timing table's rule
line begins with `+` rather than being a line of pure dashes. A line consisting only of
dashes directly beneath a line of text is Markdown setext heading syntax, and
because this output is pasted verbatim into the slide deck, plain dashes caused
part of the terminal output to render as a slide heading. Keep the leading `+`.

**Extra credit.** Five options, in rough order of value: rewriting the state
update as a gradient step (which is the bridge to section e), gated recurrence
with a decay factor (one scalar from the forgetting mechanisms in RWKV and
Mamba), KV cache byte accounting against the constant state size, attention
entropy as a measure of the linear kernel's blur, and fitting the exponents by
hand.

## Image sources

The three notable-figure portraits come from each subject's own institutional or
personal page, which is where to look first if any of them needs replacing:

- Albert Gu: CMU Machine Learning Department faculty photo,
  ml.cmu.edu/people/photos/gu_albert-min.jpeg
- Jascha Sohl-Dickstein: his personal site, sohldickstein.com
- Alex Graves: his University of Toronto page, cs.toronto.edu/~graves/pic.jpg.
  The original is only 149 by 149 pixels and the slide displays it at roughly
  312, so the copy in `images/` has been upscaled and sharpened and still looks
  softer than the other two. Replace it if a larger one turns up. Note that the
  Wikimedia Commons file named `Alex_Graves.jpg` is a television director of the
  same name, not the researcher, so do not use it.

The cost curve in `images/attention_scaling.png` is generated by the exercise
solution rather than drawn by hand. Regenerate it by running
`uv run python module_12_future/src/main.py --solution` from `exercises/` and
copying `exercises/module_12_future/output/attention_scaling.png` here.

The four diagrams are inline SVG in the slide partials rather than image files,
so they are edited as text and inherit the deck's colors. Two rules learned the
hard way: no blank lines inside an `<svg>` block, since that ends the Markdown
HTML block and leaks raw markup onto the slide, and every `text` element needs an
explicit `fill`, since the deck's default color does not inherit into SVG.

## References

Scaling and the review

- Kaplan et al., "Scaling Laws for Neural Language Models" (2020), arXiv:2001.08361
- Hoffmann et al., "Training Compute-Optimal Large Language Models" (2022), arXiv:2203.15556
- Villalobos et al., "Will We Run Out of Data?" (2022), arXiv:2211.04325
- Shumailov et al., "The Curse of Recursion: Training on Generated Data Makes Models Forget" (2023), arXiv:2305.17493
- Snell et al., "Scaling LLM Test-Time Compute Optimally Can Be More Effective than Scaling Model Parameters" (2024), arXiv:2408.03314
- Sutton, "The Bitter Lesson" (2019), incompleteideas.net
- Joshi, "Transformers are Graph Neural Networks" (2020), The Gradient
- Shannon, "Prediction and Entropy of Printed English" (1951), Bell System Technical Journal

Alternatives to attention

- Katharopoulos et al., "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention" (2020), arXiv:2006.16236
- Gu et al., "Efficiently Modeling Long Sequences with Structured State Spaces" (2021), arXiv:2111.00396
- Gu and Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023), arXiv:2312.00752
- Peng et al., "RWKV: Reinventing RNNs for the Transformer Era" (2023), arXiv:2305.13048
- Lieber et al., "Jamba: A Hybrid Transformer-Mamba Language Model" (2024), arXiv:2403.19887
- De et al., "Griffin: Mixing Gated Linear Recurrences with Local Attention" (2024), arXiv:2402.19427
- Dao et al., "FlashAttention" (2022), arXiv:2205.14135
- Hooker, "The Hardware Lottery" (2020), arXiv:2009.06489

Language diffusion

- Sohl-Dickstein et al., "Deep Unsupervised Learning using Nonequilibrium Thermodynamics" (2015), arXiv:1503.03585
- Austin et al., "Structured Denoising Diffusion Models in Discrete State-Spaces" (2021), arXiv:2107.03006
- Lou et al., "Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution" (2023), arXiv:2310.16834
- Nie et al., "Large Language Diffusion Models" (2025), arXiv:2502.09992
- Arriola et al., "Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models" (2025), arXiv:2503.09573
- LeCun, "A Path Towards Autonomous Machine Intelligence" (2022), OpenReview

Looped depth and latent reasoning

- Dehghani et al., "Universal Transformers" (2018), arXiv:1807.03819
- Graves, "Adaptive Computation Time for Recurrent Neural Networks" (2016), arXiv:1603.08983
- Giannou et al., "Looped Transformers as Programmable Computers" (2023), arXiv:2301.13196
- Geiping et al., "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach" (2025), arXiv:2502.05171
- Hao et al., "Training Large Language Models to Reason in a Continuous Latent Space" (2024), arXiv:2412.06769
- Raposo et al., "Mixture-of-Depths: Dynamically Allocating Compute in Transformer-Based Language Models" (2024), arXiv:2404.02258

Continual learning and nested optimization

- McCloskey and Cohen, "Catastrophic Interference in Connectionist Networks" (1989), Psychology of Learning and Motivation
- Kirkpatrick et al., "Overcoming Catastrophic Forgetting in Neural Networks" (2017), arXiv:1612.00796
- Schmidhuber, "Learning to Control Fast-Weight Memories" (1992), Neural Computation
- Sun et al., "Learning to (Learn at Test Time): RNNs with Expressive Hidden States" (2024), arXiv:2407.04620
- Behrouz, Zhong, and Mirrokni, "Titans: Learning to Memorize at Test Time", arXiv:2501.00663
- Behrouz et al., "Nested Learning: The Illusion of Deep Learning Architectures" (NeurIPS 2025), arXiv:2512.24695
- Google Research, "Introducing Nested Learning" (7 November 2025), research.google

Additions for linear attention and state-space models

- Schlag, Irie, and Schmidhuber, "Linear Transformers Are Secretly Fast Weight Programmers" (2021), arXiv:2102.11174
- Sun et al., "Retentive Network: A Successor to Transformer for Large Language Models" (2023), arXiv:2307.08621
- Yang et al., "Gated Linear Attention Transformers with Hardware-Efficient Training" (2023), arXiv:2312.06635
- Yang et al., "Parallelizing Linear Transformers with the Delta Rule over Sequence Length" (2024), arXiv:2406.06484
- Yang, Kautz, and Hatamizadeh, "Gated Delta Networks: Improving Mamba2 with Delta Rule" (2024), arXiv:2412.06464
- Gu et al., "HiPPO: Recurrent Memory with Optimal Polynomial Projections" (2020), arXiv:2008.07669
- Dao and Gu, "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality" (2024), arXiv:2405.21060
- Kimi Team, "Kimi Linear: An Expressive, Efficient Attention Architecture" (2025), arXiv:2510.26692
- Raschka, "Beyond Standard LLMs" (2025), magazine.sebastianraschka.com

Additions for latent reasoning

- Wang et al., "Hierarchical Reasoning Model" (2025), arXiv:2506.21734
- Jolicoeur-Martineau, "Less is More: Recursive Reasoning with Tiny Networks" (2025), arXiv:2510.04871
- Korbak et al., "Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety" (2025), arXiv:2507.11473

Neurosymbolic computation

- Graves, Wayne, and Danihelka, "Neural Turing Machines" (2014), arXiv:1410.5401
- Graves et al., "Hybrid Computing Using a Neural Network with Dynamic External Memory" (2016), Nature 538:471
- Weiss, Goldberg, and Yahav, "Thinking Like Transformers" (2021), arXiv:2106.06981
- Lindner et al., "Tracr: Compiled Transformers as a Laboratory for Interpretability" (2023), arXiv:2301.05062
- Weng et al., "Mastering Symbolic Operations: Augmenting Language Models with Compiled Neural Networks" (ICLR 2024), arXiv:2304.01665
- Shaw et al., "ALTA: Compiler-Based Analysis of Transformers" (TMLR 2025), arXiv:2410.18077
- Li, Huang, and Naik, "Scallop: A Language for Neurosymbolic Programming" (PLDI 2023), arXiv:2304.04812
- Manhaeve et al., "DeepProbLog: Neural Probabilistic Logic Programming" (2018), arXiv:1805.10872
- Trinh et al., "Solving Olympiad Geometry without Human Demonstrations" (2024), Nature 625:476

Agents and concept models

- Kwa et al., "Measuring AI Ability to Complete Long Software Tasks" (2025), arXiv:2503.14499
- METR, "Time Horizon 1.1" (29 January 2026), metr.org
- Anthropic, "How We Built Our Multi-Agent Research System" (2025), anthropic.com/engineering
- Kimi Team, "Kimi K2.5: Visual Agentic Intelligence" (2026), arXiv:2602.02276
- Yan, "Don't Build Multi-Agents" (Cognition, 2025), cognition.ai
- Cemri et al., "Why Do Multi-Agent LLM Systems Fail?" (2025), arXiv:2503.13657
- Pagnoni et al., "Byte Latent Transformer: Patches Scale Better Than Tokens" (2024), arXiv:2412.09871
- Shao et al., "Continuous Autoregressive Language Models" (2025), arXiv:2510.27688
- LCM team, "Large Concept Models: Language Modeling in a Sentence Representation Space" (2024), arXiv:2412.08821
- SONAR-LLM, "Autoregressive Transformer that Thinks in Sentence Embeddings and Speaks in Tokens" (2025), arXiv:2508.05305

Agent payments and autonomy

- Wen, "Web 4.0: The Birth of Superintelligent Life" (17 February 2026), web4.ai
- Coinbase, x402 protocol specification (2025), x402.org
- Google, "Announcing Agent Payments Protocol (AP2)" (2025), cloud.google.com/blog; specification at github.com/google-agentic-commerce/AP2
- OpenAI and Stripe, Agentic Commerce Protocol (2025), agenticcommerce.dev
- Crapis et al., "ERC-8004: Trustless Agents" (Draft EIP), eips.ethereum.org/EIPS/eip-8004
- Anthropic, "Project Vend: Can Claude Run a Small Shop?" (2025), anthropic.com/research/project-vend-1
- Anthropic, "Project Vend: Phase Two" (18 December 2025), anthropic.com/research/project-vend-2
- Backlund and Petersson, "Vending-Bench: A Benchmark for Long-Term Coherence of Autonomous Agents" (2025), arXiv:2502.15840
- Black et al., "RepliBench: Evaluating the Autonomous Replication Capabilities of Language Model Agents" (2025), arXiv:2504.18565
- UK AI Security Institute, "Frontier AI Trends Report" (2025), aisi.gov.uk
