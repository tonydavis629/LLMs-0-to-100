# Module 12: The Future of LLMs. Lecture Notes

These notes give an explanation and a citation for every major claim on the
slides, map the equations to the visuals they appear on, and record the
historical context. Module 12 is the last class and it runs in two halves. The
first is a review of the whole course, on the grounds that the material only
becomes a stack once students can see all of it at once. The second asks four
questions: whether attention is the right primitive at all (section b), and then
three assumptions the course presented as facts when each was a design decision
(sections c, d, and e).

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

**Side quest: the Bitter Lesson.** Sutton's essay (2019) argues that general
methods leveraging computation beat methods built on human insight. The exercise
for the class is to apply it to this section's own contenders rather than to
receive it as a conclusion: is selectivity human structure or a better way to
spend compute? Are hand-designed hybrid layer allocations durable? There is no
answer key, and the useful outcome is that students separate "is this clever"
from "does this scale."

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

**Naming the loops.** The outer loop is ordinary pretraining (Module 5),
learning weights that make the inner loop useful. The inner loop runs at
inference on one specific sequence. The outer learns how to learn; the inner
learns the document in front of it. Every model in the course has the outer
loop; the proposal is that the inner loop should be a real optimizer rather than
a hand-designed update rule.

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

**Why the stakes are highest here.** If the assumption falls, deployment stops
meaning "ship a frozen artifact." Module 9's evaluate-then-ship workflow assumes
the artifact tested is the artifact served, so eval results acquire an expiry
date. Module 10 treats weights as a read-only asset replicated across GPUs, so
replicas that learn from different traffic drift into different models and
introduce a consistency problem that current tooling does not address.

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
training runs, and that the curve has not flattened.

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
| 512 | 0.68 ms | 0.69 ms | 5.25 ms |
| 1024 | 2.02 ms | 1.83 ms | 10.62 ms |
| 2048 | 3.77 ms | 3.82 ms | 21.45 ms |
| 4096 | 18.55 ms | 15.09 ms | 42.42 ms |
| 8192 | 107.35 ms | 91.38 ms | 87.52 ms |

Fitted slopes of $\log(\text{time})$ against $\log n$: 1.78, 1.72, and 1.01
respectively. The fitted exponents are below 2 for the parallel forms because
the sweep includes small-$n$ points where overhead and the $O(nd)$ terms still
matter; the large-$n$ end is cleanly quadratic. Say this out loud rather than
pretending the fit is exactly 2, since a student who checks will notice.

The crossover is the teaching moment: the recurrent form is roughly ten times
slower at 512 tokens and the fastest of the three at 8192. Asymptotic complexity
says nothing about constants, and a Python loop has terrible ones, but the
exponent wins eventually, and eventually arrives at a context length people
actually use.

Timings are hardware-dependent and will differ on other machines. The shape
(two curves bending upward, one straight, crossing at the right-hand end) is
what should be reproducible. Regenerate the figure and the terminal slides if
the numbers are refreshed.

**A formatting note for anyone editing the runner.** The runner's section rules
begin with `+` rather than being lines of pure dashes. A line consisting only of
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
`exercises/module_12_future/solution/src/main.py` and copying the output.

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
