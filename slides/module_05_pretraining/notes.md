# Module 5: Pretraining — Lecture Notes

Citations, math, and explanations for every claim in the presentation.

The deck order is: a review of the Module 4 machinery, what pretraining is, the training objective (causal LM and its alternatives), the data pipeline, the training recipe, reading the loss, data quality, scaling laws, distributed training, and finally the handoff from base model to assistant. Seven Manim animations carry the dynamic processes; this file maps each claim to its visual and its source.

## Review

- Module 4 produced the *architecture*: tokens enter a decoder-only transformer, which emits one vocabulary-sized logit vector per position; a decoding strategy turns logits into text. Architecture fixes the *shape* of the computation but says nothing about the *values* of the weights.
- A freshly initialized transformer has random weights (here, Gaussian with standard deviation 0.02, the GPT-2 scheme), so its output distribution is near-uniform and its samples are noise.
- **Causal masking** (Module 3/4) lets each position attend only to itself and earlier positions, so every position can be trained to predict the *next* token without seeing it.
- **Cross-entropy** (Module 2) is the training signal: $-\log p_\theta(\text{true next token})$, averaged over the corpus.

## a. What Pretraining Is

### Self-supervised learning
- Pretraining learns from broad raw text before any task-specific specialization. It is **self-supervised**: the labels are not produced by human annotators but are derived from the data itself &mdash; the label for each position is the next token.
- A sequence of length $T$ therefore yields $T$ supervised examples (each prefix $x_{<t}$ predicts $x_t$). This is why pretraining can consume trillions of tokens cheaply: supervision is free.

### Base model
- The product of pretraining is a **base model**: it has absorbed the statistical structure of language (grammar, facts, style, code) but has not been taught to follow instructions or behave like an assistant. Alignment is Module 6.
- The "to predict the next token you must understand" framing: minimizing next-token loss at scale pressures the model to learn grammar, world facts, style, code structure, task formats, and forms of implicit reasoning, because all of these make text more predictable.

### In-context / few-shot learning
- Large base models display **in-context learning**: shown a few input/output examples in the prompt, they infer and continue the task, despite never being explicitly trained on that format (Brown et al., 2020, "Language Models are Few-Shot Learners," arXiv:2005.14165). This is an emergent consequence of scale, revisited in Module 10.

## b. The Training Objective

### Causal language modeling
- The GPT-style objective predicts $x_t$ from $x_{<t}$ only. The training pair is the same sequence shifted by one: input $x_0,\dots,x_{T-1}$, target $x_1,\dots,x_T$.
- The model emits one logit vector at every position, so a single sequence yields $T$ predictions and $T$ loss terms simultaneously. Label construction is a shift, requiring no human effort.
- **Manim animation (`next-token`):** shows the input row and the target row (the same tokens shifted left by one), draws the predict-arrows, then focuses one position to show the model emitting a distribution over the next token and the resulting loss term. It makes the "shifted targets, one loss per position" idea concrete.

### The loss
- The objective is the average negative log-likelihood
$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T}\log p_\theta\left(x_t \mid x_{<t}\right),$$
which is exactly cross-entropy between the model's predicted distribution and the one-hot true next token, averaged over positions.
- Three readouts of the same number: **perplexity** $= \exp(\mathcal{L})$ (effective number of next-token choices); **bits per token** $= \mathcal{L}/\ln 2$ (Shannon's units, Module 1). Detailed in section e.

### Notable figures
- **Alec Radford** and collaborators established the GPT line of generative pretraining (Radford et al., 2018, "Improving Language Understanding by Generative Pre-Training"; Radford et al., 2019, GPT-2). The deck introduces Radford here as the reference for causal LM.
- **Devlin, Chang, Lee, and Toutanova** introduced **masked language modeling** in BERT (2018, arXiv:1810.04805): hide ~15% of tokens and predict them from both-side context. Bidirectional and excellent for understanding tasks, but not a natural generator, and only the masked fraction produces a training signal.

### Other objectives
- **Masked LM (BERT):** bidirectional cloze prediction (above).
- **Denoising / span corruption (T5):** corrupt a passage and train an encoder-decoder to reconstruct it (Raffel et al., 2019, "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer," arXiv:1910.10683). Flexible text-to-text, but heavier machinery.

### Why decoder-only causal LM became dominant
- **Dense signal:** every position is a prediction, so one sequence trains $T$ examples; MLM only learns from the ~15% it masks.
- **Trivial data construction:** inputs and targets are one stream shifted by one.
- **Natural generation:** the training objective *is* generation.
- **Prompt compatibility:** "understanding" tasks become text completion, so one model handles classification, translation, and Q&A via prompting.

### Side quest: compression is prediction
- Encoding the next token costs about $-\log_2 p(\text{token})$ bits, so a model that assigns high probability to the actual continuation needs fewer bits to store the text: prediction quality equals compression quality.
- This is the Shannon through-line from Module 1 &mdash; cross-entropy, perplexity, and bits per token measure compression, not decoration. Ilya Sutskever has publicly framed next-token prediction as compression; the **Hutter Prize** rewards compressing a fixed Wikipedia snapshot; Del&eacute;tang et al. (2023, "Language Modeling Is Compression," arXiv:2309.10668) make the equivalence precise by using LLMs as lossless compressors.

## c. The Pretraining Pipeline

- **Collect:** web pages, books, code, papers, forums, documentation. The mixture reflects the model's goals and legal/ethical constraints.
- **Filter:** remove boilerplate and low-quality pages, near-duplicate documents (deduplication), unsafe content categories, personal information, and known benchmark text where detectable.
- **Tokenize:** convert strings to token IDs with the Module 4 tokenizer (byte-level BPE in GPT-2-style models). The Module 5 exercise uses a character-level vocabulary to keep the focus on pretraining.
- **Pack:** concatenate documents into one token stream and chop it into fixed-length blocks so each batch is a regular `(batch, block)` tensor. Insert an end-of-text token at document boundaries so the model can see where one document stops and the next begins.
- **Split:** hold out a validation set before reporting metrics, so validation loss measures generalization rather than memorized batches.
- **Train and evaluate:** each step is forward pass, shifted-target cross-entropy, backpropagation, optimizer update; periodically estimate validation loss on held-out blocks.
- **Manim animation (`sequence-packing`):** three documents of different lengths slide together into one stream with red EOS separators; the stream is cut into equal blocks (EOS tokens can fall mid-block, which is why they are marked); the blocks stack into a `(batch x block)` grid.

## d. The Training Recipe

- Module 2 supplied SGD, Adam, mini-batches, and learning rates. Real pretraining adds standard engineering for stability and efficiency. The loop itself never changes: forward, loss, backward, update.
- **Manim animation (`training-loop`):** the forward/loss/backward/update cycle as a ring with a token batch entering, a gradient pulse traveling the back half, a weight grid nudging, and the loss number descending 4.18 -> 2.29 -> 1.64 over repeated loops.

### AdamW
- **AdamW** (Loshchilov and Hutter, 2017, "Decoupled Weight Decay Regularization," arXiv:1711.05101) combines Adam's per-parameter adaptive step sizes with **decoupled** weight decay (decay applied directly to the weights rather than folded into the gradient, as plain L2 in Adam effectively does). This connects to the regularization idea from Module 2i.

### Learning-rate schedule
- **Warmup** linearly ramps the learning rate from near zero over the first few hundred steps, avoiding an early blow-up while the weights are random and Adam's running statistics have not stabilized.
- **Cosine decay** then lowers the learning rate along $\tfrac{1}{2}(1+\cos(\pi\,r))$ for decay fraction $r\in[0,1]$, from the peak down to a small floor (Loshchilov and Hutter, 2016, "SGDR: Stochastic Gradient Descent with Warm Restarts," arXiv:1608.03983, popularized the cosine shape). Big steps early to explore, small steps late to settle.
- **Manim animation (`lr-schedule`):** a dot traces the linear warmup ramp into the cosine decay, with the warmup region shaded and the max/min learning rates marked.

### Stability and scale
- **Gradient clipping:** cap the global gradient norm so a single bad batch cannot wreck the weights; prevents loss spikes.
- **Gradient accumulation:** sum gradients over several small device batches before stepping, simulating a much larger effective batch than one GPU could hold.
- **Mixed precision:** do most math in `bf16` rather than `fp32` for speed and memory (more in Module 9).

### Overfit-one-batch sanity check
- Before a long run, train repeatedly on a single batch until the loss approaches zero. A model with enough capacity can memorize one tiny batch; if the loss does **not** crater, the loop is broken (detached gradient, wrong target shift, frozen parameter, bad learning rate). The exercise runs exactly this check.

## e. Reading the Loss

- Cross-entropy is the model's average **surprise** at the true next token; lower loss means more probability assigned to what actually came next.
- **Perplexity** $=\exp(\mathcal{L})$ reads as the effective number of equally likely next-token choices. **Bits per token** $=\mathcal{L}/\ln 2$ is the same loss in Shannon's units (1 nat $=1/\ln 2$ bits).
- **Manim animation (`perplexity`):** a near-uniform next-token distribution (untrained) sharpens onto the true token as training proceeds; the panel updates loss 4.18 -> 1.64, perplexity 65 -> 5.1, bits 6.03 -> 2.36, tying lower loss to better compression.
- Training loss should fall; validation loss should fall too. A **widening gap** signals overfitting or memorization. Real runs can spike or diverge when the recipe is unstable; gradient clipping, learning rate, batch size, and data issues are the usual suspects. Lower loss does not perfectly predict every capability, so real runs also track downstream benchmarks at checkpoints; generated samples are useful for intuition but unreliable as a metric.
- **The classroom demo:** sample from the same model before and after training. In the exercise, a tiny character-level model goes from random characters (loss 4.18) to text with the *shape* of Shakespeare &mdash; capitalized character names, colons, line breaks (validation loss 1.64). Perplexity falls from ~65 to ~5; bits per token from ~6.0 to ~2.4. The numbers and samples shown in the deck are the actual output of the solution run.

## f. Data Quality, Contamination, Memorization

- More data is not automatically better: quality, diversity, deduplication, and domain balance all matter. A smaller clean corpus can beat a larger pile of boilerplate.
- **Deduplication** improves generalization even though it removes tokens, because duplicated passages bias the objective toward memorizing exact strings (Lee et al., 2021, "Deduplicating Training Data Makes Language Models Better," arXiv:2107.06499).
- **Side quest, memorization vs generalization:** a passage repeated many times in a tiny dataset gets memorized verbatim rather than learned as a reusable pattern. Verbatim recall is a privacy and copyright liability and does not transfer; deduplication and validation loss are the defenses.
- **Benchmark contamination:** when evaluation examples leak into pretraining data, the model can score by recall rather than ability, overstating capability. It is easy to introduce by accident (benchmarks are published on the scraped web) and hard to fully rule out.
- **PII and copyright:** personally identifying and copyrighted text raise legal, ethical, and product risks beyond pure accuracy.
- **Data mixture shapes behavior:** code-heavy data improves coding (and some structured reasoning); academic text shifts style and knowledge; conversational text changes dialogue handling. Open datasets made this concrete: **The Pile** (Gao, Biderman, and EleutherAI collaborators, 2020, "The Pile: An 800GB Dataset of Diverse Text for Language Modeling," arXiv:2101.00027) is a 22-source curated mixture widely used for open pretraining.
- **Side quest, the data wall:** when high-quality human text, not model size, becomes the limiting resource, responses include aggressive curation and deduplication, careful domain balance, and increasingly synthetic data. This reframes the frontier and recurs in later modules.

## g. Scaling Laws and Compute-Optimal Training

- **Scaling laws** (Kaplan, McCandlish, et al., 2020, "Scaling Laws for Neural Language Models," arXiv:2001.08361): language-model loss falls as a smooth power law in parameters $N$, data $D$, and compute $C$, each over many orders of magnitude. On log-log axes, loss versus compute is nearly a straight line, so a large model's loss can be estimated from small-scale runs.
- **Compute handle:** for a dense transformer, $C \approx 6ND$ &mdash; roughly 2 FLOPs per parameter per token for the forward pass and about twice that for the backward pass. Fixing $C$ makes $N$ and $D$ trade off directly.
- **Chinchilla** (Hoffmann et al., 2022, "Training Compute-Optimal Large Language Models," arXiv:2203.15556): for a fixed compute budget, many earlier models were too large for the number of tokens they saw; parameters and tokens should grow together, at roughly **20 training tokens per parameter**.
- **Compute-optimal training is not serving-optimal:** a model is trained once and served many times, so a model meant for heavy deployment should be smaller (cheaper at inference). Llama-style models deliberately train smaller models on far more tokens than Chinchilla suggests (Touvron et al., 2023, "LLaMA," arXiv:2302.13971), spending extra training compute to save much more inference compute. Compute-optimal training balances parameters, tokens, and deployment cost.
- This is why pretraining is an engineering problem as much as a modeling one: model size, data size, batch size, sequence length, hardware, and wall-clock time interact. **Brown and the GPT-3 team** (2020) had shown the payoff of scale via strong few-shot behavior.
- **Manim animation (`scaling-laws`):** first the descending power-law line in compute (with an extrapolation), then the Chinchilla **valley** &mdash; loss versus model size at fixed compute is U-shaped, with a compute-optimal minimum; too-small models underfit and too-big models see too few tokens.
- **Side quest, emergent abilities, real or mirage:** Wei et al. (2022, "Emergent Abilities of Large Language Models," arXiv:2206.07682) describe capabilities that appear to switch on past a scale threshold; Schaeffer et al. (2023, "Are Emergent Abilities of Large Language Models a Mirage?", arXiv:2304.15004) argue the discontinuity is often an artifact of thresholded or nonlinear metrics (e.g. exact-match accuracy), and that smoother metrics show gradual improvement.
- **Side quest, double descent:** Nakkiran et al. (2019, "Deep Double Descent," arXiv:1912.02292) show test error can fall, rise near the interpolation point, then fall again as models grow, complicating the classical Module 2 overfitting story.

## h. Distributed Training at Scale

- Pretraining becomes a distributed-systems problem once the model, batch, or run no longer fits on one device.
- **Data parallelism:** replicate the full model on each GPU, split the batch across them, and synchronize gradients with an **all-reduce** (average), so every replica stays identical. Scales throughput when the model fits on one device.
- **Model / tensor / pipeline parallelism:** split the model itself when parameters, activations, or layers do not fit &mdash; tensor parallelism splits individual weight matrices (Narayanan et al., 2021, "Megatron-LM," arXiv:2104.04473); pipeline parallelism puts different layers on different GPUs (Huang et al., 2019, "GPipe," arXiv:1909.08053). Large runs combine all of these.
- Frontier runs can use thousands of GPUs for months, so networking, storage, and scheduling become first-order concerns, and **checkpoint-and-resume** is mandatory because hardware failures are expected. Implementation details are Module 9.
- **Manim animation (`data-parallel`):** the model is replicated across four GPUs, the batch splits into shards that flow to each GPU, each computes a local gradient, and an all-reduce hub averages them and broadcasts the result back so every replica updates identically.

## i. From Base Model to Assistant

- Pretraining yields a model that **continues text**, not one that reliably follows intent. A base model will complete a chat transcript (inventing both sides), imitate a document, write code, or continue harmful text, because all of these are patterns in its training distribution.
- **Side quest, base vs assistant:** the same prompt "What is the capital of France?" framed as a document is, to a base model, likely followed by *more questions* (as on a worksheet); an assistant model treats it as a request and answers. Same knowledge, different behavior.
- **Instruction finetuning** continues training on prompts paired with desired responses, shifting the data distribution so a prompt is something to satisfy. **Preference optimization and reinforcement learning** further shape helpfulness, honesty, refusal behavior, and tool use. The optimization machinery is familiar; the data and behavioral target change. This is the handoff to Module 6.

## Exercise: Pretraining NanoGPT

- The student implements the pretraining loop around a provided tiny decoder-only model (4 layers, 4 heads, width 128, context 128; 818,048 parameters) trained on the public-domain tiny-Shakespeare corpus (~1.1M characters, 65-character vocabulary). Steps: encode text, train/validation split, build shifted `(x, y)` batches, cross-entropy loss, one optimizer step (zero-grad, backprop; clipping and the step are provided), the warmup + cosine learning-rate schedule, multi-batch loss estimation, perplexity and bits per token, and autoregressive sampling.
- The runner skips any unimplemented step, prints model/dataset size, a before-training sample, a per-checkpoint loss table, final perplexity and bits per token, an after-training sample, and saves a loss-curve image. A `--overfit` mode trains on one fixed batch until the loss craters (4.18 -> 0.07 in the captured run), confirming the loop is wired correctly.
- All numbers and samples shown in the slides are captured from the solution run, per the course rule that sample outputs must be real.
- References: Karpathy's nanoGPT (<https://github.com/karpathy/nanoGPT>) and build-nanogpt (<https://github.com/karpathy/build-nanogpt>) walkthroughs informed the exercise design.

## References

- Radford et al., "Improving Language Understanding by Generative Pre-Training" (2018); Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2, 2019).
- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," arXiv:1810.04805.
- Brown et al., "Language Models are Few-Shot Learners," arXiv:2005.14165.
- Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (T5), arXiv:1910.10683.
- Kaplan et al., "Scaling Laws for Neural Language Models," arXiv:2001.08361.
- Hoffmann et al., "Training Compute-Optimal Large Language Models" (Chinchilla), arXiv:2203.15556.
- Touvron et al., "LLaMA: Open and Efficient Foundation Language Models," arXiv:2302.13971.
- Gao et al., "The Pile: An 800GB Dataset of Diverse Text for Language Modeling," arXiv:2101.00027.
- Lee et al., "Deduplicating Training Data Makes Language Models Better," arXiv:2107.06499.
- Del&eacute;tang et al., "Language Modeling Is Compression," arXiv:2309.10668.
- Wei et al., "Emergent Abilities of Large Language Models," arXiv:2206.07682.
- Schaeffer et al., "Are Emergent Abilities of Large Language Models a Mirage?", arXiv:2304.15004.
- Nakkiran et al., "Deep Double Descent: Where Bigger Models and More Data Hurt," arXiv:1912.02292.
- Loshchilov and Hutter, "Decoupled Weight Decay Regularization" (AdamW), arXiv:1711.05101; "SGDR: Stochastic Gradient Descent with Warm Restarts," arXiv:1608.03983.
- Huang et al., "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism," arXiv:1909.08053.
- Narayanan et al., "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM," arXiv:2104.04473.
