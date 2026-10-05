# Module 5: Pretraining — Lecture Notes

Citations, math, and explanations for every claim in the presentation.

The deck order is: a review of the Module 4 machinery, what pretraining is, the training objective (causal LM and its alternatives), the data pipeline, the training recipe, reading the loss, data quality, scaling laws, distributed training, and finally the handoff from base model to assistant. Eight Manim animations carry the dynamic processes; this file maps each claim to its visual and its source.

## Review

- Module 4 produced the *architecture*: tokens enter a decoder-only transformer, which emits one vocabulary-sized logit vector per position; a decoding strategy turns logits into text. Architecture fixes the *shape* of the computation but says nothing about the *values* of the weights.
- A freshly initialized transformer has random weights (here, Gaussian with standard deviation 0.02, the GPT-2 scheme), so its output distribution is near-uniform and its samples are noise.
- **Causal masking** (Module 3/4) lets each position attend only to itself and earlier positions, so every position can be trained to predict the *next* token without seeing it.
- **Cross-entropy** (Module 2) is the training signal: $-\log p_\theta(\text{true next token})$, averaged over the corpus.
- **Prediction is compression** (Module 1). Encoding a symbol of probability $p$ costs $-\log_2 p$ bits (Shannon, 1948), and cross-entropy $H(p, q) = -\sum_x p(x)\log_2 q(x)$ is the average cost when the code is built from the model's $q$ rather than the true $p$. Module 1's demo shrank "the cat sat on the mat" from 110 bits (uniform 5-bit code) to 70 bits (frequency-based prefix code). Because the pretraining loss is a cross-entropy, lowering it lowers the bits per token: a better predictor is a better compressor. Ilya Sutskever has publicly framed next-token prediction as compression; the Hutter Prize rewards compressing a fixed Wikipedia snapshot; Del&eacute;tang et al. (2023, "Language Modeling Is Compression," arXiv:2309.10668) make the equivalence precise by using LLMs as lossless compressors.

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
- **Manim animation (`next-token`):** shows the input row and the target row (the same tokens shifted left by one), draws the predict-arrows, then focuses one position: the model emits a predicted distribution over the next token, the target is shown as a **one-hot distribution** (probability 1 on the true token "on", 0 elsewhere), and cross-entropy between the two reduces to $-\log p(\text{on}) = 0.48$ because only the true token's term survives the one-hot sum. It makes the "shifted targets, one loss per position" idea concrete.

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

## c. The Pretraining Pipeline

- **Collect:** web pages, books, code, papers, forums, documentation. The mixture reflects the model's goals and legal/ethical constraints.
- **Filter:** remove boilerplate and low-quality pages, near-duplicate documents (deduplication), unsafe content categories, personal information, and known benchmark text where detectable.
- **Tokenize:** convert strings to token IDs with the Module 4 tokenizer (byte-level BPE in GPT-2-style models). The Module 5 exercise uses a character-level vocabulary to keep the focus on pretraining.
- **Pack:** concatenate documents into one token stream and chop it into fixed-length blocks so each batch is a regular `(batch, block)` tensor. Mark document boundaries with **special tokens** &mdash; reserved vocabulary IDs that never arise from tokenizing ordinary text. The conventional pair is **BOS** (beginning of sequence) and **EOS** (end of sequence); GPT-2 uses a single `<|endoftext|>` token for both roles (Radford et al., 2019). The boundary token lets the model learn that context resets at a document edge, and at inference time sampling EOS is the model's signal that generation is complete.
- **Split:** hold out a validation set before reporting metrics, so validation loss measures generalization rather than memorized batches.
- **Train and evaluate:** each step is forward pass, shifted-target cross-entropy, backpropagation, optimizer update; periodically estimate validation loss on held-out blocks.
- **Manim animation (`sequence-packing`):** three documents of different lengths slide together into one stream with red EOS separators; the stream is cut into equal blocks (EOS tokens can fall mid-block, which is why they are marked); the blocks stack into a `(batch x block)` grid.

## d. The Training Recipe

- Module 2 supplied SGD, Adam, mini-batches, and learning rates. The deck walks through one PyTorch pretraining loop line by line; the same code is shown on every slide with the current lines highlighted. It mirrors the exercise's `pretrain()` and `train_step()` (`exercises/module_05_pretraining`).
- **Setup.** A freshly initialized model predicts a near-uniform distribution over the vocabulary of size $V$, so its loss starts near $-\log(1/V) = \ln V$. The exercise uses a 65-character vocabulary (Tiny Shakespeare), so $\ln 65 \approx 4.17$; its captured validation loss before training is 4.19. AdamW stores a first-moment estimate $m$ and second-moment estimate $v$ per parameter (Kingma and Ba, 2014), so optimizer state is twice the parameter count.
- **Learning rate.** PyTorch keeps the rate in `optimizer.param_groups`; writing a new value there before each step is how a manual schedule is applied (the exercise's `lr_at_step` in `src/schedules.py`).
- **Batch.** `get_batch` draws `batch_size` random offsets and slices `block_size` tokens; the target `y` is the same slice shifted by one, so each batch carries $B \times T$ next-token problems.
- **Forward and loss.** Logits have shape $(B, T, V)$. `F.cross_entropy` takes a 2-D `(N, C)` input and 1-D `(N,)` targets, hence the flatten to $(BT, V)$ and $(BT,)$. The output is $\mathcal{L} = -\frac{1}{BT}\sum_{b,t}\log p_\theta(y_{b,t}\mid x_{b,\le t})$.
- **Backward.** `loss.backward()` computes $\partial\mathcal{L}/\partial\theta$ by reverse-mode automatic differentiation and **accumulates** it into each parameter's `.grad` (PyTorch autograd documentation), which is why `optimizer.zero_grad()` is called every step.
- **Clip and update.** `clip_grad_norm_` computes the global norm $\lVert g\rVert_2$ over all parameters' gradients and, if it exceeds the threshold $c$, rescales every gradient by $c / \lVert g\rVert_2$, preserving direction (Pascanu et al., 2013, "On the difficulty of training recurrent neural networks," arXiv:1211.5063). `optimizer.step()` then applies the AdamW update. The exercise repeats the loop for 800 steps.

### AdamW
- **AdamW** (Loshchilov and Hutter, 2017, "Decoupled Weight Decay Regularization," arXiv:1711.05101) combines Adam's per-parameter adaptive step sizes with **decoupled** weight decay (decay applied directly to the weights rather than folded into the gradient, as plain L2 in Adam effectively does). This connects to the regularization idea from Module 2i.

### Learning-rate schedule
- **Warmup** linearly ramps the learning rate from near zero over the first few hundred steps, avoiding an early blow-up while the weights are random and Adam's running statistics have not stabilized.
- **Cosine decay** then lowers the learning rate along $\tfrac{1}{2}(1+\cos(\pi\,r))$ for decay fraction $r\in[0,1]$, from the peak down to a small floor (Loshchilov and Hutter, 2016, "SGDR: Stochastic Gradient Descent with Warm Restarts," arXiv:1608.03983, popularized the cosine shape). Big steps early to explore, small steps late to settle.
- **Manim animation (`lr-schedule`):** a dot traces the linear warmup ramp into the cosine decay, with the warmup region shaded and the max/min learning rates marked.

### Gradient clipping
- A rare or malformed batch can yield a gradient far larger than typical; one step of size $\eta\,\lVert\mathbf g\rVert$ then moves the weights far from the trajectory and shows up as a loss spike.
- Global-norm clipping (Pascanu et al., 2013, arXiv:1211.5063) rescales $\mathbf g \leftarrow \mathbf g \cdot \min(1, c/\lVert\mathbf g\rVert)$, where $\lVert\mathbf g\rVert = \sqrt{\sum_p \lVert \mathbf g_p\rVert_2^2}$ is the norm over all parameters concatenated. Direction is preserved; only the length is capped. The slide's five-line version matches what `torch.nn.utils.clip_grad_norm_` computes (PyTorch adds a small epsilon to the denominator and returns the pre-clip norm).
- GPT-3 clipped the global gradient norm at 1.0 (Brown et al., 2020, arXiv:2005.14165, Appendix B), the same value the exercise and the walkthrough loop use.

### Gradient accumulation
- GPT-3 175B used a batch of 3.2M tokens (Brown et al., 2020, Table 2.1). Activations for every example must be stored for the backward pass, so memory grows with batch size.
- Because PyTorch autograd sums into `.grad` on each `backward()` call, running $k$ micro-batches before `optimizer.step()` sums their gradients. Scaling each micro-batch loss by $1/k$ makes the result equal the gradient of the mean loss over the full batch: $\nabla \frac{1}{k}\sum_i \mathcal L_i = \frac{1}{k}\sum_i \nabla\mathcal L_i$.
- Effective batch = micro-batch size × accumulation steps × data-parallel replicas (section h).

### Mixed precision
- IEEE 754 binary32 (fp32) has 1 sign, 8 exponent, and 23 mantissa bits; binary16 (fp16) has 1/5/10; bfloat16 has 1/8/7 (Kalamkar et al., 2019, "A Study of BFLOAT16 for Deep Learning Training," arXiv:1905.12322). Max finite values: fp32 and bf16 about $3.4\times10^{38}$, fp16 65,504 (checked with `torch.finfo`).
- NVIDIA A100 datasheet: 19.5 TFLOPS fp32 (non-tensor-core), 312 TFLOPS bf16/fp16 on tensor cores (dense).
- fp16's narrow exponent range makes small gradients underflow, so fp16 training needs loss scaling (Micikevicius et al., 2017, "Mixed Precision Training," arXiv:1710.03740). bf16 keeps fp32's exponent and range, so loss scaling is unnecessary.
- bf16 has 8 significant bits (7 stored plus the implicit leading 1), so the spacing near 1.0 is $2^{-7}\approx0.0078$: in bf16, $1 + 0.001 = 1$ (checked with torch). An update of that relative size to a bf16 weight is lost, which is why mixed precision keeps fp32 master weights and optimizer state and applies the update in fp32 (Micikevicius et al., 2017).
- `torch.autocast` runs eligible ops (matmuls, convolutions) in the lower precision while parameters remain fp32. Lower-precision weights for inference (quantization) are covered in Module 10.
- **What is actually used.** bf16 mixed precision is the standard recipe for LLM pretraining (e.g. BLOOM, Llama, OLMo): matmuls, activations, and gradients in bf16; master weights and optimizer state in fp32; precision-sensitive ops (softmax, layer norm, loss reductions) kept in fp32 by autocast. Training "bf16 the whole way" is uncommon because updates smaller than bf16's spacing are lost. fp16 needs dynamic loss scaling and is prone to overflow (the 104B case later in the deck); bf16's fp32-sized exponent removes both problems, which is why it displaced fp16 for large-model training. Frontier runs now push matmuls lower: DeepSeek-V3 used an FP8 mixed-precision framework (DeepSeek-AI, 2024, "DeepSeek-V3 Technical Report," arXiv:2412.19437).
- **Quantization-aware training (QAT)** inserts simulated ("fake") quantization of weights, and sometimes activations, into the forward pass during training or a late training phase, with gradients passed straight through the rounding, so the model learns weights that survive quantization to e.g. int4 for serving. Released examples include Meta's quantized Llama 3.2 1B/3B (2024, QAT with LoRA adapters) and Google's Gemma 3 QAT checkpoints (2025). Post-training quantization is covered in Module 10.

### Overfit-one-batch sanity check
- Before a long run, train repeatedly on a single batch until the loss approaches zero. A model with enough capacity can memorize one tiny batch; if the loss does **not** crater, the loop is broken (detached gradient, wrong target shift, frozen parameter, bad learning rate). The exercise runs exactly this check.

## e. Reading the Loss

- Cross-entropy is the model's average **surprise** at the true next token; lower loss means more probability assigned to what actually came next.
- The raw loss is measured in **nats**: the information unit of the natural logarithm, just as the bit is the unit of $\log_2$. Cross-entropy is computed with $\ln$ (what calculus and library `log` functions provide), so the loss comes out in nats; $1$ nat $= 1/\ln 2 \approx 1.443$ bits. The nat and the bit measure the same quantity, surprise, in different bases (Cover & Thomas, *Elements of Information Theory*, ch. 2).
- **Perplexity** $=\exp(\mathcal{L})$ reads as the effective number of equally likely next-token choices. **Bits per token** $=\mathcal{L}/\ln 2$ is the same loss in Shannon's units.
- **Manim animation (`perplexity`):** one prediction position with word tokens: the next token after "The cat sat on the", with candidates " mat", " floor", " bed", ... and an "all other tokens" bar for the rest of a ~50k-token vocabulary (GPT-2's has 50,257). The probabilities are illustrative. For a single position the loss is $-\ln p(\text{true})$, so perplexity $= e^{\mathcal L} = 1/p(\text{true})$ and bits $= \mathcal L / \ln 2 = -\log_2 p(\text{true})$. Before: $p = 0.05$, loss $3.00$ nats, perplexity $20$, $4.32$ bits. After: $p = 0.60$, loss $0.51$, perplexity $1.67$, $0.74$ bits. Averaged over a corpus, perplexity is $\exp$ of the mean loss, i.e. the geometric mean of $1/p$ across positions.
- Training loss should fall; validation loss should fall too. A **widening gap** signals overfitting or memorization. Real runs can spike or diverge when the recipe is unstable; gradient clipping, learning rate, batch size, and data issues are the usual suspects. Lower loss does not perfectly predict every capability, so real runs also track downstream benchmarks at checkpoints; generated samples are useful for intuition but unreliable as a metric.
- **Loss scaling (`loss-scaling`):** fp16's smallest positive (subnormal) value is about $6\times10^{-8}$ and its largest finite value is 65,504. Many activation gradients are smaller than the first and would flush to zero. Mixed-precision training multiplies the loss by a factor $S$ before `backward()`; by the chain rule every gradient is multiplied by $S$, shifting them into representable range, and the gradients are divided by $S$ (in fp32) before the optimizer step, so the update is unchanged. With dynamic loss scaling, any inf/NaN gradient causes the step to be skipped and $S$ halved; after a run of clean steps $S$ is increased again (Micikevicius et al., 2017, arXiv:1710.03740). The slide's diagram shifts illustrative gradient magnitudes by $S = 65{,}536 = 2^{16}$. A scale driven down to 1 means the gradients overflow fp16 even without scaling. bf16 has fp32's exponent range, so it needs no loss scaling.
- **Real training logs (`bloom-176b-log`, `bloom-104b-log`).** Both charts are plotted directly from BigScience's public TensorBoard event files on Hugging Face (<https://huggingface.co/bigscience/tr11-176B-logs>, `tensorboard/main`; <https://huggingface.co/bigscience/tr8-104B-logs>, `tensorboard/tr8-104B-exp12-f`). Raw values are drawn faint with a rolling mean (loss) or rolling median (grad norm) on top. Every number on the slides was read from those files:
  - **BLOOM 176B** (tags `lm-loss-training/lm loss`, `grad-norm/grad-norm`, converted from steps to tokens with `steps-vs-tokens`): steps 1 to 95,281 (0 to 366.5B tokens), logged 2022-03-11 to 2022-07-06. Loss is 4.40 at step 1,000 and averages 1.93 over the last 500 steps. The largest spike is at step 31,219 (97.85B tokens): loss 2.19 to 5.10 and grad norm 960 against a median of 0.14, recovering to about 2.2 within roughly 80 steps. 50 of the 117 event files begin at a step already logged by an earlier file, i.e. the job resumed from an earlier checkpoint; the plot keeps the last-written value per step.
  - **104B prototype, experiment 12-f** (tags `lm loss`, `loss-scale`): six attempts logged 2021-11-05 to 2021-11-17, each drawn separately rather than merged. The first reached step 6,600; later attempts resumed from 6,301, 8,401, 9,901, 9,601, and 8,101. The 6,301 attempt jumped from loss 3.4 to 7.27 at step 8,741, hit its first NaN at 9,027, and its fp16 loss scale reached 1 at 9,040; the 8,401, 9,901, and 9,601 attempts also produced NaN losses and loss-scale collapse (the 9,901 attempt ends at loss 20.78, step 10,669). The 8,101 attempt stops at step 8,664. Dynamic loss scaling halves the scale whenever a gradient overflows to inf or NaN (Micikevicius et al., 2017), so a scale driven to 1 means overflow on step after step.
  - BLOOM 176B was trained in bf16 mixed precision, informed by these fp16 instabilities in the 104B experiments (BigScience Workshop, 2022, "BLOOM: A 176B-Parameter Open-Access Multilingual Language Model," arXiv:2211.05100).
- **Side quest, Llama 3 trained slower at noon:** all figures are from Section 3.3.4 ("Reliability and Operational Challenges") of Meta's "The Llama 3 Herd of Models" (Dubey et al., 2024, arXiv:2407.21783). Llama 3 405B trained on up to 16K H100 GPUs. Over a 54-day snapshot of pretraining there were 466 job interruptions: 47 planned (firmware upgrades, configuration or dataset updates) and 419 unexpected, about one every three hours. Roughly 78% of the unexpected interruptions were attributed to confirmed or suspected hardware issues; the largest categories in the paper's Table 5 are faulty GPUs (148) and GPU HBM3 memory (72). Significant manual intervention was required only three times; automation handled the rest, and effective training time stayed above 90%. The paper also reports a diurnal 1-2% throughput variation caused by higher mid-day temperatures affecting GPU dynamic voltage and frequency scaling, and instant data-center power swings on the order of tens of megawatts when tens of thousands of GPUs idle or resume together (e.g. waiting on checkpointing or collective communication), "stretching the limits of the power grid."
- **The classroom demo:** sample from the same model before and after training. In the exercise, a tiny character-level model goes from random characters (loss 4.19) to text with the *shape* of Shakespeare &mdash; capitalized character names, colons, line breaks (validation loss 2.01 after 800 steps). Perplexity falls from ~66 to ~7.5; bits per token from ~6.0 to ~2.9. The numbers and samples shown in the deck are the actual output of the solution run.

## f. Data Quality, Contamination, Memorization

- More data is not automatically better: quality, diversity, deduplication, and domain balance all matter. A smaller clean corpus can beat a larger pile of boilerplate.
- **Deduplication** improves generalization even though it removes tokens, because duplicated passages bias the objective toward memorizing exact strings (Lee et al., 2021, "Deduplicating Training Data Makes Language Models Better," arXiv:2107.06499).
- **Side quest, memorization vs generalization:** a passage repeated many times in a tiny dataset gets memorized verbatim rather than learned as a reusable pattern. Verbatim recall is a privacy and copyright liability and does not transfer; deduplication and validation loss are the defenses.
- **Benchmark contamination:** when evaluation examples leak into pretraining data, the model can score by recall rather than ability, overstating capability. It is easy to introduce by accident (benchmarks are published on the scraped web) and hard to fully rule out.
- **PII and copyright:** personally identifying and copyrighted text raise legal, ethical, and product risks beyond pure accuracy.
- **Data mixture shapes behavior:** code-heavy data improves coding (and some structured reasoning); academic text shifts style and knowledge; conversational text changes dialogue handling. Open datasets made this concrete: **The Pile** (Gao, Biderman, and EleutherAI collaborators, 2020, "The Pile: An 800GB Dataset of Diverse Text for Language Modeling," arXiv:2101.00027) is a 22-source curated mixture widely used for open pretraining.
- **The data wall:** when high-quality human text, not model size, becomes the limiting resource, responses include aggressive curation and deduplication, careful domain balance, and increasingly synthetic data. This reframes the frontier and recurs in later modules.

## g. Scaling Laws and Compute-Optimal Training

- **Scaling laws** (Kaplan, McCandlish, et al., 2020, "Scaling Laws for Neural Language Models," arXiv:2001.08361): language-model loss falls as a smooth power law in parameters $N$, data $D$, and compute $C$, each over many orders of magnitude. On log-log axes, loss versus compute is nearly a straight line, so a large model's loss can be estimated from small-scale runs.
- **Compute handle:** for a dense transformer, $C \approx 6ND$ &mdash; roughly 2 FLOPs per parameter per token for the forward pass and about twice that for the backward pass (Kaplan et al., 2020, section 2.1). Fixing $C$ makes $N$ and $D$ trade off directly. The slide cites Kaplan et al. (2020) directly.
- **Chinchilla** (Hoffmann et al., 2022, "Training Compute-Optimal Large Language Models," arXiv:2203.15556): for a fixed compute budget, many earlier models were too large for the number of tokens they saw; parameters and tokens should grow together, at roughly **20 training tokens per parameter**.
- **Compute-optimal training is not serving-optimal:** a model is trained once and served many times, so a model meant for heavy deployment should be smaller (cheaper at inference). Llama-style models deliberately train smaller models on far more tokens than Chinchilla suggests (Touvron et al., 2023, "LLaMA," arXiv:2302.13971), spending extra training compute to save much more inference compute. Compute-optimal training balances parameters, tokens, and deployment cost.
- This is why pretraining is an engineering problem as much as a modeling one: model size, data size, batch size, sequence length, hardware, and wall-clock time interact. **Brown and the GPT-3 team** (2020) had shown the payoff of scale via strong few-shot behavior.
- **Manim animation (`scaling-laws`):** first the descending power-law line in compute (with an extrapolation), then the Chinchilla **valley** &mdash; loss versus model size at fixed compute is U-shaped, with a compute-optimal minimum; too-small models underfit and too-big models see too few tokens.
- **Side quest, emergent abilities, real or mirage:** Wei et al. (2022, "Emergent Abilities of Large Language Models," arXiv:2206.07682) describe capabilities that appear to switch on past a scale threshold; Schaeffer et al. (2023, "Are Emergent Abilities of Large Language Models a Mirage?", arXiv:2304.15004) argue the discontinuity is often an artifact of thresholded or nonlinear metrics (e.g. exact-match accuracy), and that smoother metrics show gradual improvement.

- **Interactive widget (`:::interactive widget="scalingPlanner"`):** fixes a compute budget $C$, then splits it using $C \approx 6ND$ with $D = r \cdot N$, so $N = \sqrt{C / 6r}$. Loss comes from the Chinchilla parametric form
  $$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}$$
  using the corrected coefficients $E = 1.8172$, $A = 482.01$, $\alpha = 0.3478$, $B = 2085.43$, $\beta = 0.3658$ from Besiroglu et al. (2024), "Chinchilla Scaling: A Replication Attempt". Those are the values whose optimum reproduces the paper's own ~20 tokens per parameter; the coefficients as printed in Hoffmann et al. (2022) put the optimum nearer 60, which is the discrepancy the replication identified. The GPT-3 (~1.7 tokens/param) and Llama 3 8B (~1875) presets show the same budget spent two other ways, and the loss-above-optimal readout quantifies what serving-optimal training costs in training loss.
- **Interactive widget (`:::interactive widget="servingPlanner"`):** compares two models that reach the same loss, following the inference-aware accounting of Sardana et al. (2023, "Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws," arXiv:2401.00448). Model A is the Chinchilla choice: $N_A$ parameters on $20N_A$ tokens, giving target loss $L^\ast = L(N_A, 20N_A)$ under the corrected fit above. Model B has $N_B = f N_A$ parameters; the tokens it needs to match $L^\ast$ come from inverting the fit, $D_B = \left(B / (L^\ast - E - A/N_B^{\alpha})\right)^{1/\beta}$. Lifetime compute after serving $d$ tokens is $6ND + 2Nd$ (one forward pass, $2N$ FLOPs, per served token). B costs more to train but less per served token, so the two lines cross at the break-even volume $d^\ast = (6N_BD_B - 6N_AD_A) / (2(N_A - N_B))$. Past $d^\ast$, the smaller, longer-trained model wins, which is the rationale for Llama-style token counts. For B below about 20% of A the fit demands tens of thousands of tokens per parameter, far outside the ratios Chinchilla measured, so the slider stops there; Sardana et al. trained models at up to 10,000 tokens per parameter and found quality kept improving.

## h. Distributed Training at Scale

- Pretraining becomes a distributed-systems problem once the model, batch, or run no longer fits on one device.
- **Data parallelism:** replicate the full model on each GPU, split the batch across them, and synchronize gradients with an **all-reduce** (average), so every replica stays identical. Scales throughput when the model fits on one device.
- **Model / tensor / pipeline parallelism:** split the model itself when parameters, activations, or layers do not fit &mdash; tensor parallelism splits individual weight matrices (Shoeybi et al., 2019, "Megatron-LM," arXiv:1909.08053; Narayanan et al., 2021, arXiv:2104.04473); pipeline parallelism puts different layers on different GPUs (Huang et al., 2019, "GPipe," arXiv:1811.06965). Large runs combine all of these.
- **Fully sharded data parallelism (FSDP):** keeps data parallelism's structure (every GPU sees a different slice of the batch) but shards the parameters, gradients, and optimizer state across GPUs instead of replicating them. Each layer's full weights are rebuilt with an **all-gather** just before that layer computes and freed immediately after; gradients are **reduce-scattered** so each GPU keeps only its shard. This is the ZeRO idea (Rajbhandari et al., 2020, "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models," arXiv:1910.02054), productized as PyTorch FSDP (Zhao et al., 2023, arXiv:2304.11277). Memory per GPU drops roughly by the number of GPUs, at the cost of extra per-layer communication.
- Frontier runs can use thousands of GPUs for months, so networking, storage, and scheduling become first-order concerns, and **checkpoint-and-resume** is mandatory because hardware failures are expected. Implementation details are Module 9.
- **Manim animation (`data-parallel`):** the model is replicated across four GPUs, the batch splits into shards that flow to each GPU, each computes a local gradient, and an all-reduce hub averages them and broadcasts the result back so every replica updates identically.
- **Manim animation (`tensor-parallel`):** one layer's weight matrix $W$ in $y = xW$ is split column-wise into $W_1$ on GPU 0 and $W_2$ on GPU 1; both GPUs receive the same input $x$, each computes its half of the output, and an all-gather concatenates the halves into the full $y$. Column-wise splitting needs no communication before the multiply; the gather happens at every layer of every step (Shoeybi et al., 2019).
- **Manim animation (`fsdp`):** four GPUs first hold full copies of a four-layer model (plain data parallelism); FSDP then shards every layer so GPU $k$ permanently stores only slice $k$ (a quarter of the parameters); the batch then splits into per-GPU shards exactly as in data parallelism &mdash; the visual contrast with tensor parallelism, where every GPU received the *same* input; to compute layer 1 the GPUs all-gather its full weights just in time; after use the full weights are freed and gradients are reduce-scattered back into per-GPU shards (Rajbhandari et al., 2020; Zhao et al., 2023). Closing note on screen: each GPU runs the full model on its own data &mdash; only the storage is sharded.

## i. From Base Model to Assistant

- Pretraining yields a model that **continues text**, not one that reliably follows intent. A base model will complete a chat transcript (inventing both sides), imitate a document, write code, or continue harmful text, because all of these are patterns in its training distribution.
- **Side quest, base vs assistant:** the same prompt "What is the capital of France?" framed as a document is, to a base model, likely followed by *more questions* (as on a worksheet); an assistant model treats it as a request and answers. Same knowledge, different behavior. The slide links the Hugging Face model page for Qwen3-4B-Base (<https://huggingface.co/Qwen/Qwen3-4B-Base>), whose Inference Providers widget does raw completion with a base model, against HuggingChat (<https://huggingface.co/chat>) for the instruction-tuned side; both require a free Hugging Face account to run.
- **Instruction finetuning** continues training on prompts paired with desired responses, shifting the data distribution so a prompt is something to satisfy. **Preference optimization and reinforcement learning** further shape helpfulness, honesty, refusal behavior, and tool use. The optimization machinery is familiar; the data and behavioral target change. This is the handoff to Module 6.

## Exercise: Pretraining NanoGPT

- The student implements the pretraining loop around a provided tiny decoder-only model (2 layers, 4 heads, width 64, context 32; 106,304 parameters, no dropout) trained on the public-domain tiny-Shakespeare corpus (~1.1M characters, 65-character vocabulary). Steps: encode text, train/validation split, build shifted `(x, y)` batches, cross-entropy loss, one optimizer step (zero-grad, backprop; clipping and the step are provided), multi-batch loss estimation, converting the loss to perplexity and bits per token (`loss_to_perplexity_and_bits`), and autoregressive sampling. The warmup + cosine learning-rate schedule (`lr_at_step`) is provided in full rather than left as a blank. The run is sized for weak laptop CPUs: 800 steps of 32 sequences x 32 characters is about 0.8M tokens, less than one pass over the 1.0M-token training split, so dropout is off and the whole solution run finishes in well under a minute.
- The runner tags each step CORRECT, INCORRECT, or INCOMPLETE on its header line. After the step's output it runs that step's tests from `tests/` (one file per step), which call the student's function on tiny hand-made tensors with known answers: `F.cross_entropy` against a hand-written `-log_softmax`, targets equal to the inputs shifted by one, perplexity $= e^{\text{loss}}$ and bits $= \text{loss}/\ln 2$, and seeded sampling at temperature 0.01 matching the argmax. Only Step 7 trains the real model. It prints a per-checkpoint loss table and saves a loss-curve image. Step 8 converts the validation loss before and after training into perplexity and bits per token, and Step 10 prints a sample from the model before and after training. A `--overfit` mode trains on one fixed batch until the loss craters (4.18 -> 0.01 in 100 steps in the captured run), confirming the loop is wired correctly.
- All numbers and samples shown in the slides are captured from the solution run, per the course rule that sample outputs must be real.
- References: Karpathy's nanoGPT (<https://github.com/karpathy/nanoGPT>) and build-nanogpt (<https://github.com/karpathy/build-nanogpt>) walkthroughs informed the exercise design.

## References

- Radford et al., "Improving Language Understanding by Generative Pre-Training" (2018); Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2, 2019).
- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," arXiv:1810.04805.
- Brown et al., "Language Models are Few-Shot Learners," arXiv:2005.14165.
- Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (T5), arXiv:1910.10683.
- Kaplan et al., "Scaling Laws for Neural Language Models," arXiv:2001.08361.
- Hoffmann et al., "Training Compute-Optimal Large Language Models" (Chinchilla), arXiv:2203.15556.
- BigScience Workshop, "BLOOM: A 176B-Parameter Open-Access Multilingual Language Model," arXiv:2211.05100; training logs at <https://huggingface.co/bigscience/tr11-176B-logs> and <https://huggingface.co/bigscience/tr8-104B-logs>.
- Dubey et al., "The Llama 3 Herd of Models," arXiv:2407.21783.
- Sardana et al., "Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws," arXiv:2401.00448. The model behind the `servingPlanner` widget.
- Besiroglu et al., "Chinchilla Scaling: A Replication Attempt," arXiv:2404.10102. Corrects the parametric-fit coefficients reported in Hoffmann et al.; the corrected values are the ones used by the `scalingPlanner` widget.
- Touvron et al., "LLaMA: Open and Efficient Foundation Language Models," arXiv:2302.13971.
- Gao et al., "The Pile: An 800GB Dataset of Diverse Text for Language Modeling," arXiv:2101.00027.
- Lee et al., "Deduplicating Training Data Makes Language Models Better," arXiv:2107.06499.
- Del&eacute;tang et al., "Language Modeling Is Compression," arXiv:2309.10668.
- Wei et al., "Emergent Abilities of Large Language Models," arXiv:2206.07682.
- Schaeffer et al., "Are Emergent Abilities of Large Language Models a Mirage?", arXiv:2304.15004.
- DeepSeek-AI, "DeepSeek-V3 Technical Report," arXiv:2412.19437.
- Micikevicius et al., "Mixed Precision Training," arXiv:1710.03740.
- Kalamkar et al., "A Study of BFLOAT16 for Deep Learning Training," arXiv:1905.12322.
- NVIDIA A100 Tensor Core GPU datasheet.
- Kingma and Ba, "Adam: A Method for Stochastic Optimization," arXiv:1412.6980.
- Pascanu, Mikolov, and Bengio, "On the difficulty of training recurrent neural networks" (gradient clipping), arXiv:1211.5063.
- Loshchilov and Hutter, "Decoupled Weight Decay Regularization" (AdamW), arXiv:1711.05101; "SGDR: Stochastic Gradient Descent with Warm Restarts," arXiv:1608.03983.
- Huang et al., "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism," arXiv:1811.06965.
- Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism," arXiv:1909.08053.
- Narayanan et al., "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM," arXiv:2104.04473.
- Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models," arXiv:1910.02054.
- Zhao et al., "PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel," arXiv:2304.11277.
