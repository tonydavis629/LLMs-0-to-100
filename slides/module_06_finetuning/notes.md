# Module 6: Finetuning — Lecture Notes

These notes give an explanation and a citation for every claim on the slides, map
the two equations (masked cross-entropy, LoRA update) to the visuals they appear
on, and record the historical context. Module 6 owns the **supervised finetuning
(SFT)** stage; the reward model and RLHF are previewed here and taught in Module 7.

## Review

- Module 5 ended with a **base model**: next-token prediction at scale produced a
  model that continues text fluently but does not reliably follow user intent. A
  base model is a pattern-matcher over its training distribution; an assistant
  infers and satisfies intent. (See Module 5i, "From Base Model to Assistant.")
- The optimization machinery is unchanged from Modules 2 and 5: AdamW (Loshchilov
  and Hutter, 2017, arXiv:1711.05101), cross-entropy, backpropagation, batching,
  gradient clipping. What changes in finetuning is the **data distribution** and
  the **behavioral target**, plus a loss mask and a much smaller learning rate.
- The slide pairs this with the Module 5 handoff framing: pretraining built the
  engine; finetuning teaches it to drive.

## a. What Finetuning Is

### Definition and transfer learning
- Finetuning is continuing to train a pretrained model on new, narrower data to
  specialize behavior or knowledge. It is an instance of **transfer learning**: the
  latent capability learned in pretraining is inherited, so finetuning only has to
  teach format and behavior, not language itself. This is why pretraining costs
  weeks on thousands of accelerators while finetuning costs hours on modest
  hardware.
- Pretraining learns the structure of language from billions of tokens; finetuning
  nudges an already-capable model toward a target distribution with thousands to
  millions of examples.

### The same algorithm, almost
- The forward pass, the cross-entropy loss, and backpropagation are identical to
  pretraining. Three dials move: a much **smaller learning rate**, **far fewer
  steps / less data** (often 1-3 epochs), and the objective is **masked** to the
  response (section b). Finetuning is the same algorithm pointed at a new
  distribution, not a new algorithm.

### A taxonomy
- **Continued pretraining:** more next-token prediction on domain text (medicine,
  code, a new language) to shift knowledge.
- **Supervised finetuning (SFT) / instruction tuning:** teach the model to follow
  prompts from prompt-response pairs. The focus of this module.
- **Preference optimization / RL:** learn from comparisons of better vs worse
  responses (Module 7).

### Catastrophic forgetting
- Pushing the weights too hard on narrow data overwrites broadly useful features
  with narrow ones, so the model loses general capability it had after pretraining
  (McCloskey and Cohen, 1989, introduced the term for connectionist networks;
  French, 1999, surveys it). This single failure mode motivates the small learning
  rate, the few epochs, and the parameter-efficient methods in section d, which
  freeze the base entirely.

## b. Supervised Finetuning and Instruction Tuning

### Prompt-response pairs
- The data unit changes from a raw token stream to **prompt-response pairs**: an
  instruction paired with the response we want. We train the model to produce the
  response given the prompt.

### Chat templates and special tokens
- A conversation has roles (system, user, assistant), but the model only ever sees
  a flat sequence of token ids (callback to Module 4 tokenization). A **chat
  template** serializes the roles into that flat stream using **special tokens** &mdash;
  dedicated ids that mark structure, in the style of ChatML
  (huggingface.co/docs/transformers/chat_templating).
- The markers are **atomic**: `<|user|>` is one token id with its own embedding,
  not the seven characters that spell it. The exercise adds four such tokens
  (`<|user|>`, `<|assistant|>`, `<|end|>`, `<|pad|>`) to the Module 5 65-character
  vocabulary, giving 69 tokens.
- **Manim animation (`chat-template`):** two raw turns (a user "hi" and an assistant
  "HI") are wrapped with role markers, then flattened into one token stream
  `<|user|> h i <|end|> <|assistant|> H I <|end|>`; the special tokens are then
  highlighted (orange) versus the text tokens (blue). It makes serialization
  concrete: the model only ever sees a flat list of ids.
- At generation time, we build the same template but stop right after the assistant
  marker, leaving the response for the model to fill in (the exercise's
  `build_generation_prompt`). The assistant has learned that the assistant marker
  means "your turn to answer."

### Loss masking
- The example holds both prompt and response, but we only want to teach the model
  to produce the **response**. So we compute cross-entropy **only over the response
  tokens** and set the target at every prompt position to a sentinel (`-100`), which
  `F.cross_entropy(..., ignore_index=-100)` skips. If we trained on the whole
  sequence, the model would also learn to generate the user's turns &mdash; at
  inference it would hallucinate its own prompts (the side quest below, reused as
  exercise extra credit).
- It is the same cross-entropy objective from Module 5, restricted to the
  completion:
$$\mathcal{L} = -\frac{1}{|R|}\sum_{t \in R}\log p_\theta\left(x_t \mid x_{<t}\right)$$
  where $R$ is the set of response-token positions, $|R|$ their count, and $x_{<t}$
  the full prefix (the model still reads the prompt as context; the prompt is
  conditioning, not a target). For multi-turn dialogue, mask every assistant turn
  so one sequence trains several responses at once.
- **Implementation detail (exercise `build_targets`):** target $t$ is the token
  position $t$ should predict, i.e. $\text{ids}[t+1]$. The first response token is
  predicted at the assistant-marker position (`prompt_span - 1`), so positions
  $0\ldots\text{prompt\_span}-2$ are masked, the response targets are
  $\text{ids}[\text{prompt\_span}:]$, and the final position is `-100`. Masking one
  position too many (an off-by-one that drops the first response token's
  supervision) makes the model emit a garbage first character &mdash; the model never
  learns what to produce right after the assistant marker.
- **Manim animation (`loss-mask`):** the eight-token row, each position drawing an
  arrow to its next-token target; the prompt-prediction targets (and the final
  position) turn to `-100` in red while the response targets (H, I, `<|end|>`) turn
  green and are braced as "response tokens R"; the loss then averages over the green
  terms only. This matches `build_targets` exactly.

### Instruction tuning generalizes
- Training on **many tasks phrased as instructions** teaches the format of
  instruction-following itself, which transfers to instructions never seen in
  finetuning: FLAN (Wei et al., 2021, arXiv:2109.01652) instruction-tuned across
  60+ tasks and showed strong zero-shot generalization; T0 (Sanh et al., 2021,
  arXiv:2110.08207) showed multitask prompted training enables zero-shot task
  generalization; Super-NaturalInstructions (Wang et al., 2022, arXiv:2204.07705)
  scaled to 1,600+ tasks.
- **Notable figure: Jason Wei and the FLAN team** &mdash; reframed finetuning from
  "teach this task" to "teach the model to follow instructions," a direct ancestor
  of the instruction-following behavior in modern chat assistants.

- **Interactive widget (`:::interactive widget="lossMask"`):** a two-turn conversation rendered in a chat template, with each whitespace-delimited piece shown as one token. Green tokens contribute to the loss, grey ones are set to `-100` and skipped by `ignore_index`. The three modes correspond to real choices: masking every assistant turn (31 tokens in the sequence, 12 supervised), training only the final turn (which discards the earlier response's supervision from the same forward pass), and no mask at all (100% supervised, which teaches the model to generate user turns and system prompts too). The counts are computed from the rendered sequence, not hardcoded.

## c. InstructGPT, the Birth of the Assistant

### The three-stage recipe
- InstructGPT (Ouyang et al., 2022, arXiv:2203.02155) has three stages: (1)
  supervised finetuning on human-written demonstrations; (2) a reward model trained
  on human preference rankings; (3) reinforcement learning from human feedback with
  PPO (Schulman et al., 2017, arXiv:1707.06347). Module 6 owns stage 1; stages 2-3
  are the spine of Module 7 &mdash; the same handoff seam Module 5 made to Module 6.

### The headline result
- A **1.3B**-parameter InstructGPT model was preferred by human raters over the
  **175B** GPT-3 (Brown et al., 2020, arXiv:2005.14165), despite being more than
  100x smaller (Ouyang et al., 2022). Alignment to intent beat raw scale. This is
  the direct lineage of ChatGPT: a capable base model is necessary but not
  sufficient; the finetuning is what makes it an assistant.

### HHH
- The behavioral target is **helpful, honest, and harmless** (Askell et al., 2021,
  arXiv:2112.00861). Finetuning is where these stop being slogans and become a
  data-collection and training problem: which demonstrations you show and which
  behaviors you reward.
- **Notable figures: Long Ouyang and the InstructGPT team** (turned a base model
  into an instruction-follower with the SFT-plus-RLHF recipe; showed 1.3B beats
  175B at following intent) and **Amanda Askell and collaborators** (articulated
  the HHH framing, making alignment a concrete engineering objective).

### Side quest: the alignment tax
- Finetuning for helpfulness and safety can reduce raw benchmark performance, a
  measurable cost (discussed in Ouyang et al., 2022, who mitigate it by mixing
  pretraining gradients back into the RL stage, "PPO-ptx"). Mitigations include
  rehearsing pretraining data during finetuning and using parameter-efficient
  methods that perturb the base less.

## d. Parameter-Efficient Finetuning and LoRA

### The cost of full finetuning
- Full finetuning updates all $N$ parameters, must hold a full optimizer state
  (AdamW keeps two moment buffers per parameter, roughly tripling training memory;
  callback to Module 5d), and produces a full-size checkpoint per task. For a 70B
  model this is enormous and a separate copy per task is untenable.

### The LoRA insight
- Hu et al. (2021, arXiv:2106.09685) observed that the **update** a weight matrix
  undergoes during finetuning has low intrinsic rank. So freeze the pretrained
  weight $W$ and learn only a low-rank correction:
$$W' = W + \frac{\alpha}{r} BA, \qquad B \in \mathbb{R}^{d \times r}, \quad A \in \mathbb{R}^{r \times d}, \quad r \ll d$$
  Only $A$ and $B$ are trained &mdash; often well under 1% of parameters; the frozen
  base $W$ is shared across tasks. $B$ is initialized to **zero**, so the adapter
  starts as a no-op and finetuning begins exactly at the base model. The fixed
  scale $\alpha/r$ lets you change $r$ without retuning the learning rate. Because
  the base is frozen, the optimizer state is tiny, and the limited capacity also
  means less catastrophic forgetting.
- **Manim animation (`lora`):** a $d\times d$ grid $W$ is frozen; a tall $B$
  ($d\times r$) and wide $A$ ($r\times d$) are added; $r \ll d$ is annotated with
  the parameter comparison (full $d^2$ vs LoRA $2dr$; at $d=4096, r=8$ that is
  16.8M vs 65K); finally $BA$ folds back into $W$, the same $d\times d$ matrix, for
  zero added latency. This is the one genuine spatial visualization in the module.

### Merge and swap
- At inference you can **merge** $W + \frac{\alpha}{r}BA$ into one matrix of the
  same shape as $W$, so the model runs at exactly base-model speed (zero added
  latency &mdash; the exercise's merge-equality check verifies the logits are
  identical to ~1e-5). Or keep the adapter separate and **swap** adapters over one
  frozen base to switch behaviors on the fly. In practice, checkpoints shrink from
  gigabytes to megabytes and finetuning fits on consumer hardware.

### QLoRA and the PEFT family
- **QLoRA** (Dettmers et al., 2023, arXiv:2305.14314) quantizes the frozen base to
  4-bit and trains LoRA adapters on top, finetuning a 65B model on a single GPU.
  (Quantization itself: Module 9.)
- The broader PEFT family, named not detailed (in the style of Module 4e): adapters
  (Houlsby et al., 2019, arXiv:1902.00751), prefix tuning (Li and Liang, 2021,
  arXiv:2101.00190) and prompt tuning (Lester et al., 2021, arXiv:2104.08691), and
  (IA)^3 (Liu et al., 2022, arXiv:2205.05638). The HuggingFace PEFT library
  (huggingface.co/docs/peft) implements these.
- **The honest trade-off:** PEFT is slightly less expressive than full finetuning,
  but for most instruction tuning the gap is small and the cost savings are large.
- **Notable figures: Edward Hu and the LoRA team** (Microsoft; introduced low-rank
  adaptation) and **Tim Dettmers** (drove practical quantization and QLoRA, putting
  finetuning of very large models on a single consumer GPU).

### Side quest: adapters as diffs for weights
- A LoRA adapter is a **diff** against the frozen base &mdash; like a code patch, but
  for model weights. One frozen base plus many small, swappable adapters means
  task-specific behaviors are megabyte-sized files you load and unload; model hubs
  host thousands of community adapters for one popular base.

- **Interactive widget (`:::interactive widget="loraCalculator"`):** draws one attention projection to scale &mdash; $W$ as a $d \times d$ square, $B$ as $d \times r$ and $A$ as $r \times d$ at the same pixels-per-unit, so the rank appears as an actual sliver. The scale $\alpha/r$ from the previous slide is carried through: it is shown on the update path, in the live equation $W' = W + \frac{\alpha}{r} BA$, and as its own readout. It multiplies the low-rank product only, never $W$, and it changes no parameter count &mdash; holding $\alpha$ fixed while raising $r$ shrinks the scale, which is what keeps the update magnitude in range and lets one learning rate carry across ranks. Counts assume adapters on the four attention projections of every layer: frozen $4 L d^2$, trainable $4 L \cdot 2 d r$. Optimizer memory assumes AdamW's two fp32 moments per trainable parameter (8 bytes), and the checkpoint figure assumes bf16 adapter weights (2 bytes). At $d = 4096$, $L = 32$, $r = 8$ that is 2.1B frozen against 8.4M trainable, 0.39%, and a 16.8 MB adapter against 17.2 GB of full-finetuning optimizer state.

## e. The Craft: Data and Evaluation

### Data quality dominates quantity
- A small set of high-quality, diverse demonstrations beats a large noisy one.
  **LIMA** (Zhou et al., 2023, arXiv:2305.11206), "Less Is More for Alignment,"
  finetuned on roughly 1,000 carefully curated examples and still produced a strong
  assistant. The base model already has the knowledge; alignment mostly teaches
  format and style, which a small clean set can convey.

### Where SFT data comes from
- **Human-written** demonstrations (highest quality, slowest). **Distillation** from
  a stronger model: Self-Instruct (Wang et al., 2022, arXiv:2212.10560) and Stanford
  Alpaca (Taori et al., 2023, crfm.stanford.edu/2023/03/13/alpaca.html) generate
  instructions and responses from a teacher. **Filtered real conversations**.

### Hyperparameters and evaluation
- What matters: a small learning rate, only 1-3 epochs (more overfits and forgets),
  and a held-out set (with small data the model memorizes fast). Evaluation is hard:
  unlike pretraining's clean validation loss, "did it become a better assistant?"
  needs preference judgments, downstream benchmarks, and human or model graders
  (forward reference to Module 11).

### Side quest: the superficial alignment hypothesis
- LIMA's claim: a base model already contains the knowledge and skills, and
  alignment mostly teaches format, style, and which subset of behaviors to surface.
  As a critical-thinking exercise (in the spirit of Module 5's "emergence: real or
  mirage?"): if 1,000 examples can align a model that saw trillions of tokens, how
  much is finetuning teaching versus revealing?

### Side quest: synthetic data and model collapse
- Alpaca and Self-Instruct generate finetuning data from a stronger model, which
  raises quality, licensing, and **model-collapse** questions: training repeatedly
  on model-generated text narrows the distribution toward the model's own quirks
  (Shumailov et al., 2023/2024, "The Curse of Recursion" / "AI models collapse when
  trained on recursively generated data," Nature). Ties back to Module 5's
  data-wall side quest.

## f. The Limits of SFT, and the Handoff to Module 7

- SFT can only **imitate** the demonstrations it was shown: it learns "produce
  responses like these," not "this response is better than that one." It has no
  clean way to use **negative signal** &mdash; every demonstration is positive.
- **Exposure bias / distribution shift:** SFT trains on ground-truth prefixes, but
  at generation time the model must continue its own (imperfect) outputs, so early
  mistakes compound (Ranzato et al., 2015, arXiv:1511.06732, framed this for
  sequence models).
- These limits are exactly what preference optimization and RL address: RLHF
  (Christiano et al., 2017, arXiv:1706.03741; Ouyang et al., 2022), DPO (Rafailov et
  al., 2023, arXiv:2305.18290), and GRPO (Shao et al., 2024, arXiv:2402.03300,
  DeepSeekMath) &mdash; the subject of Module 7. Same handoff shape as Module 5 to 6:
  SFT shaped the format; RL shapes the preferences.

## Exercise: Finetune NanoGPT into an Instruct Model

- The bundled Module 5 base checkpoint is finetuned (frozen, never re-pretrained)
  into a toy instruct model via from-scratch LoRA (pure PyTorch, no `peft`). The
  ten student steps are: format the chat template, build the masked targets,
  masked cross-entropy, build the optimizer over adapters, the LoRA forward delta,
  freeze a base parameter, one SFT step, count trainable parameters, build the
  generation prompt, and merge the LoRA weight.
- Captured run (seed 1337, rank 8, alpha 32, 1000 steps, ~350 toy pairs): the same
  prompt `uppercase: hello` flips from `'ers\nIn the father '` (base continues
  Shakespeare-style text, ignoring the instruction) to `'HELLO'` (finetuned answers
  it); 65,536 of 884,096 parameters are trainable (7.41%); the masked loss falls
  from ~6.1 to ~0.30; and the merge-equality check passes (max logit difference
  ~1e-5). The "reverse" task is the hardest and is honestly presented as a minority
  case &mdash; a tiny model may not nail it.
- Extra credit: full-FT vs LoRA (skip the freeze, compare counts/quality); vary the
  rank $r$; catastrophic-forgetting probe (a raw base-style prompt after
  finetuning); loss-mask ablation (train without the mask and watch the model
  hallucinate prompts).

## References

- Ouyang et al., 2022, "Training Language Models to Follow Instructions with Human Feedback" (InstructGPT), arXiv:2203.02155.
- Hu et al., 2021, "LoRA: Low-Rank Adaptation of Large Language Models," arXiv:2106.09685.
- Dettmers et al., 2023, "QLoRA: Efficient Finetuning of Quantized LLMs," arXiv:2305.14314.
- Wei et al., 2021, "Finetuned Language Models Are Zero-Shot Learners" (FLAN), arXiv:2109.01652.
- Sanh et al., 2021, "Multitask Prompted Training Enables Zero-Shot Task Generalization" (T0), arXiv:2110.08207.
- Wang et al., 2022, "Super-NaturalInstructions," arXiv:2204.07705.
- Zhou et al., 2023, "LIMA: Less Is More for Alignment," arXiv:2305.11206.
- Wang et al., 2022, "Self-Instruct," arXiv:2212.10560.
- Taori et al., 2023, "Stanford Alpaca," crfm.stanford.edu/2023/03/13/alpaca.html.
- Askell et al., 2021, "A General Language Assistant as a Laboratory for Alignment" (HHH), arXiv:2112.00861.
- Houlsby et al., 2019, "Parameter-Efficient Transfer Learning for NLP" (adapters), arXiv:1902.00751.
- Li and Liang, 2021, "Prefix-Tuning," arXiv:2101.00190; Lester et al., 2021, "The Power of Scale for Parameter-Efficient Prompt Tuning," arXiv:2104.08691.
- Liu et al., 2022, "Few-Shot Parameter-Efficient Fine-Tuning Is Better and Cheaper than In-Context Learning" ((IA)^3), arXiv:2205.05638.
- Rafailov et al., 2023, "Direct Preference Optimization," arXiv:2305.18290; Shao et al., 2024, "DeepSeekMath" (GRPO), arXiv:2402.03300; Christiano et al., 2017, "Deep RL from Human Preferences," arXiv:1706.03741.
- Shumailov et al., 2024, "AI models collapse when trained on recursively generated data," Nature 631.
- HuggingFace PEFT library, huggingface.co/docs/peft; chat templating, huggingface.co/docs/transformers/chat_templating.
- S. Raschka, "Practical Tips for Finetuning LLMs," magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms.
