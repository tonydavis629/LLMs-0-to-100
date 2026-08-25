# Module 9: Evaluation and Benchmarking — Lecture Notes

These notes give an explanation and a citation for every major claim on the
slides, map the equations to the visuals they appear on, and record the
historical context. Module 9 introduces no new architecture and trains no new
model. It is about **measurement**: given a checkpoint, how do you find out
whether it is any good, and how does the industry report that publicly?

The organizing idea is that each training stage produces a different kind of
model, so each stage admits a different kind of measurement. A base model is
measured on how well it predicts text and what it knows; an instruction-tuned
model on whether it does what was asked; a reasoning model on hard problems with
checkable answers. Multimodal models and agents each add their own machinery.

## Review

- Every previous training module ended with a number, introduced when it was
  needed rather than as part of a general theory of evaluation:
  - **Module 5** tracked held-out loss and perplexity during pretraining.
  - **Module 6** measured task accuracy after supervised finetuning.
  - **Module 7** plotted a reward curve and before/after accuracy on the reverse
    task.
  - **Module 8** noted that exact-match scoring becomes brittle once answers are
    about images.
- What carries over: the train/test split from Module 2 (a score on training data
  measures memorization), cross-entropy and perplexity from Module 5, the chat
  template from Module 6, and the verifiable reward from Module 7 — the last of
  these reused here as a *scoring* function rather than a training signal.
- What is new: the standard public benchmarks, methods for scoring answers with no
  single right answer, the reasons the same model gets different scores at
  different labs, and an explicit account of what a benchmark number does not tell
  you.

## a. Evaluations, benchmarks, and leaderboards

### Three words that get used interchangeably

- An **evaluation** is a dataset plus a scoring rule: run the model on the cases,
  score the outputs, report a number.
- A **benchmark** is an evaluation that is published and reused, so different labs
  can compare models on the same cases under the same rules.
- A **leaderboard** is a public table of benchmark results. It is a summary of
  evaluations, not an evaluation itself. The Hugging Face Open LLM Leaderboard is
  the best-known example for open-weight models; it is a presentation layer over
  EleutherAI's `lm-evaluation-harness`.

### The two requirements

- **Held-out data.** If test cases appear in training, the score measures
  memorization. This is the same train/test discipline introduced in Module 2 and
  used for the held-out loss curve in Module 5. Section g treats the hard version
  of this problem, where the training set is a large fraction of the public web.
- **A scoring rule the field agrees on.** "Accuracy" is undefined until one
  specifies how an answer is extracted from generated text and what counts as a
  match. SQuAD's release (Rajpurkar et al., 2016, arXiv:1606.05250) is the usual
  reference point for shipping a benchmark *with its official scoring script*,
  precisely so that the rule is not reinvented per paper.

### The three ways to score an output

1. **Automatic and exact.** Compare against a key, execute code against tests,
   parse and check a number. Cheap and objective; only applicable when there is a
   right answer.
2. **Human judgment.** People read outputs and rate or rank them. Expensive and
   slow, and still the ground truth for open-ended work.
3. **Model judgment (LLM-as-a-judge).** A strong model grades outputs against a
   rubric. Cheap and scalable, but a proxy that must be validated against humans
   (Zheng et al., 2023, arXiv:2306.05685).

### The two scoring shapes

This distinction recurs throughout the module and is measured directly in the
exercise:

- **Multiple choice, scored by likelihood.** Present the question and each
  candidate answer; select the option to which the model assigns the highest
  probability. Nothing is generated, so the method works on base models that
  cannot follow instructions. Used by MMLU, HellaSwag, ARC, WinoGrande.
- **Free generation, scored by a checker.** Let the model write an answer, then
  extract and check it. Used by GSM8K, HumanEval, IFEval, and most instruct
  benchmarks.

The exercise demonstrates that these two shapes can disagree completely about the
same pair of models: the multiple-choice benchmark scores both checkpoints 16/16
while the generation-based suite shows one of them collapsing.

### How likelihood scoring works mechanically

The `two-shapes` table describes this in words; the underlying quantity is the
score assigned to option $i$ with answer tokens $a_i$, given question $q$:

$$s_i = \sum_{t=1}^{|a_i|} \log p_\theta\big(a_{i,t} \mid q, a_{i,<t}\big)$$

These are the same per-token log-probabilities the training objective averages
into a loss (Module 5). The only difference is what is done with them: training
averages and minimizes, benchmarking sums per option and takes the argmax. A
four-option question therefore costs four forward passes over the same prompt
with four different continuations, and involves no sampling, temperature,
decoding, or answer extraction — which is exactly why it runs on models that
cannot follow an instruction. The slides state the rule in words rather than in
symbols; the equation lives here and in step 7 of the exercise.

### Length normalization in likelihood scoring

The raw sum is not usable directly. Each additional token contributes a further
$\log p < 0$, so longer options accumulate more negative totals independently of
correctness. Real harnesses offer several variants — normalize by token count,
normalize by byte or character count, or condition on the answer's unconditional
likelihood (the "answer context" normalization used by the original MMLU and by
`lm-evaluation-harness`) — and the choice moves reported MMLU numbers by several
points on the same weights. This is one of the concrete mechanisms behind the
protocol-dependence discussed in section g. Step 7 of the exercise implements the
token-count-normalized version.

### Where the practice came from

- **ImageNet** (Deng et al., 2009; ILSVRC ran 2010–2017) showed that a large
  shared test set with a public leaderboard can organize an entire research field.
  The AlexNet result discussed in Module 1 (Krizhevsky, Sutskever, and Hinton,
  2012) is an ImageNet result; without the shared benchmark it would have been one
  laboratory's claim rather than a measurable jump.
- **SQuAD** (2016) brought the pattern to NLP, together with the normalization +
  exact-match + token-F1 scoring script that this module's exercise reimplements.
- **GLUE** (Wang et al., 2018, arXiv:1804.07461) bundled nine tasks into a single
  score with a held-out test server. Human performance on GLUE was surpassed
  within roughly a year, which is why **SuperGLUE** (Wang et al., 2019,
  arXiv:1905.00537) was built; models exceeded its human baseline by early 2021.

### Notable figures introduced here

- **Fei-Fei Li** — built ImageNet and its competition. The slide's framing is that
  ImageNet was not an algorithm but a *measurement instrument*.
- **Samuel Bowman, Alex Wang, and collaborators** — GLUE and SuperGLUE, the
  template every later LLM benchmark suite follows.
- **Pranav Rajpurkar and collaborators** — SQuAD, and the scoring script that made
  exact match and token F1 the default for short free-form answers.

## b. Evaluating a pretrained model

### Why the question is constrained

A base model does not reliably follow instructions (the behavioral gap Module 6
exists to close), so most questions cannot be asked directly. Two things remain
measurable: how well the model predicts text, and what it knows.

### Loss and perplexity

The slide `base-perplexity` shows the Module 5 equations unchanged:

$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t \mid x_{<t}), \qquad \mathrm{PPL} = \exp(\mathcal{L})$$

- $\mathcal{L}$ is the average negative log-likelihood per token, in nats, over
  held-out text.
- $\mathrm{PPL}$ is its exponential, interpretable as the model's average
  **branching factor**: roughly how many equally likely tokens it is choosing
  among at each position. A perplexity of 1 corresponds to a model that is never
  surprised; a perplexity equal to the vocabulary size corresponds to uniform
  guessing.
- Perplexity requires no labels, which is why it is the metric available at every
  point in training.

### Bits per byte

Perplexity is per *token*, so it is not comparable across tokenizers: a model with
a larger vocabulary makes fewer, larger predictions over the same text. The
standard fix, used in the Pile and GPT-3-era reports, is **bits per byte**:

$$\mathrm{BPB} = \frac{\mathcal{L}_{\text{total}}}{\ln(2) \cdot \text{number of bytes}}$$

The worked figures on the slide `base-bpb` are constructed to isolate the effect.
Two models each spend a total of 300 nats on the same 1,000-byte passage, so by
construction they model it equally well:

| Model | Vocabulary | Tokens | $\mathcal{L}_{\text{total}}$ | PPL | BPB |
| --- | --- | --- | --- | --- | --- |
| X | 32,000 | 250 | 300 nats | $e^{300/250} = 3.32$ | 0.433 |
| Y | 128,000 | 200 | 300 nats | $e^{300/200} = 4.48$ | 0.433 |

Perplexity differs by 35%; bits per byte is identical, because
$300 / (\ln 2 \cdot 1000) = 0.433$ either way. Note the direction: the *larger*
vocabulary reports the *worse* perplexity, because the same total surprisal is
spread over fewer, individually harder predictions.

#### Why the units are not circular

A recurring and reasonable student objection is that bits and bytes both measure
information, so the ratio looks vacuous — analogous to quarts per gallon. It is
not, because the two quantities are not being used as competing information
measures. The slide `base-bpb-units` separates them:

- The **byte count** is the UTF-8 size of the held-out passage: a fixed physical
  property of the *data*, invariant to the model, the tokenizer, and the
  laboratory. It appears in the denominator because it is the one quantity all
  parties compute identically.
- The **bit count** is the model's total surprisal on that passage, converted from
  nats by dividing by $\ln 2$: a property of the *model*.

The ratio is therefore a cost per unit of fixed content — structurally analogous
to dollars per mile, not to quarts per gallon. (A byte would carry a full 8 bits
of information only if all 256 values were equiprobable, which natural text is
not.)

#### The compression reading

Because an uncompressed byte occupies 8 bits, bits per byte is directly a
compression rate: $\mathrm{BPB} \times \text{bytes}$ is the size in bits of the
passage encoded with the model as the entropy model (arithmetic coding attains
this bound to within a bit or two). The reference points on
`base-bpb-compression`:

| BPB | What achieves it |
| --- | --- |
| 8.0 | No model; store the raw bytes |
| ~4.1 | The character/byte histogram of English (order-0 entropy) |
| ~1.0 | Shannon's 1951 estimate for printed English, from human guessing experiments (he bracketed 0.6–1.3 bits per character) |
| ~0.7 | A competent modern LLM on general English text; the Pile and Chinchilla-era reports quote figures in roughly the 0.6–0.9 range depending on the subset |

This is the quantitative form of the claim made in Module 1 that prediction and
compression are the same operation (Shannon, 1948; Shannon, 1951).

#### When each metric is admissible

Same-family checkpoint comparisons can use perplexity; cross-family comparisons
should use bits per byte. The slide `base-bpb-rule` also makes the limiting point
that the exercise depends on: both checkpoints there share one tokenizer, so the
perplexity comparison is *valid* and still selects the *worse* model.
Tokenizer-comparability and predictive usefulness are independent problems, and
bits per byte addresses only the former.

### Perplexity is a weak predictor of usefulness

Perplexity rewards fluent continuation, not correct answers. The exercise makes
this concrete and slightly uncomfortable: the GRPO checkpoint has **lower**
perplexity on held-out Shakespeare (1599 versus 1986) than the instruct checkpoint
while scoring 33 points worse on the task suite. The slide states this before the
exercise so that students read the result as a lesson rather than as a bug.

### Knowledge and reasoning benchmarks

All of the following are scored by likelihood, so they run on base models:

- **MMLU** (Hendrycks et al., 2020, arXiv:2009.03300): 57 subjects of
  multiple-choice questions from elementary through professional level. For
  several years the default headline number for a base model.
- **HellaSwag** (Zellers et al., 2019, arXiv:1905.07830): adversarially filtered
  sentence completion.
- **ARC** (Clark et al., 2018, arXiv:1803.05457): grade-school science, split into
  Easy and Challenge.
- **WinoGrande** and **PIQA**: pronoun resolution and physical commonsense; part of
  the standard set reported in open-model releases.
- **TriviaQA** and **Natural Questions**: factual recall with short free-form
  answers, scored with exact match and F1.
- **GPQA** (Rein et al., 2023, arXiv:2311.12022): graduate-level science questions
  written to remain difficult even with web search, built after MMLU began
  saturating.

### Few-shot prompting is part of the protocol

Base models are run with `k` solved examples in the context so the model can infer
the answer format — the in-context learning behavior reported in the GPT-3 paper
(Brown et al., 2020, arXiv:2005.14165) and covered in Module 5. Scores move with
`k`, and conventions differ per benchmark (MMLU is conventionally 5-shot; some
leaderboards report ARC at 25-shot). A score without its shot count is not
reproducible.

### What pretraining evaluation is used for

Not shipping decisions: checkpoint selection during a run, comparison of data
mixtures, and confirmation that a scaling run is tracking the loss curve its
scaling law predicted (Kaplan et al., 2020, arXiv:2001.08361; Hoffmann et al.,
2022, arXiv:2203.15556, both introduced in Module 5).

### Notable figure introduced here

- **Dan Hendrycks** — MMLU and MATH, the two benchmarks that defined frontier
  model comparison for years, both deliberately built harder than contemporary
  models could handle.

## c. Evaluating an instruction-tuned model

### Two families of question

After SFT the model answers prompts, so questions can be asked directly. They
split into those with a right answer (scored automatically) and those without
(scored by comparison). Much of the practical work in modern evaluation consists
of moving questions from the second category into the first by making answers
**checkable**.

### Accuracy and the brittleness of exact match

$$\mathrm{accuracy} = \frac{\text{number correct}}{\text{number evaluated}}$$

Exact match is the most transparent metric available and the most brittle: `"4"`,
`"4."`, `" 4 "`, and `"The answer is 4."` are one answer and four strings. Every
benchmark therefore ships two pieces of machinery before any comparison happens:

- a **normalization step** — lowercase, strip punctuation and articles, collapse
  whitespace (the SQuAD convention, reimplemented as step 2 of the exercise); and
- an **answer-extraction rule** — take the text after `####` (GSM8K), take the last
  number, or take the contents of `\boxed{}` (MATH).

### Token-level precision, recall, and F1

Shown on slide `instruct-f1`:

$$\mathrm{precision} = \frac{TP}{TP+FP}, \qquad \mathrm{recall} = \frac{TP}{TP+FN}$$

$$F_1 = 2 \cdot \frac{\mathrm{precision}\cdot\mathrm{recall}}{\mathrm{precision}+\mathrm{recall}}$$

Applied at the token level: precision is the fraction of predicted tokens that
appear in the reference, recall the fraction of reference tokens that were
produced, and F1 their harmonic mean. The worked example on the slide — predicted
"it is bluu" against reference "it is blue" — scores 0.0 under exact match and
0.67 under F1, because two of three tokens overlap on both sides.

The counting detail that matters in implementation: the overlap is computed with a
multiset intersection (`Counter(pred) & Counter(ref)` in the exercise), so
repeating a word cannot inflate the numerator.

The vocabulary is inherited from information retrieval. **Karen Spärck Jones**
(1935–2007) is the figure the slides use for this lineage; her work on term
weighting and retrieval evaluation, including inverse document frequency (1972),
is where the precision/recall framing entered the surrounding literature.

### Checkable answers

Where an answer can be executed, execution beats string comparison: parse and
compare the number, validate JSON against a schema, run the generated function
against hidden tests, or check a programmatic constraint. This is exactly Module
7's verifiable reward, used for scoring rather than for training.

### The standard instruct benchmarks

- **GSM8K** (Cobbe et al., 2021, arXiv:2110.14168): grade-school math word
  problems with a single numeric answer after a `####` marker. The canonical
  checkable-answer benchmark.
- **MATH** (Hendrycks et al., 2021, arXiv:2103.03874): competition mathematics,
  with answers checked by symbolic equivalence rather than string match.
- **HumanEval** (Chen et al., 2021, arXiv:2107.03374) and **MBPP**: write a Python
  function from a docstring; scored by running hidden tests.
- **IFEval** (Zhou et al., 2023, arXiv:2311.07911): instructions with
  programmatically checkable constraints ("exactly three bullet points"), which
  isolates instruction-following from knowledge.
- **BBH** (Suzgun et al., 2022, arXiv:2210.09261) and **MMLU-Pro** (Wang et al.,
  2024, arXiv:2406.01574): harder successors adopted once the originals stopped
  separating models.

### pass@k

From the HumanEval paper, and shown on slide `instruct-passk`:

$$\mathrm{pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}} \quad \text{for } c \text{ correct out of } n \text{ samples}$$

The reading: draw `n` samples, observe `c` correct, and estimate the probability
that a randomly chosen subset of `k` contains at least one correct sample. The
complement $\binom{n-c}{k}/\binom{n}{k}$ is the probability that a `k`-subset is
drawn entirely from the wrong samples. When $n - c < k$ there are too few wrong
samples to fill such a subset and the estimate is exactly 1 — the guard clause
provided in step 8 of the exercise.

pass@1 is ordinary single-sample accuracy. Large-`k` pass@k asks a different
question: is the correct answer anywhere in the model's distribution? Section d
depends on the gap between them.

### Scoring answers with no single key

- **The overlap era.** BLEU (Papineni et al., 2002) for translation and ROUGE
  (Lin, 2004) for summarization score n-gram overlap against human reference
  texts. Both made twenty years of progress measurable, and both are still
  reported; both are also known to correlate weakly with human judgment on modern
  systems, because n-gram overlap cannot express whether an answer is true or
  useful.
- **The comparison era.** Show two answers to the same prompt and ask which is
  better. No reference text is required and the quantity measured is closer to
  what users care about.

### Chatbot Arena and human preference at scale

**Chatbot Arena** (LMSYS; Chiang et al., 2024, arXiv:2403.04132) collects
anonymous side-by-side comparisons from real users on real prompts and aggregates
them into an Elo-style rating. It became the industry's de facto public scoreboard
because it measures preference on the prompt distribution people actually send.

The caveat stated on the slide is important and empirically supported: preference
is not correctness. Longer, better-formatted, and more agreeable answers win
votes. This is the same length and sycophancy bias Module 7 identified as reward
hacking, now appearing on the measurement side.

### LLM-as-a-judge

**MT-Bench** and the LLM-as-a-judge methodology (Zheng et al., 2023,
arXiv:2306.05685) and **AlpacaEval** automate the comparison: a strong model
grades answers against a rubric or picks a winner between two. Agreement with
human preference is high enough to be useful, and the cost is negligible, which is
why the practice is now ubiquitous.

Known biases, all easy to demonstrate:

- **Position bias** — preferring whichever answer appeared first. Mitigated by
  running both orders (the "Swap the answers" side quest).
- **Verbosity bias** — preferring longer answers. AlpacaEval 2.0 introduced an
  explicit length control for this reason.
- **Self-preference** — favoring outputs resembling the judge's own style.
- **Competence ceiling** — a judge cannot reliably grade work it could not perform
  itself.

The practical rule on the slide: validate any judge against a few hundred
human-labeled examples from your own task before trusting its numbers.

## d. Evaluating a reasoning or RL-trained model

### A reward curve is not an evaluation

Module 7's climbing reward curve shows that the optimizer is working. It cannot
show that the ability generalizes, that other abilities survived, or that the
policy did not find a degenerate way to score, because the reward is exactly the
quantity being optimized and is measured on the training prompts.

### Benchmarks used for reasoning models

The RLVR setup from Module 7, used as a test set rather than a training signal:

- **AIME** and other competition mathematics: short integer answers, currently
  unsaturated.
- **GPQA Diamond**: the hardest slice of GPQA and the standard science-reasoning
  number in current model announcements.
- **Competitive programming** (Codeforces-style), often reported as an Elo rating.
- **SWE-bench** for code, covered in section f.
- **ARC-AGI** (Chollet, 2019, arXiv:1911.01547): abstract visual puzzles designed
  to resist memorization, with a private held-out set.

### Reporting rules specific to this setting

1. **State how samples were combined**: pass@1, majority vote over `n` samples
   (self-consistency, Wang et al., 2022, arXiv:2203.11171), or best-of-`n` with a
   verifier. These can differ by tens of points on the same model and benchmark.
2. **State the reasoning budget.** Test-time compute is a dial; the same model at
   2,000 and at 32,000 thinking tokens is not the same system, and a score without
   a token budget is not reproducible.

### The pass@1 versus pass@k gap

Module 7's claim — that RL sharpens the sampling distribution rather than expanding
the set of solvable problems — is, stated as a measurement, a claim about this
gap. If pass@1 rises while large-`k` pass@k stays flat, probability mass moved onto
answers the model could already occasionally produce; the solvable set is
unchanged. Reporting only pass@1 therefore overstates what RL added.

The exercise produces a small instance of the same phenomenon running in the other
direction: on the `qa` task the GRPO model scores 70% pass@1 and 100% pass@5. The
correct answer is still in the distribution; it is no longer on top.

### What RL breaks that the target benchmark will not show

- **Regressions on other tasks** — the alignment tax noted in Module 7 and
  originally reported for RLHF in the InstructGPT paper (Ouyang et al., 2022,
  arXiv:2203.02155). The remedy is procedural: keep running the old evaluations.
- **Reward hacking that survives to test time** — padding, hedging, or formatting
  tricks that scored well during training do not disappear when training stops.

The exercise is a direct demonstration. GRPO on `reverse` alone gains 16.7 points
on `reverse` and loses 50 to 75 points on the three tasks that were not being
watched. The KL-to-reference penalty does not prevent this: the KL term is summed
over completions of the *reverse* prompts, so it constrains the policy where it
trains and does nothing on prompts it never sees. Raising `BETA` in
`make_checkpoints.py` does not shrink the tax, which is precisely why the tax needs
its own evaluation rather than a training-side fix.

### Behavior evaluations come in pairs

Refusal of harmful requests must be reported alongside false refusals on benign
ones, measured by over-refusal suites such as **XSTest** (Röttger et al., 2023,
arXiv:2308.01263). Either number alone is trivially maximized by a degenerate
model that refuses everything or nothing. The structure is the same as precision
and recall: one number can always be bought with the other.

### Goodhart's law

"When a measure becomes a target, it ceases to be a good measure." Module 7 showed
the training-side version (reward hacking); benchmark chasing is the same
mechanism at industry scale, and it motivates sections g and h.

- **Interactive widget (`:::interactive widget="passAtK"`):** plots $\text{pass@}k = 1 - (1 - p)^k$ for two models with different per-sample success rates $p$, under the independent-samples assumption. The default 30% and 10% models are 20 points apart at $k = 1$ and half a point apart at $k = 50$: both curves saturate, so a pass@k headline compresses exactly the difference a user experiences. This is the arithmetic behind the reporting rule on the surrounding slides &mdash; a pass@k number must always be quoted with its $k$, and pass@1 is the number that describes a single-shot product.

## e. Evaluating multimodal models

### Why scoring gets harder

Module 8 built models that condition on images. Two mundane facts make grading
harder. First, correctness often depends on visual detail — spatial relations,
small print, a chart axis — that no string comparison can inspect. Second,
free-form visual answers defeat exact match ("two" versus "there are two
people"), which is why multimodal suites lean more heavily on multiple choice and
LLM judges than text suites do.

### Understanding benchmarks

- **VQAv2** (Goyal et al., 2017, arXiv:1612.00837): short-answer questions about
  natural images, scored against ten human answers. Defined the visual
  question-answering task and was explicitly rebalanced to weaken language priors.
- **TextVQA, DocVQA** (Mathew et al., 2020, arXiv:2007.00398), **ChartQA** (Masry
  et al., 2022, arXiv:2203.10244), **AI2D**: reading text in images, documents,
  charts, and diagrams. This is where OCR ability shows up, and it is what most
  business use of vision models actually requires.
- **MMMU** (Yue et al., 2023, arXiv:2311.16502): college-level multi-discipline
  questions with figures; the current headline multimodal number.
- **MathVista** (Lu et al., 2023, arXiv:2310.02255): mathematical reasoning over
  visual inputs.
- **MMBench** (Liu et al., 2023, arXiv:2307.06281) and **MME**: broad capability
  suites with per-ability breakdowns (recognition, spatial relations, counting,
  OCR).
- **Video-MME** (Fu et al., 2024, arXiv:2405.21075): video understanding, adding
  temporal reasoning over much longer inputs.
- **POPE** (Li et al., 2023, arXiv:2305.10355): a hallucination probe that asks
  about objects absent from the image and measures whether the model agrees they
  are present.

### Captioning and generation

- **CIDEr** (Vedantam et al., 2015, arXiv:1411.5726) and **SPICE** score captions
  against several human references on MS COCO.
- **CLIPScore** (Hessel et al., 2021, arXiv:2104.08718) uses a contrastive
  image-text model — the CLIP objective from Module 8 — to measure image-caption
  agreement with no reference caption required.
- **FID** measures distributional similarity between generated and real images;
  human preference remains the standard for perceived quality.
- **Word error rate** is the metric for speech recognition, and one of the oldest
  clean automatic metrics in the field.

The pattern from section c repeats: where a reference exists, compare to it; where
one does not, compare two outputs and ask a judge.

### The text-only ablation (side quest)

Many multiple-choice visual benchmarks can be partly answered from language priors
alone. The check is to run the benchmark with the image removed; whatever score
survives above chance was produced by the question and options. On several popular
suites a text-only model scores well above chance, meaning the reported number is
partly a language benchmark. The dataset-side fix is balancing, which is what
VQAv2 did relative to VQA v1, and what Module 8's synthetic scene dataset does by
construction.

## f. Evaluating agents

### The question changes

An agent is a model plus tools plus many turns. Scoring shifts from inspecting a
string to inspecting the **final state of an environment**: run the project's test
suite, query the database, check whether the file exists and contains what it
should. At this point evaluation stops being a metric on outputs and becomes an
integration test.

### Benchmarks

- **SWE-bench** (Jimenez et al., 2023, arXiv:2310.06770): resolve a real GitHub
  issue in a real repository, scored by running the project's tests. **SWE-bench
  Verified** is the human-filtered subset that is now conventionally reported.
- **GAIA** (Mialon et al., 2023, arXiv:2311.12983): general-assistant questions
  requiring browsing, files, and tools, with short verifiable answers.
- **WebArena** (Zhou et al., 2023, arXiv:2307.13854) and **OSWorld**: complete
  tasks in a simulated browser or desktop, scored on end state.
- **tau-bench** (Yao et al., 2024, arXiv:2406.12045): multi-turn customer-service
  tasks with tool calls and policy rules.

### Why it is hard

Runs are long and expensive; environments are stateful and sometimes flaky;
partial progress is real but usually scored pass/fail; and the agent can succeed
for the wrong reason — editing the tests instead of the code. That last failure is
reward hacking in the place where it is hardest to notice, because the scoring
function is a program the agent can reach.

Modules 10 and 11 build the systems these benchmarks measure, which is why this
section is deliberately short.

## g. Why the same model gets different scores

### Protocol dependence

A benchmark number belongs to a model **and a protocol**. The same checkpoint can
move several points on MMLU because of choices that have nothing to do with the
weights:

- prompt template, chat formatting, and option labeling (A–D versus 1–4);
- zero-shot versus few-shot, and how many examples;
- free generation versus option-likelihood scoring, and whether likelihoods are
  length-normalized;
- the answer-extraction regex;
- normalization rules for lowercasing, punctuation, articles, and units;
- decoding settings: greedy versus sampled, temperature, seed, sample count.

Reproducible reports therefore log the model revision, dataset version, prompt,
decoding settings, and the exact harness version. Documented cases of large
protocol-driven discrepancies in public MMLU numbers are the standard motivating
example, and they are why the Open LLM Leaderboard pins a harness version.

The side quest "One model, two scores" is the hands-on version, and it is the
first extra credit in the exercise: change the chat template in `_prefix_ids()`,
keep the weights identical, and report how far the numbers move.

### Contamination

If test items or near-duplicates appear in the training data, a high score
measures recall rather than generalization. Web-scale training sets make this
impossible to exclude completely: a benchmark question discussed on a forum,
translated, or reworded in a textbook is still contamination and will not match an
n-gram check. Partial defenses:

- **N-gram overlap checks** against the training corpus (catches exact and
  near-exact copies);
- **time-based splits** — test on material published after the training cutoff,
  which works once per benchmark;
- **private held-out test sets** — the approach taken by GLUE, SuperGLUE, and
  ARC-AGI.

The side quest "Was the test in the training set?" is built into the exercise
data on purpose. The `uppercase`, `repeat`, and `reverse` evaluation cases use
words that appear nowhere in finetuning, but all eight `qa` facts were memorized
verbatim during SFT. That task is contaminated by construction, and it is the task
on which the instruct model scores 100%. The intended follow-up question: this
check took one `grep` over 1,500 examples — what could it mean for a model trained
on a substantial fraction of the public internet?

### Saturation, and the to-scale chart

Slide `protocol-saturation-chart` is a to-scale span chart: each bar runs from a
benchmark's publication year to the year the best systems reached the human or
expert ceiling, on a linear 1998–2026 axis.

| Benchmark | Published | Ceiling reached | Span |
|---|---|---|---|
| MNIST | 1998 | ~2012 | ~14 years |
| ImageNet (ILSVRC) | 2009 | 2015 | ~6 years |
| SQuAD | 2016 | 2018 | ~2 years |
| GLUE | 2018 | 2019 | ~1 year |
| SuperGLUE | 2019 | 2021 | ~2 years |
| MMLU | 2020 | ~2024 | ~4 years |
| GSM8K | 2021 | 2023 | ~2 years |
| GPQA Diamond | 2023 | ~2025 | ~2 years |

The dates for "ceiling reached" are approximate by nature and are meant to be read
as the point at which the benchmark stopped separating frontier systems, not as a
single dated event. The claim the chart supports is the coarse one: the window has
gone from over a decade to roughly two years. It is deliberately **not** claimed
that the shrinkage is monotonic — MMLU took about four years, longer than SQuAD or
GLUE.

Once the best models sit near the ceiling, remaining gains are mostly noise and
overfitting, which is why the field keeps building successors: GLUE to SuperGLUE,
MMLU to MMLU-Pro and GPQA, GSM8K to AIME, ARC to ARC-AGI.

The practical consequence: public benchmarks compare models in general and are not
a substitute for a small private test set drawn from your own task — which is also
the only set nobody else can train on.

## h. Running evaluations in practice, and the handoff to Module 10

### Tooling

- **lm-evaluation-harness** (EleutherAI) — the de facto standard for
  likelihood-scored academic benchmarks and the engine behind the Open LLM
  Leaderboard.
- **HELM** (Liang et al., 2022, arXiv:2211.09110) — evaluate many scenarios
  against many metrics at once (accuracy, calibration, robustness, fairness, bias,
  toxicity, efficiency) and report explicitly what was not covered. HELM is the
  work that shifted expectations from a single headline number to a grid with
  visible gaps.
- **Inspect**, **lighteval**, and **OpenAI Evals** — general frameworks for writing
  your own evaluations, including agent and tool-use tasks.
- **VLMEvalKit** and **lmms-eval** — the multimodal equivalents.

### A three-tier setup

1. A small **fast eval** run on every change during development.
2. A **regression suite** of cases that used to fail and must keep passing; it
   grows each time something is fixed.
3. A slower, broader **release eval** before swapping models in production,
   including public benchmarks and a private set.

Always keep per-case outputs rather than only averages: reading the failures is
where an evaluation turns into a fix, and an aggregate can hide a total regression
on one task inside a small overall gain. Report the per-task breakdown alongside
the average for the same reason. The exercise is built around exactly this point.

### Handoff to Module 10

Quality is one axis of a deployment decision; the other is cost — latency, memory,
and throughput. A model that scores two points higher and runs four times slower
may still be the wrong choice, and that trade is where serving begins.

## Exercise notes

### What the exercise is

`exercises/module_09_evaluation/` bundles two finished checkpoints and asks the
student to write the eight metrics that decide which is better. Nothing is
trained.

- `data/instruct_model.pt` — the Module 6 story. The Module 5 base model, with its
  vocabulary widened by four chat special tokens, supervised-finetuned on four toy
  tasks: `uppercase`, `repeat`, `qa`, and `reverse`.
- `data/rl_model.pt` — the Module 7 story. That same checkpoint after 150 GRPO
  steps on the **reverse task only**, with an exact-match verifiable reward.

Both are regenerated by `solution/src/make_checkpoints.py`; the evaluation files by
`src/data.py`. The GRPO mean reward climbs from about 0.49 to about 0.81 during
those 150 steps.

### The eight steps

| Step | Function | What it computes |
|---|---|---|
| 1 | `perplexity()` | $\exp(\mathcal{L})$ from the mean held-out token loss |
| 2 | `normalize_answer()` | Lowercase, strip punctuation, collapse whitespace |
| 3 | `exact_match()` | 1.0 if the normalized answer matches any acceptable answer |
| 4 | `token_f1()` | Harmonic mean of token precision and recall |
| 5 | `task_accuracy()` | Mean score within each task |
| 6 | `suite_score()` | Mean of the per-task scores |
| 7 | `score_multiple_choice()` | Highest average log-probability per token |
| 8 | `pass_at_k()` | $1 - \binom{n-c}{k}/\binom{n}{k}$ |

Steps 1–6 are enough to produce the full per-task report; steps 7 and 8 add the
two benchmarks that tell a different story about the same models.

### Why step 7 divides by length

Each option is scored by summing $\log p_\theta$ over its own tokens. Every extra
token contributes another negative number, so **total** log-probability
systematically favors the shortest option. Dividing by the token count is the
standard length normalization used by likelihood-scored multiple-choice harnesses.
Scoring the same set with total log-probability instead is one of the extra
credits, and the ranking changes in a predictable direction.

### The results, and what each one teaches

Actual output from `solution/output/run.txt`:

| Metric | instruct | rl |
|---|---|---|
| Perplexity (held-out text) | 1986.48 | **1599.26** |
| Multiple choice (16 questions) | 100.0% | 100.0% |
| Suite score (mean of four tasks) | **88.8%** | 55.4% |

Per-task exact match (greedy decoding):

| Task | cases | instruct | rl | diff |
|---|---|---|---|---|
| uppercase | 10 | 80.0% | 30.0% | −50.0 |
| repeat | 8 | 100.0% | 25.0% | −75.0 |
| reverse | 24 | 75.0% | 91.7% | **+16.7** |
| qa | 8 | 100.0% | 75.0% | −25.0 |

Three lessons fall out of one run:

1. **Perplexity disagrees with usefulness.** The RL model has the better
   perplexity and is far worse at three of four tasks.
2. **The scoring shape decides what is visible.** The likelihood-scored
   multiple-choice benchmark scores both models 16/16, because it never asks
   either model to produce anything. Only the generation-based suite detects the
   collapse.
3. **The average hides the structure.** The single headline number is −33.3, which
   is neither the +16.7 the RL work actually bought nor the −75 it cost.

A fourth appears in the pass@k tables: on `qa` the RL model scores 70% pass@1 and
100% pass@5. The right answer is still in the distribution.

### Deliberate design choices worth flagging in class

- The `qa` task is **contaminated by construction** (section g). The eight facts
  tested are the eight facts memorized. This is why its 100% should be read as
  recall.
- The `uppercase` and `repeat` finetuning sets are padded with hundreds of random
  strings of varying length. Without that padding a model this small memorizes the
  thirty English words and fails every held-out word, which would make those tasks
  measure memorization rather than the transformation.
- `repeat` has only 8 cases, so its ±12.5-point granularity is coarse. That is the
  motivation for the bootstrap-confidence-interval extra credit: is the `repeat`
  gap larger than the noise on eight cases?
- The runner prints the **protocol** — template, normalization, generation budget,
  decoding, seed, suite size — before any number, which is the reporting habit
  section g argues for.

## Quiz answer notes

The ten questions in the deck test connections rather than recall of the exercise:

1. **Base-model measurement** — perplexity and likelihood-scored multiple choice;
   perplexity is insufficient because it rewards fluency, not correctness.
2. **Tokenizer comparability** — perplexity is per token; report bits per byte.
3. **Why likelihood scoring** — it runs on models that cannot follow instructions
   and removes answer extraction from the measurement; the cost is that it cannot
   see a generation collapse.
4. **Protocol variance** — shot count, prompt template, scoring shape, extraction,
   normalization, length normalization.
5. **Reward curve** — it measures the optimized quantity on the training prompts.
6. **pass@1 up, pass@10 flat** — sharpening, not expansion.
7. **Contamination** — recall rather than generalization, unfalsifiable at web
   scale.
8. **Arena without improving** — longer, better formatted, more agreeable.
9. **Refusal pairs** — each number alone is maximized by a degenerate policy.
10. **Multimodal ablation** — remove the image and see what survives.

## References

Foundational benchmarks and metrics

- Deng et al., "ImageNet: A Large-Scale Hierarchical Image Database" (2009);
  ILSVRC, <https://www.image-net.org/>
- Rajpurkar et al., "SQuAD: 100,000+ Questions for Machine Comprehension of Text,"
  arXiv:1606.05250
- Wang et al., "GLUE," arXiv:1804.07461; Wang et al., "SuperGLUE," arXiv:1905.00537
- Papineni et al., "BLEU: a Method for Automatic Evaluation of Machine
  Translation," ACL 2002
- Lin, "ROUGE: A Package for Automatic Evaluation of Summaries," ACL 2004
- Chen et al., "Evaluating Large Language Models Trained on Code" (HumanEval and
  pass@k), arXiv:2107.03374

Language model benchmarks

- Hendrycks et al., "Measuring Massive Multitask Language Understanding" (MMLU),
  arXiv:2009.03300
- Hendrycks et al., "Measuring Mathematical Problem Solving with the MATH Dataset,"
  arXiv:2103.03874
- Zellers et al., "HellaSwag," arXiv:1905.07830
- Clark et al., "Think you have Solved Question Answering?" (ARC), arXiv:1803.05457
- Rein et al., "GPQA: A Graduate-Level Google-Proof Q&A Benchmark," arXiv:2311.12022
- Cobbe et al., "Training Verifiers to Solve Math Word Problems" (GSM8K),
  arXiv:2110.14168
- Zhou et al., "Instruction-Following Evaluation for Large Language Models"
  (IFEval), arXiv:2311.07911
- Suzgun et al., "Challenging BIG-Bench Tasks" (BBH), arXiv:2210.09261
- Wang et al., "MMLU-Pro," arXiv:2406.01574
- Wang et al., "Self-Consistency Improves Chain of Thought Reasoning,"
  arXiv:2203.11171
- Chollet, "On the Measure of Intelligence" (ARC-AGI), arXiv:1911.01547

Human and model judgment

- Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena,"
  arXiv:2306.05685
- Chiang et al., "Chatbot Arena: An Open Platform for Evaluating LLMs by Human
  Preference," arXiv:2403.04132
- Röttger et al., "XSTest: A Test Suite for Identifying Exaggerated Safety
  Behaviours," arXiv:2308.01263

Multimodal benchmarks

- Goyal et al., "Making the V in VQA Matter" (VQAv2), arXiv:1612.00837
- Yue et al., "MMMU," arXiv:2311.16502
- Lu et al., "MathVista," arXiv:2310.02255
- Mathew et al., "DocVQA," arXiv:2007.00398
- Masry et al., "ChartQA," arXiv:2203.10244
- Liu et al., "MMBench," arXiv:2307.06281
- Fu et al., "Video-MME," arXiv:2405.21075
- Li et al., "Evaluating Object Hallucination in Large Vision-Language Models"
  (POPE), arXiv:2305.10355
- Hessel et al., "CLIPScore," arXiv:2104.08718
- Vedantam et al., "CIDEr," arXiv:1411.5726

Agent benchmarks

- Jimenez et al., "SWE-bench," arXiv:2310.06770
- Mialon et al., "GAIA," arXiv:2311.12983
- Zhou et al., "WebArena," arXiv:2307.13854
- Yao et al., "tau-bench," arXiv:2406.12045

Practice and tooling

- Liang et al., "Holistic Evaluation of Language Models" (HELM), arXiv:2211.09110
- EleutherAI, Language Model Evaluation Harness,
  <https://github.com/EleutherAI/lm-evaluation-harness>
- VLMEvalKit, <https://github.com/open-compass/VLMEvalKit>
- Inspect, <https://inspect.aisi.org.uk/>

Carried over from earlier modules

- Brown et al., "Language Models are Few-Shot Learners" (GPT-3, in-context
  learning), arXiv:2005.14165
- Kaplan et al., "Scaling Laws for Neural Language Models," arXiv:2001.08361
- Hoffmann et al., "Training Compute-Optimal Large Language Models" (Chinchilla),
  arXiv:2203.15556
- Ouyang et al., "Training Language Models to Follow Instructions with Human
  Feedback" (InstructGPT, alignment tax), arXiv:2203.02155
- Shannon, "A Mathematical Theory of Communication," *Bell System Technical
  Journal*, 1948 (entropy; the prediction/compression identity)
- Shannon, "Prediction and Entropy of Printed English," *Bell System Technical
  Journal*, 1951 (the ~1 bit-per-character estimate cited on
  `base-bpb-compression`)
- Gao et al., "The Pile," arXiv:2101.00027 (bits per byte as the reported
  cross-tokenizer metric)

## Image credits

Photographs on the notable-figure slides, each taken from the subject's own
institutional or personal page except where noted:

- `feifei_li.jpg` and `sparck_jones.jpg` — Wikimedia Commons
- `hendrycks.jpg` — Wikimedia Commons
- `bowman_rajpurkar.jpg` — Samuel Bowman's personal page
  (<https://sleepinyourhat.github.io/>) and the Harvard DBMI faculty directory
- `papineni_lin.jpg` — Custom.MT's MT-leaders profile and Microsoft Research's
  people directory
- `chiang_zheng.jpg` — Wei-Lin Chiang's and Lianmin Zheng's personal pages
- `percy_liang.jpg` — Stanford CS faculty page

Two contributors named on these slides could not be given a photograph and are
credited in the body text instead: **Alex Wang** (GLUE) and **Ying Sheng**
(Chatbot Arena).
