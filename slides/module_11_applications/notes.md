# Module 11: Practical Applications and Case Studies. Lecture Notes

These notes give an explanation and a citation for every major claim on the
slides, map the two equations to the visuals they appear on, and record the
historical context. Module 11 trains nothing and changes no weights. It asks
what can be built around a finished, served model, and its organizing idea is
that every technique in the module (prompting, retrieval, tools, agents) is a
way of choosing which tokens sit in front of a frozen model. The context window
is the programmable surface.

## Review

- Modules 5 through 7 changed the weights: pretraining supplied capability,
  supervised finetuning supplied the chat format and instruction following, and
  RL supplied reasoning trained against verifiable rewards. Module 10 ended with
  those weights served behind an API. Nothing in this module touches a weight;
  everything happens at inference time, through the prompt.
- Three earlier results carry the module:
  - GPT-3's few-shot in-context learning (Brown et al., 2020, arXiv:2005.14165),
    reframed from an emergent curiosity into the primary engineering tool.
  - The chat template from Module 6, which makes roles (system, user, assistant,
    and later tool) representable as special tokens the model was finetuned to
    respect.
  - Module 9f's agent benchmarks (SWE-bench, tau-bench), introduced there with a
    promise that they would make sense once this module built the systems they
    measure.
- Module 8's contrastive objective (CLIP) returns as the training recipe for the
  sentence encoders used in dense retrieval, and Module 10's KV cache returns as
  prompt caching, a product feature with a price attached.

## a. In-context learning, revisited

### The reframing

- In Module 5, in-context learning was presented as GPT-3's headline finding:
  conditioning on examples in the prompt substitutes for gradient updates
  (Brown et al., 2020, arXiv:2005.14165). Here the same phenomenon is treated as
  an engineering interface. The prompt is a program: it specifies the task, the
  format, the examples, and the constraints, and the frozen model executes it.
- Zero-shot means a task description alone; few-shot adds worked examples.
  Few-shot examples pin down output format and edge-case policy far better than
  prose instructions, which is the same protocol effect Module 9 documented:
  shot count moves benchmark scores on identical weights. The design heuristic
  on the slide ("add an example rather than an adjective") is the applied form.

### System prompts

- The system prompt is the application developer's channel, separate from the
  user's. The separation is representable only because the chat template assigns
  each role its own special tokens, and it is effective only because the model
  was finetuned on conversations in which system instructions take precedence.
- The slide flags that this separation is a learned behavior, not an enforced
  mechanism. That fact is deliberately planted early because section g's prompt
  injection discussion depends on it.

### Chain-of-thought prompting

- Wei et al., 2022 (arXiv:2201.11903) showed that prompting a model to produce
  intermediate reasoning steps before its answer substantially improves accuracy
  on arithmetic, commonsense, and symbolic reasoning tasks, with the effect
  growing with model scale.
- Kojima et al., 2022 (arXiv:2205.11916) reduced the prompt to a single zero-shot
  trigger ("Let's think step by step").
- Mechanically, each generated step becomes context for the next prediction, so
  the answer is computed in many small conditional steps rather than one large
  one.
- The Module 7 connection: reasoning models are chain-of-thought pushed into the
  weights by RL against verifiable rewards. Prompted chain-of-thought is the
  inference-time version available on any instruct model. The quiz returns to
  this distinction.
- Jason Wei was introduced in Module 6 for FLAN; the slide gives him a one-line
  mention rather than a second figure treatment.

### Structured output and constrained decoding

- Applications need machine-readable output. Two mechanisms:
  - Asking: describe the schema in the prompt. Works because format-following
    was finetuned in (Module 6); fails occasionally.
  - Constrained decoding: compile the schema or grammar into a token-level mask
    applied to the logits at each step, so tokens that would violate the grammar
    receive zero probability and the output cannot fail to parse. This is a
    direct extension of Module 5's sampling machinery (same logits, same
    softmax, one extra mask).
- Constrained decoding is what makes function-calling APIs reliable (section d).

### Sampling parameters as product knobs

- Temperature near 0 for extraction, classification, and tool calls, where the
  top of the distribution is wanted every time; higher temperatures for
  drafting and brainstorming, where variety is wanted. Same dial as Module 5,
  now a per-request product decision.

### The limit

- Prompting shapes behavior; it cannot add knowledge the model never saw.
  This sentence is the hinge of the lecture: it motivates the knowledge problem
  (section b) and everything after it.

### Side quest: why does in-context learning work?

- Olsson et al., 2022, "In-context Learning and Induction Heads"
  (arXiv:2209.11895, Anthropic) identified induction heads: pairs of attention
  heads that find an earlier occurrence of the current token pattern in the
  context and promote whatever followed it. Their emergence during training
  coincides with a measurable jump in in-context learning ability, and the
  paper argues (with multiple lines of circumstantial evidence) that they
  constitute much of its mechanism.
- The slide's framing: this is Module 3's attention machinery doing something
  nobody designed, and it gives the prompt-as-program metaphor real circuitry.

## b. The knowledge problem

### Three gaps

- Training cutoff: the weights contain nothing after their data's end date.
- Private data: the model has never seen the user's documents, records, or code.
- Finite context: the window is bounded and every token in it is billed on every
  call (Module 10's prefill cost), so "paste everything" does not scale.

### Hallucination as trained behavior

- When asked past its knowledge, a language model completes plausibly, because
  next-token prediction (Module 5) rewards the most probable continuation and a
  confident wrong answer is more probable in the training distribution than an
  admission of ignorance. The slide states this as "doing exactly what it was
  trained to do, with nothing true to condition on"; the remedy is to put
  something true in the context, which is retrieval's premise.

### The three options

- Finetune (Module 6): effective for form, style, and tool syntax; poor for
  facts. Knowledge lands diffusely across weights, cannot be edited or audited
  per fact, and goes stale the day the documents change.
- Long context: works for small stable document sets; costs tokens on every
  call, and Module 10 located the cost (KV cache growth, prefill compute).
- Retrieve: store documents outside the model, search at query time, place only
  the relevant chunks in the context.
- The rule of thumb the field converged on: finetune for behavior, retrieve for
  knowledge. The phrasing circulates widely in practitioner writing; the slide
  presents it as a heuristic, not a theorem.

## c. Retrieval-augmented generation

### The name and the simplification

- Lewis et al., 2020, "Retrieval-Augmented Generation for Knowledge-Intensive
  NLP Tasks" (arXiv:2005.11401, Facebook AI) named the pattern and trained
  retriever and generator jointly, marginalizing over retrieved passages.
- The version the industry runs is simpler: a frozen retriever feeding a frozen
  generator through the prompt. The slides emphasize the shift because it is
  this module's thesis in miniature: grounding became a text-assembly problem
  once the context window became the interface.

### The pipeline

- Ingest and chunk documents; embed and index the chunks; at query time embed
  the question, retrieve the nearest chunks, assemble them with the question
  into a prompt; generate a grounded answer, ideally with citations.
- Steps one through three are classical information retrieval; step four is
  Module 5's sampling loop. The exercise deliberately skips chunking (its
  articles are short enough to index whole) to keep the focus on retrieval.

### Sparse retrieval and TF-IDF

- The equation on the slide:

  $$w_{t,d} = \mathrm{tf}(t,d) \cdot \log\frac{N}{\mathrm{df}(t)}$$

  where tf(t,d) is the count of term t in document d, N is the number of
  documents, and df(t) is the number of documents containing t.
- Term weighting by specificity is Karen Spärck Jones, 1972, "A Statistical
  Interpretation of Term Specificity and its Application in Retrieval"
  (Journal of Documentation; doi:10.1108/eb026526).
- The information-theoretic reading the slide gives: df(t)/N estimates the
  probability that a random document contains t, so log(N/df) is the surprisal
  (Module 1) of the event "a document contains this term". Rare terms carry
  more information about which document is wanted; a term in every document
  carries none and scores log(1) = 0.
- BM25 (Robertson and Zaragoza, "The Probabilistic Relevance Framework: BM25
  and Beyond," doi:10.1561/1500000019) is the tuned descendant: saturating term
  frequency and length normalization added to the same idea. It remains the
  standard sparse baseline, and the exercise's extra credit implements its TF
  saturation.

### Dense retrieval

- The equation on the slide:

  $$\mathrm{sim}(q,d) = \frac{\mathbf{q}\cdot\mathbf{d}}{\lVert\mathbf{q}\rVert\lVert\mathbf{d}\rVert}$$

  cosine similarity between a query embedding and a document embedding produced
  by a bi-encoder (each text encoded independently).
- Lineage: Dense Passage Retrieval (Karpukhin et al., 2020, arXiv:2004.04906)
  established dense retrieval for open-domain QA; Sentence-BERT (Reimers and
  Gurevych, 2019, arXiv:1908.10084) established the practical recipe for
  sentence embeddings from BERT-style encoders.
- The encoders are trained contrastively: pull matching pairs together, push
  mismatched pairs apart. This is the CLIP objective from Module 8 with text on
  both sides. The exercise's encoder, all-MiniLM-L6-v2, is a 6-layer, 384-
  dimension, 23M-parameter model from the Sentence-Transformers project,
  finetuned on over a billion sentence pairs.

### The complementary failure modes

- Sparse wins exact identifiers (error codes, part numbers, names): the literal
  string is its own dimension with a large IDF weight. Dense encoders split
  rare identifiers into subwords and average them into the vector, so nearby
  codes (E-341, E-350) embed almost identically and surrounding natural-language
  words dominate the geometry.
- Dense wins paraphrase: synonyms land near each other because they occur in
  similar contexts, so zero lexical overlap still scores high. Sparse has no
  notion of synonymy at all; its score for a no-overlap pair is exactly zero.
- The exercise is engineered so both failure modes appear in the same run, with
  a labeled query set split into keyword, paraphrase, and verbatim categories.

### Production practice

- Hybrid search: run both retrievers and merge rankings, typically with
  reciprocal rank fusion (Cormack et al., 2009: score each document by the sum
  of 1/(60 + rank) across rankings).
- Reranking: a cross-encoder reads query and candidate together through one
  transformer and scores the pair. More accurate than comparing independently
  produced vectors, too slow for the full corpus, so it runs on a shortlist
  (retrieve 100 cheaply, rerank carefully).
- Scale: exact nearest-neighbor search is linear in corpus size. Approximate
  indexes trade a little recall for large speedups; HNSW (Malkov and Yashunin,
  arXiv:1603.09320) is the standard graph-based method. A vector database is
  such an index plus storage plumbing. FAISS
  (github.com/facebookresearch/faiss) is the standard library.

### Failure modes and evaluation

- A retrieval miss produces a confident wrong answer downstream that is
  indistinguishable from hallucination, though the generator behaved correctly.
- Lost in the middle: Liu et al., 2023 (arXiv:2307.03172) measured accuracy as
  a function of where the answer-bearing document sits in the context and found
  a U-shaped curve: models use information at the start and end of the context
  far better than information in the middle. Practical consequence: order
  retrieved chunks so the best-ranked land at the prompt's edges.
- Faithfulness: a fluent answer can contradict its retrieved sources; grounding
  reduces but does not eliminate this.
- Evaluation splits along the pipeline seam, using Module 9's vocabulary:
  retrieval is scored alone with recall@k and MRR against labeled
  query-document pairs (cheap, exact); generation is scored for faithfulness to
  the retrieved text, typically with an LLM judge spot-checked by humans.

### Side quest: is RAG dead?

- Context windows grew from 4K tokens to millions, and each jump restarts the
  argument that retrieval is obsolete. The counterarguments: cost per token on
  every call, prefill latency (Module 10), lost-in-the-middle degradation, and
  corpora measured in billions of tokens. The slide runs it as a
  critical-thinking exercise in the style of Module 5's emergence debate: what
  evidence would settle the question, and for which corpus size? The honest
  answer is that the crossover point is application-dependent.

- **Interactive widget (`:::interactive widget="retrievalCompare"`):** six documents and three queries, with hand-set sparse and dense scores chosen to expose each retriever's characteristic failure; the hybrid mode computes reciprocal rank fusion, $\text{RRF}(d) = \sum_{\text{retrievers}} \frac{1}{K + \text{rank}(d)}$ with $K = 60$ (Cormack, Clarke, and Buettcher, 2009), from the two rankings at run time rather than from stored numbers. On the rare-identifier query the embedding ranks the exact match 4th, behind documents that merely sound related; on the paraphrase query word overlap ranks the answer 4th, because query and answer share almost no terms. Fusion recovers rank 1 on two of the three queries and rank 2 on the paraphrase, which is the honest result: RRF cannot fully rescue a document one retriever buried while a competitor placed 1st and 2nd. That residual gap is what the reranker on the following slide is for.

## d. Tool use

### The mechanism

- The model emits a structured call in its output; the runtime detects it,
  pauses generation, executes the call in ordinary software, appends the result
  to the context as tokens, and resumes generation. The model never executes
  anything itself; tool use is a protocol layered on sampling and enforced by
  the runtime.

### Where the ability comes from

- Tool calling is trained in, not innate: an SFT pass (Module 6 machinery) over
  examples of when to call, how to format the call, and how to read the result.
  The chat template grows a tool role alongside system, user, and assistant.
- Toolformer (Schick et al., 2023, arXiv:2302.04761) is the research version:
  the model annotates its own training text with candidate API calls, calls
  that reduce the loss on subsequent tokens are kept, and the model is
  finetuned on the result, teaching itself where a calculator, search engine,
  or translator would have helped.

### The standard set and the design insight

- Calculator (models are unreliable at arithmetic because digits are tokens,
  Module 4's tokenization discussion), web search (facts past the cutoff), code
  execution (the universal escape hatch), and retrieval (section c packaged as
  a tool).
- The organizing insight: tools let the model outsource exactly what next-token
  prediction is bad at: precise computation, current knowledge, and side
  effects on the world.

### Function-calling APIs

- The productized form: the developer sends JSON schemas describing available
  tools; the model returns a call matching a schema; constrained decoding
  (section a) compiles the schema into a grammar that masks invalid tokens, so
  the call is guaranteed to parse.

## e. Agents

### The loop

- One tool call answers a question; an agent is the loop: generate (reason,
  choose an action), act (call a tool), observe (the result lands in context),
  repeat until a stop condition (goal met, budget exhausted, or human input
  needed).

### ReAct

- Yao et al., 2022, "ReAct: Synergizing Reasoning and Acting in Language
  Models" (arXiv:2210.03629) crystallized the pattern: interleave reasoning
  traces with actions and observations in one generation stream, so each
  observation informs the next thought. The slide's framing: chain-of-thought
  from section a, with the world talking back between thoughts.
- The example transcript on the ReAct slide is illustrative (composed for the
  slide), not output from a specific system.

### Context management

- The context window is the agent's working memory, and long tasks overflow it.
  Standard techniques: summarize the transcript so far and continue from the
  summary; drop stale tool output; keep the goal and constraints pinned at a
  fixed location in the prompt. The slide insists this is core engineering
  because both correctness (forgetting the goal) and cost (section g) live in
  the context.

### Error compounding

- The arithmetic on the slide: a 95% per-step success rate over twenty
  independent steps gives 0.95^20, approximately 0.358, so roughly a 36% task
  success rate. The independence assumption is a simplification (real agents
  can recover), but the qualitative point stands and explains why agent
  reliability lags single-shot quality.
- The field's answer is verification: run the tests, check the end state,
  confirm before irreversible actions. A verifier converts errors from fatal to
  retryable, which changes the structure of the failure model rather than its
  constant. The quiz makes this quantitative.

### The Model Context Protocol

- MCP (modelcontextprotocol.io, introduced by Anthropic in November 2024) is an
  open standard for connecting agent clients to tool servers: discovery, call
  format, and result format on the wire. The slide's point is the signal, not
  the acronym: protocols appear when a pattern stops being research, as they
  did for the web.

## f. Case study: the agentic coding assistant

- Claude Code and Codex are presented as the synthesis exhibit. The component
  table maps each part to its lecture section: system prompt (a), tool set of
  read, edit, search, run (d), retrieval over the repository (c), the agent
  loop with context management (e), and the project's test suite as verifier.
- Tests-as-verifier is Module 7's verifiable-reward idea reused at inference
  time: during training the checkable signal selected which reasoning to
  reinforce; here it tells the loop whether to stop or retry.
- Why coding became the flagship agent domain: the work is text-native (code is
  tokens), tool-rich (compilers, linters, and test runners predate the agent),
  and checkable (the verifier already exists). Few domains offer all three.
- The stack table reads the whole course into one product: pretraining
  (Module 5) supplied capability, SFT (Module 6) the instruction format and
  tool syntax, RL (Module 7) judgment on hard problems, evaluation (Module 9)
  the yardstick, serving (Module 10) the tokens per second, and this module
  everything wrapped around the API call.

## g. Engineering realities

### Cost

- The cost model of an LLM application is tokens: price per token, times tokens
  per request, times requests. Context stuffing pays for every retrieved chunk
  on every call; agent transcripts grow with each step, so multi-turn loops
  scale roughly quadratically in total tokens processed.
- Latency follows Module 10's split: prefill is compute-bound and scales with
  prompt length (long prompts delay the first token); decode is memory-bound
  and scales with output length.

### Prompt caching

- Prompt caching is the KV cache from Module 10 productized across requests: a
  stable prefix (system prompt, tool schemas, few-shot examples) is computed
  once and reused at reduced price and latency. Design rule: static parts
  first, variable parts last, because one differing token invalidates the cache
  for everything after it. Major providers (Anthropic, OpenAI, Google) all ship
  a version of this.

### Prompt injection

- The vulnerability: instructions and data share one token stream, and the
  instruction hierarchy is a finetuned behavior (section a), not an enforced
  mechanism. Retrieved documents, web pages, tool results, and emails are
  untrusted text piped next to trusted instructions, and text crafted to read
  as instructions gets obeyed some fraction of the time.
- Simon Willison coined the term in September 2022 and maintains the running
  catalog of real attacks (simonwillison.net/series/prompt-injection/),
  including data exfiltration through markdown image URLs and email agents
  forwarding private mail.
- No complete defense exists. Mitigations are classical security thinking:
  least privilege on tools, separating trusted instructions from untrusted
  content, human confirmation before irreversible actions. The side-quest slide
  shows a minimal injected document against a toy RAG pipeline like the
  exercise's; the demo document is composed for the slide.

### Evaluation in production

- Module 9's practice section applied to a weekly-changing system: a fast eval
  on every prompt change (prompts are code; this is their unit test), a
  regression suite built from past production failures, a broader release eval
  before shipping, and per-case outputs kept readable because aggregates hide
  what matters.

### Handoff

- Everything in this module treats the transformer as settled infrastructure.
  Module 12 asks how long that holds: successor architectures, data limits, and
  which of this module's patterns survive the next substrate.

## Notable figures

- **Karen Spärck Jones** (1935-2007). Cambridge computer scientist; worked in
  natural language processing and information retrieval from the late 1950s.
  The 1972 term-specificity paper introduced IDF. Module 9 borrowed her field's
  vocabulary (precision, recall, F1); this module gives her the full figure
  treatment, placed immediately before the TF-IDF equation. Her slogan
  "Computing is too important to be left to men" is well documented from her
  later advocacy (widely quoted, including in her 2007 Guardian and Times
  obituaries).
- **Patrick Lewis and Douwe Kiela**. Named authors of the RAG paper (Facebook
  AI, 2020). Lewis has worked at the retrieval/LLM seam throughout (FAIR,
  Cohere); Kiela co-founded Contextual AI in 2023 to commercialize
  retrieval-augmented systems. Placed in section c at the paper's introduction.
- **Shunyu Yao**. Princeton PhD, then OpenAI. First author of ReAct (2022),
  which fixed the reason-act-observe loop as the standard agent pattern, and of
  tau-bench (2024), the end-state agent benchmark cited in Module 9f. Placed in
  section e.
- **Jason Wei**. Chain-of-thought prompting (2022). Already introduced as a
  figure in Module 6 for FLAN, so section a gives a one-line mention instead of
  a repeat figure slide.

## Exercise notes

### What the exercise is

- Build two retrievers over one corpus and read a per-category report showing
  that neither dominates. The corpus is 48 short support articles for a
  fictional PX-series printer line, written so both retrievers have something
  to win: ten error-code articles and six part-number articles carry exact
  identifiers; the how-to articles use manual vocabulary ("creased or wrinkled
  output") that user queries will not reuse ("my pages come out crumpled").
- 30 labeled queries in three designed categories: keyword (contains an exact
  identifier; sparse should win), paraphrase (near-zero lexical overlap with
  the answer article; dense should win), and verbatim (reuses the article's own
  wording; both should get it, establishing the ceiling).
- The dense encoder is bundled (all-MiniLM-L6-v2, fp16, ~44MB, CPU-only, no
  network access at runtime). Students implement pooling and similarity, not
  the transformer. No chunking: articles embed whole.

### The eight functions, seven steps

1. `tokenize`: lowercase, map non-alphanumerics to spaces, split. Note that
   E-341 becomes the terms "e" and "341"; the rare term "341" is the sparse
   signal.
2. `inverse_document_frequency`: log(N/df) per term, the Spärck Jones weight
   and Module 1's surprisal.
3. `tfidf_vector`: counts times IDF, one vector per document over the corpus
   vocabulary (811 terms in the bundled corpus; about 38 nonzero per article,
   which is the concrete meaning of "sparse").
4. `cosine_similarity`: dot product over the product of norms; shared verbatim
   by both retrievers.
5. `rank_documents`: score every document, return the top k indices. Steps 1-5
   are a complete retriever.
6. `mean_pool`: average the encoder's per-token vectors under the attention
   mask (batching pads texts, so the mask matters). This one function plus the
   provided encoder is the entire dense retriever; ranking code is reused.
7. `recall_at_k` and `reciprocal_rank`: the retrieval metrics from section c's
   evaluation slide.

### The results (actual solution run)

- Worked examples printed before any aggregate: the keyword query "pages take
  ages to come out and now it says E-341" (sparse: relevant article first at
  0.360; dense: art-slow, art-blank, err-e520, relevant missing) and the
  paraphrase query "my pages come out crumpled" (sparse: three irrelevant
  results; dense: relevant article first at 0.637).
- The report, per category (recall@1 / recall@3 / MRR):
  - keyword: sparse 100% / 100% / 1.00, dense 30% / 60% / 0.53
  - paraphrase: sparse 10% / 30% / 0.25, dense 50% / 100% / 0.72
  - verbatim: both 100% / 100% / 1.00
  - overall: sparse 70% / 77% / 0.75, dense 60% / 87% / 0.75
- The designed punchline: overall MRR ties at 0.75, recall@1 favors sparse,
  recall@3 favors dense. The overall row cannot pick a winner; only the
  category breakdown can, which is Module 9's aggregate-hides-the-story lesson
  restated about retrievers.
- The keyword queries deliberately embed the identifier in a natural symptom
  sentence ("do I need a DR-4410 if my pages have streaks"), because that is
  how users write and because the symptom words are what drag the dense
  embedding toward the wrong articles. Pure identifier-only queries would let
  the dense encoder score too well through subword matching.
- The bar chart (`retrieval_comparison.png`) plots recall@3 per category plus
  overall; the slide image is the actual output file.

### Extra credit

- Reciprocal rank fusion (hybrid), BM25 TF saturation, watching IDF zero out a
  ubiquitous term, assembling a grounded RAG prompt with the Module 6 chat
  template, and re-embedding the corpus with the Module 5 checkpoint's hidden
  states to measure what the contrastive objective contributes.

## Quiz answer notes

- IDF and Module 1: log(N/df) is the surprisal of "a document contains this
  term" because df/N estimates that event's probability.
- Dense misses E-341: subword tokenization plus mean pooling averages the
  identifier away, and sibling codes embed nearly identically; sparse keeps the
  literal term as a heavily weighted dimension.
- Sparse misses the paraphrase: TF-IDF has no notion of synonymy; zero shared
  terms means zero score. The embedding space supplies the missing structure.
- Diagnosing RAG: cut at the pipeline seam; recall@k/MRR for retrieval,
  faithfulness (LLM judge) for generation.
- Identical overall scores: the per-category table; Module 9's per-task lesson.
- Contrastive objective: CLIP (Module 8), image and text encoders in one space.
- Finetune versus retrieval for daily-changing facts: diffuse storage, retrain
  latency, no audit or access control; re-indexing wins.
- Prompted versus trained chain-of-thought: elicited at inference versus
  optimized into the weights by RL (Module 7).
- Verifier versus better steps: compounding is multiplicative; verification
  makes errors retryable instead of fatal (0.95^20 = 0.36 vs 0.97^20 = 0.54).
- Prompt injection: one token stream, learned rather than enforced hierarchy;
  defenses must live outside the model.

## References

- Brown et al., "Language Models are Few-Shot Learners," arXiv:2005.14165.
- Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language
  Models," arXiv:2201.11903.
- Kojima et al., "Large Language Models are Zero-Shot Reasoners,"
  arXiv:2205.11916.
- Olsson et al., "In-context Learning and Induction Heads," arXiv:2209.11895.
- Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP
  Tasks," arXiv:2005.11401.
- Spärck Jones, "A Statistical Interpretation of Term Specificity and its
  Application in Retrieval," Journal of Documentation 28(1), 1972,
  doi:10.1108/eb026526.
- Robertson and Zaragoza, "The Probabilistic Relevance Framework: BM25 and
  Beyond," Foundations and Trends in Information Retrieval, 2009,
  doi:10.1561/1500000019.
- Karpukhin et al., "Dense Passage Retrieval for Open-Domain Question
  Answering," arXiv:2004.04906.
- Reimers and Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese
  BERT-Networks," arXiv:1908.10084.
- Cormack, Clarke, and Buettcher, "Reciprocal Rank Fusion Outperforms Condorcet
  and Individual Rank Learning Methods," SIGIR 2009.
- Malkov and Yashunin, "Efficient and Robust Approximate Nearest Neighbor
  Search Using Hierarchical Navigable Small World Graphs," arXiv:1603.09320.
- Liu et al., "Lost in the Middle: How Language Models Use Long Contexts,"
  arXiv:2307.03172.
- Schick et al., "Toolformer: Language Models Can Teach Themselves to Use
  Tools," arXiv:2302.04761.
- Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models,"
  arXiv:2210.03629.
- Yao et al., "tau-bench: A Benchmark for Tool-Agent-User Interaction in
  Real-World Domains," arXiv:2406.12045.
- Jimenez et al., "SWE-bench: Can Language Models Resolve Real-World GitHub
  Issues?," arXiv:2310.06770.
- The Model Context Protocol, modelcontextprotocol.io.
- Anthropic, "Building Effective Agents,"
  anthropic.com/research/building-effective-agents.
- Simon Willison, prompt injection series,
  simonwillison.net/series/prompt-injection/.
- FAISS, github.com/facebookresearch/faiss.

## Image credits

- `sparck_jones.jpg`: reused from Module 9's images (same photograph).
- `lewis_kiela.jpg`: composite of Patrick Lewis's portrait from his personal
  site (patricklewis.io) and Douwe Kiela's photo from his academic homepage
  (douwekiela.github.io).
- `shunyu_yao.jpg`: from his academic homepage (ysymyth.github.io).
- `retrieval_comparison.png`: actual output of the exercise solution run.
