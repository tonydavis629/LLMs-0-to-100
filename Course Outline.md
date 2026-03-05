# Course Outline

### Content/citations/diagrams/links

(Moved to individual modules below)

## Each class should include

- Lecture
- Code exercise / demo (embed into lecture and release separately)
- Notable figures
- Side quest: interesting information not critical to main point

---

## 1. Course Introduction

### Lecture

- a. Introductions
  - Grace Hopper wrote the first compiler -- a program that translates human-readable code into machine instructions. LLMs are the next step: natural language compilers that translate plain text into tasks. They are language agnostic and not limited to code.
  - Course overview/structure
  - Expect: programming, history, math, challenging learning tasks
  - Resources (slides, code, books, links)
- b. We built talking machines. Why now? How?
  - Information theory (1940s-50s) -- Shannon and von Neumann laid the mathematical foundations. Shannon proved that communication could be quantified in bits, and that any signal could be transmitted reliably through a noisy channel. This gave us a formal way to think about language as information.
  - GOFAI (1950s-80s) -- "Good Old-Fashioned AI." Hand-written rules and symbolic logic. Expert systems, search trees, knowledge bases. Worked for narrow tasks but couldn't handle ambiguity or scale to real language.
  - AI summers and winters -- cycles of hype and disappointment. Funding surged when demos impressed, collapsed when promises weren't met. Pattern repeated multiple times from the 1960s through 2000s.
  - Deep learning (2000s-present) -- instead of writing rules, learn patterns from data. Neural networks with many layers trained on large datasets.
  - AlexNet and GPUs (2012) -- Krizhevsky, Sutskever, and Hinton used an NVIDIA GPU to train a deep neural network that crushed the ImageNet competition. Proved that GPUs could make deep learning practical. Jensen Huang bet NVIDIA's future on this.
  - Transformers (2017) -- Vaswani et al., "Attention Is All You Need." Replaced recurrence with self-attention, enabling massive parallelism. The architecture behind every modern LLM.
  - Masked language modeling and next token prediction -- two strategies for learning language. Predict a hidden word (BERT, Devlin et al.) or predict the next word (GPT, Radford et al.). Ilya Sutskever and others showed that next token prediction at scale produces surprisingly capable models.
  - InstructGPT (2022) -- OpenAI trained GPT to follow instructions using human feedback. Turned a text predictor into an assistant.
  - RL post-training -- reinforcement learning aligns models to desired behavior. Originally RLHF (human feedback as reward), now broader: GRPO, DPO, and other algorithms that don't require a separate reward model.
- c. Language as prediction (Shannon 1948)
  - The core insight: language is not random. Given context, the next letter or word is partially predictable. Shannon demonstrated this by asking humans to guess the next character in a sequence -- they could do it far better than chance.
  - This is the same idea behind modern LLMs, just at vastly greater scale and with learned representations instead of counted frequencies.
  - Transition: "All of this starts with one idea -- predicting the next symbol. Let's go back to 1948 and build it ourselves."

### Notable Figures

- Claude Shannon -- information theory, entropy, the bit
- Grace Hopper -- first compiler, bridging human/machine language
- Jensen Huang -- GPU computing enabling deep learning at scale

### Side Quests

- ELIZA (Weizenbaum, 1966) -- simple pattern matching that convinced people it was a therapist. The "ELIZA effect": people attribute understanding to systems that have none. A reminder to be skeptical.

### Resources

- <https://huggingface.co/learn/llm-course/chapter1/1> -- HuggingFace LLM course, broad companion resource

### Exercise: N-gram Language Models (Alice in Wonderland)

- Set up uv python environment (used in whole course)
- Uniform random letters (0th-order)
- Character unigrams (1st-order)
- Character bigrams/trigrams (2nd/3rd-order Markov)
- Word unigrams
- Word bigrams/trigrams
- Stretch: compute cross-entropy/perplexity at each order
- Quiz: character frequencies, model order identification, conditional probability, entropy intuition

---

## 2. Perceptrons and Optimization

### Lecture

- a. The neuron model
  - WX + B
  - Universal approximation theorem
  - Backpropagation
- b. The matrix -- graph equivalence principle
  - Brain is a network -- can be described as a matrix
  - GPUs are good at matrix multiplication
- c. XOR problem visualization
- d. Solving the XOR problem
- e. Folding in high dimensional space
- f. Gradient descent
  - Stochastic gradient descent, batching
  - Other optimizers
- g. Loss landscape visualizations

### Notable Figures

- (TBD)

### Side Quests

- Embedding geometry and embedding arithmetic

### Resources

- <https://github.com/MITDeepLearning/introtodeeplearning> -- MIT intro to deep learning labs (neural nets, optimization)

### Exercise: Text classifier and visualization

---

## 3. Attention Mechanisms

### Lecture

- a. Q, K and V
  - Database analogy
  - V * softmax(QK^T/sqrt(d))
- b. Multihead self attention
- c. Cross attention
- d. Positional embedding and permutation variance
- e. Trade offs between memory and compute
- f. KV cache

### Notable Figures

- (TBD)

### Side Quests

- (TBD)

### Resources

- <https://jalammar.github.io/illustrated-transformer/> -- visual guide to attention and encoder-decoder
- <https://codecut.ai/bertviz-visualize-attention-in-transformer-language-models/> -- BertViz attention visualization tool

### Exercise: Attention visualization and positional embedding visualization

---

## 4. Transformer Architectures

### Lecture

- a. Encoder -- Decoder (Vaswani 2017)
  - Seq2seq
  - Positional encoding
- b. Encoder Only, BERT
- c. Decoder only, GPT

### Notable Figures

- (TBD)

### Side Quests

- (TBD)

### Resources

- <https://github.com/karpathy/build-nanogpt> -- build GPT-2 from scratch, step by step
- <https://github.com/rasbt/LLMs-from-scratch> -- full LLM implementation book (chapters 2-4 most relevant here)
- <https://nlp.seas.harvard.edu/annotated-transformer/> -- annotated encoder-decoder implementation
- <https://poloclub.github.io/transformer-explainer/> -- interactive GPT-2 architecture walkthrough
- <https://www.krupadave.com/articles/everything-about-transformers?x=v3> -- comprehensive visual transformer guide

### Exercise: NanoGPT

---

## 5. Pretraining

### Lecture

- a. Masked language modeling
  - Bidirectional
- b. Autoregressive / causal MLM benefits

### Notable Figures

- (TBD)

### Side Quests

- (TBD)

### Exercise: Pretraining NanoGPT

---

## 6. Finetuning

### Lecture

- a. What is finetuning, really?
  - Same algorithms (almost), different data
- b. InstructGPT -- birth of the assistant
- c. Parameter-efficient methods (LoRA)

### Notable Figures

- (TBD)

### Side Quests

- (TBD)

### Exercise: Finetune NanoGPT

---

## 7. Reinforcement Learning Post Training

### Lecture

- a. What is RL, really?
  - Same optimization algorithms, different target data
  - Reframes as agent / environment
- b. RL for LLMs
  - Why? Training data limited -- need specific behavior
- c. Value/Reward models
  - RLHF
- d. Value free models
  - GRPO

### Notable Figures

- (TBD)

### Side Quests

- (TBD)

### Exercise: RL environment -- train model to use tool

---

## 8. Multimodal Models

### Lecture

- a. Vision
  - Not a sequence
  - CLIP
- b. Video
- c. Audio
- d. RF

### Notable Figures

- (TBD)

### Side Quests

- Visual registers -- are they needed? What are they?

### Exercise: Align CLIP embeddings with NanoGPT using image-caption dataset

---

## 9. LLM Deployment

### Lecture

- a. Bottleneck is HBM, why?
- b. MoE
  - More sparse, fits hybrid GPU/CPU architecture
  - Trades off lot less cost for little less speed
- c. vLLM use and deployment

### Notable Figures

- (TBD)

### Side Quests

- (TBD)

### Exercise: Server with vLLM, then write API client to it

---

## 10. Practical Applications and Case Studies

### Lecture

- a. In context learning
- b. RAG
- c. Agentic LLM
  - Codex / Claude Code

### Notable Figures

- (TBD)

### Side Quests

- (TBD)

### Exercise: Adding RAG and tool to our vLLM model

---

## 11. Evaluation and Benchmarking

### Lecture

- a. Metrics
  - Perplexity
- b. Benchmarks
  - GLUE, MMLU, etc

### Notable Figures

- (TBD)

### Side Quests

- (TBD)

### Exercise: Evaluate NanoGPT on benchmark, red team prompt injection

- Write tools, share with the class
- Try to own other's LLM with tool prompts to demonstrate MCP/tool security

---

## 12. The Future of LLMs

### Lecture

- a. Scaling laws, universal function approximation theorem again
- b. SSM / Mamba
- c. RWKV
- d. GNNs
- e. Multi token prediction
- f. Nested optimizers, continual learning
- g. Diffusion

### Notable Figures

- (TBD)

### Side Quests

- (TBD)

### Exercise: (TBD)
