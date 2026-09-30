# Module 4: LLM Architectures — Lecture Notes

Citations, math, and explanations for every claim in the presentation.

The deck order is: the recurrent era, tokenization, generating text (sampling), the full decoder-only walkthrough ("Putting It All Together"), beyond the vanilla transformer, and finally the three architectural families. Sampling is taught before the walkthrough so that the walkthrough can end at "Step 6: Sampling" and refer back to it. The architectural families close the conceptual content before the exercise.

## Before Transformers: The Recurrent Era

### Recurrent Neural Networks
- An RNN maintains a hidden state that is updated at each time step: $\mathbf{h}_t = f(\mathbf{h}_{t-1}, \mathbf{x}_t; W)$.
- The same weight matrix $W$ is reused at every step, giving a fixed parameter count regardless of sequence length.
- The hidden state is a fixed-size vector that must compress all previous context.

### The Problems with Recurrence
- **Vanishing/exploding gradients:** Backpropagation through time unfolds the recurrence into a deep network. Gradients involve repeated multiplication by the same weight matrix. If its largest singular value is less than 1, gradients vanish; if greater than 1, they explode.
- **Sequential dependency:** Token $t$ cannot be computed until tokens $1$ through $t-1$ are complete. This makes training impossible to parallelize across the sequence dimension.
- **Fixed-size bottleneck:** No matter how long the input, all information must be compressed into a hidden vector of fixed dimension.

### LSTMs and GRUs
- Hochreiter and Schmidhuber (1997) introduced gating mechanisms: forget, input, and output gates control information flow using sigmoid-activated multiplicative gates.
- The cell state can carry information across many time steps with near-unit gradient because the forget gate multiplies by values close to 1.
- Gated Recurrent Units (Cho et al., 2014) simplified LSTMs to two gates (update and reset) with fewer parameters.
- Both mitigated gradient problems but kept sequential processing and the bottleneck.

### Sequence-to-Sequence
- Sutskever et al. (2014) split translation into an encoder RNN and a decoder RNN.
- The encoder compresses the input into a single context vector; the decoder generates from it.
- Bahdanau attention (2014) was invented to fix the fixed-vector bottleneck by letting the decoder attend to all encoder states.

### Attention Is All You Need
- Vaswani et al. (2017) removed recurrence entirely, replacing it with multi-head self-attention.
- Full parallelism: all positions are processed simultaneously during training.
- Constant path length: any two tokens interact directly in one attention layer, not through $O(n)$ recurrent steps.
- **Manim animation (`recurrence-vs-attention`):** contrasts the two regimes on a five-token sentence. In the recurrence pass a single hidden state is carried along the chain and a marker travels token-by-token, making the $O(n)$ path from the first token to the last explicit. In the attention pass every earlier token is linked to the last token directly, illustrating the $O(1)$ path length and full parallelism.

- **Manim animation (`parallel-forward`):** inference on "The capital of France is" with GPT-2 medium ($T = 5$ tokens: 464, 3139, 286, 4881, 318; $d = 1024$; 24 blocks; $V = 50{,}257$). The embedded prompt is a matrix $X \in \mathbb{R}^{T \times d}$. Every block maps a $T \times d$ matrix to a $T \times d$ matrix, and the animation draws each output as its own matrix: $H_1 = \text{Block}_1(X)$, then after blocks 2 to 23, $H_{24}$. All five rows are processed together as batched matrix multiplies rather than five sequential steps. The mask panel is tethered to Block 1: inside each attention layer the scores $QK^\top/\sqrt{d_k}$ form a $T \times T$ matrix; the causal mask adds $-\infty$ above the diagonal, so after the softmax row $i$ puts zero weight on positions $j > i$ (Vaswani et al., 2017, Section 3.2.3). The attention weights drawn in that grid are illustrative. The LM head is one linear map applied to each row independently (the same weights for every position), so $T$ rows in gives $T$ rows of $V$ scores out: logits $\in \mathbb{R}^{T \times V}$. Row $i$ is the model's score for the token at position $i+1$. The animation labels this explicitly: rows 1 to 4 "predict" capital, of, France, is, which are already in the prompt (these rows are what training uses, see the preview slide), and only row 5 predicts something new. The outlined cells on the shifted diagonal are the true next tokens. Logit cells show eight vocabulary columns in token-ID order (the order the LM head emits them in, not sorted by score): "ing" 278, " of" 286, " is" 318, " capital" 3139, " happy" 3772, " France" 4881, " Paris" 6342, " banana" 25996, plus an ellipsis for the rest; "ing", " happy" and " banana" are unrelated distractors. Fill brightness is each raw logit scaled between the row's lowest and highest shown value (raised to the power 1.6 to separate the top cell), so brighter means a higher score. In every row the brightest of these eight columns is the true next token, which the animation outlines. All numbers are real GPT-2 medium outputs (Hugging Face `gpt2-medium`, run in float64 because float32 matrix multiplies produced NaN logits on the build machine). The softmax of the last row gives " Paris" 0.174, " the" 0.054, " Lyon" 0.042, " not" 0.031, " a" 0.026. For comparison GPT-2 small ranks " the" first (0.085) and " Paris" at 0.032, which is why the larger model is used here. During generation the chosen token is appended and the pass repeats; with a KV cache later steps only process the new token, but the first pass over the prompt (the "prefill") is exactly this parallel computation.

### The Transformer Architecture (overview slide)
- Uses the canonical figure from Vaswani et al. (2017): an encoder stack (bidirectional self-attention + feed-forward) on the left and a decoder stack (masked self-attention + cross-attention + feed-forward) on the right.
- The slide names the shared building blocks — attention, feed-forward, residual connections, normalization — and states that the rest of the module follows the decoder-only variant used by modern LLMs.

## Tokenization

### The Unit Problem
- Characters: tiny vocabulary, very long sequences, low information per token.
- Words: meaningful units, but vocabulary explodes and any unseen word is out-of-vocabulary.
- Subwords: frequent words stay whole, rare words split into reusable pieces.

### Byte-Pair Encoding
- BPE starts from characters and iteratively merges the most frequent adjacent pair.
- The learned list of merges defines the vocabulary. Every input can be decomposed into vocabulary tokens.
- The BPE animation uses the toy corpus "low low lower lowest." It first counts adjacent pairs, then merges `lo`, then merges `low`, showing how "lower" and "lowest" reuse the frequent `low` prefix.
- GPT-2 uses byte-level BPE: the merge process runs over raw bytes, so every Unicode string is representable.
- GPT-2 vocabulary: ~50k tokens. Modern models use 100k&ndash;200k.

### Byte-Level BPE (UTF-8)
- Text is first encoded as UTF-8, which writes each Unicode character as 1 to 4 bytes: ASCII characters take one byte, "é" (U+00E9) takes two, `c3 a9`. The slide shows the bytes in hexadecimal.
- Byte-level BPE (Radford et al., 2019) starts from the 256 possible byte values, so any string can be encoded and no unknown-token fallback is needed; merges are then learned over byte sequences.
- The slide example "The model reads café" is 21 UTF-8 bytes and 4 GPT-2 tokens: 464 "The", 2746 " model", 9743 " reads", 40304 " café" (verified with the Hugging Face `gpt2` tokenizer).

### Special Tokens
- Special tokens are reserved vocabulary entries that the tokenizer never produces from ordinary text; a chat template or the tokenizer inserts them.
- **Top strip:** the exact output of Llama 3's chat template for one user question and one assistant answer, tokenized with the Llama 3 tokenizer (checked with Hugging Face `NousResearch/Meta-Llama-3-8B-Instruct`, a mirror of Meta's tokenizer).
- **Bottom strip:** a Llama 3.1 built-in tool call, `<|python_tag|>brave_search.call(query="Paris weather")<|eom_id|>`, tokenized with the Llama 3.1 tokenizer. In Meta's Llama 3.1 prompt format, `<|python_tag|>` starts a tool call and `<|eom_id|>` ends the message without ending the turn, so the tool's output can be returned in an `ipython` turn and the model continues (Meta, "Llama 3.1 model card and prompt format", 2024).
- **Table:** Llama 3.1 IDs (verified with `NousResearch/Meta-Llama-3.1-8B-Instruct`): `<|begin_of_text|>` 128000, `<|end_of_text|>` 128001, `<|finetune_right_pad_id|>` 128004, `<|start_header_id|>` 128006, `<|end_header_id|>` 128007, `<|eom_id|>` 128008, `<|eot_id|>` 128009, `<|python_tag|>` 128010. "Same idea elsewhere": BERT `[CLS]` 101 and `[PAD]` 0 (`bert-base-uncased`); GPT-2 `<|endoftext|>` 50256; ChatML (used by Qwen) `<|im_start|>` 151644 and `<|im_end|>` 151645; Qwen2.5 wraps tool calls in `<tool_call>` 151657 and `</tool_call>` 151658. The Qwen pair marks the span of the call, which is close to, but not identical to, Llama's start token plus pause token.
- **BOS** gives position 1 an input, so the first real token is itself predicted. GPT-2 has no separate BOS; its single `<|endoftext|>` separates documents in training and so plays both roles (Radford et al., 2019). The Hugging Face GPT-2 tokenizer does not prepend it by default.
- **PAD** fills shorter sequences in a batch; an attention mask keeps real tokens from attending to pads and the loss ignores them. Byte-level BPE vocabularies need no UNK token because any byte sequence can be encoded.

### Tokenization in Practice
- Token counts drive context limits and API cost.
- Non-English text often requires more tokens per sentence (the "token tax").
- Models struggle with letter counting, string reversal, and digit arithmetic because these operations are not natural at the token level.
- Tokenizers and models are trained separately. Glitch tokens (rare tokens with near-zero training frequency) can cause unpredictable behavior.

### Glitch Tokens (textual side quest)
- The slide is prose plus a short worked example exchange (prompt asking the model to repeat the glitch token, model substituting an unrelated word); the earlier card diagram was removed.
- The canonical example is <code>&nbsp;SolidGoldMagikarp</code> (a Reddit username; the leading space is part of the token in many BPE vocabularies).
- Mechanism: the tokenizer and the language model are trained separately. A string can be frequent enough to enter the tokenizer vocabulary while later model training supplies too few contexts to learn a reliable embedding. Its embedding row stays close to its random initialization.
- Rumbelow and Watkins (2023) documented that prompts asking GPT-2/GPT-3 models to repeat or explain anomalous tokens could produce evasions, substitutions, spelling-like output, and other unstable completions. Source: <https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation>.
- Land and Bartolo (2024) formalize the issue as under-trained tokens: tokens present in the tokenizer vocabulary but nearly or entirely absent during model training. Source: <https://aclanthology.org/2024.emnlp-main.649/>.
- Pedagogical point: this makes the abstraction boundary concrete. The model does not receive the characters in "SolidGoldMagikarp"; it receives a token ID and a learned vector for that ID.

## Embeddings

This section follows tokenization directly: once text is a list of token IDs, the next question is how an ID becomes a vector.

### Step 1: Words to Vectors
- Token ID $t_i$ indexes row $W_E[t_i]$ of the embedding matrix: $\mathbf{x}_i = W_E[t_i]$.
- The slide shows the full handoff from words to subword tokens, token IDs, and embedding rows. The transformer receives vectors, not raw text.
- **Manim animation (`embedding-lookup`):** "France" maps to id 4881, the id selects one row of the embedding matrix, and that row is pulled out as the token's vector.

### Where the Embedding Matrix Comes From
- **Diagram:** three panels. (1) Initialize: a Gaussian (normal) curve with mean 0 and standard deviation 0.02, the distribution GPT-2 samples each entry from; about 95% of draws fall within $\pm 0.04$ (two standard deviations). (2) $E$ at step 0: rows The, capital, of, France filled with real draws (`torch.manual_seed(0)`, `torch.randn(...) * 0.02`); the numbers carry no information about the tokens. (3) $E$ after training: the first three entries of GPT-2 small's actual rows 464, 3139, 286, 4881. Trained GPT-2 embeddings have standard deviation about 0.14, roughly seven times the initial 0.02.
- $E \in \mathbb{R}^{V \times d}$; for GPT-2 small $V = 50{,}257$, $d = 768$, so $E$ holds $50{,}257 \times 768 = 38{,}597{,}376$ parameters.
- Lookup equals a one-hot product: if $\mathbf o_k$ is the one-hot vector for token $k$, then $\mathbf o_k^\top E$ is row $k$. Frameworks implement it as indexing (`nn.Embedding`) because multiplying by a mostly-zero vector wastes compute.
- Initialization: GPT-2's Hugging Face config sets `initializer_range = 0.02`, and token embeddings are drawn from $\mathcal N(0, 0.02^2)$ (Radford et al., 2019; `GPT2PreTrainedModel._init_weights`). Random initialization breaks symmetry; the initial vectors carry no information about the tokens.
- Training: $E$ is updated by backpropagation like every other weight. For a batch, $\partial \mathcal L / \partial E$ is nonzero only in the rows of tokens that occur in the batch (sparse gradient), so rare tokens are updated rarely. This is the mechanism behind under-trained "glitch tokens" (see the tokenization side quest).
- Weight tying: GPT-2 shares $E$ with the output projection, logits $= \mathbf h E^\top$ (Press & Wolf, 2017, "Using the Output Embedding to Improve Language Models").

### Vectors Are Coordinates
- **Diagram:** left, cat and dog vectors separated by a small angle with Paris pointing elsewhere (schematic); right, the word2vec parallelogram man to woman, king to queen (schematic, after Mikolov et al., 2013).
- **Interactive (`cosineDial`):** two vectors $\mathbf a$, $\mathbf b$ lie on a floor plane drawn in perspective with a grid (dims 1 and 2); dragging either tip changes its direction and length. $\cos\theta$ is drawn as a shaded 3-D bar on the vertical axis (ticks at -1, -0.5, 0.5, +1). Changing a vector's length leaves the bar unchanged, which shows that cosine similarity ignores magnitude.
- A vector in $\mathbb{R}^d$ is a point with $d$ coordinates. Similar tokens receive similar gradient signals because they appear in similar contexts (the distributional hypothesis: Harris, 1954; Firth, 1957), so their rows end up close.
- Cosine similarity $\cos\theta = \mathbf a \cdot \mathbf b / (\lVert \mathbf a \rVert \lVert \mathbf b \rVert)$ compares direction and ignores length; it is the standard similarity for embeddings.
- Linear offsets: Mikolov, Yih & Zweig (2013), "Linguistic Regularities in Continuous Space Word Representations," and Mikolov et al. (2013), "Distributed Representations of Words and Phrases and their Compositionality," showed that $\text{vec}(\text{king}) - \text{vec}(\text{man}) + \text{vec}(\text{woman})$ is nearest to $\text{vec}(\text{queen})$, and that country-to-capital offsets are roughly parallel. These results are for word2vec vectors; transformer input embeddings show weaker and noisier versions of the same structure.
- Projection to 2-D: PCA keeps the directions of largest variance; t-SNE (van der Maaten & Hinton, 2008) keeps local neighborhoods but distorts global distances.
- **Manim animation (`embedding-space`):** a toy $d = 2$ embedding table. Rows start as random noise near the origin; one row is read as $(x, y)$ coordinates; training moves the points into neighborhoods (countries, capitals, animals); country-to-capital arrows are parallel; a final note gives GPT-2 small's $d = 768$. All coordinates are schematic, not measured from a model.

## Putting It All Together

A single prompt is followed through a decoder-only transformer. This part has no divider slide: Steps 2 to 6 continue straight on from the Embeddings section (Step 1), and the residual-stream slides come before Step 6 so that Step 6 (sampling) leads directly into the "Generating Text" section. The architecture-only slide (`decoder-only-arch`) sits at the end of the "Before Transformers" section, right after the original transformer figure, so students see the decoder-only stack before tokenization and embeddings are covered in detail. It shows the full stack bottom-to-top: the input text ("The capital of France"), a single Tokenize step (text to token vectors), added positional encoding, an $N\times$ block of masked multi-head self-attention and a feed-forward network with Add & Norm around each (the residual skip connections are drawn explicitly), then the unembedding (linear) layer and softmax to next-token probabilities. The overview diagram collapses tokenization and embedding into one box and omits the final LayerNorm for clarity (normalization has its own slide); the detailed Step 6 still notes the final layer norm that precedes the head. The recurring pipeline locator has six boxes: **word &rarr; vector**, add position, multi-head attention, feed-forward network, repeat blocks, sampling. Tokenization is collapsed into the first box because it has its own earlier section; the first box represents the entire word-to-vector embedding stage. The key concept slides (embedding, feed-forward, normalization, residual stream) are each paired with a dedicated step-through Manim animation.

### Step 2: Add Position
- Positional embedding $W_P[i]$ adds order information: $\mathbf{x}_i = W_E[t_i] + W_P[i]$.
- Without a position signal, self-attention is permutation-equivariant: the same token vectors in a different order would carry no ordering information.

### Step 3: Multi-Head Self-Attention
- The only sub-layer where information moves between positions in the same layer.
- $\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V$.
- Each token is projected to a query, key, and value; scores are scaled dot products; the causal mask keeps only the lower triangle so a token never attends to the future; softmax produces weights; the output is the weighted sum of values. Several heads run in parallel.
- Wrapped in a residual connection and normalization.

### Step 4: Position-Wise Feed-Forward Network
- Two linear layers with a ReLU nonlinearity, applied independently at every position: $\text{FFN}(x) = W_2 \operatorname{ReLU}(W_1 x)$, with $d \rightarrow 4d \rightarrow d$.
- The slide and animation use ReLU for clarity; GPT-2 actually uses GELU (a smooth ReLU), and modern models use SwiGLU.
- Typically $d_{\text{ff}} \approx 4 \cdot d_{\text{model}}$.
- Stores much of the model's factual knowledge (Geva et al., 2021).
- **Manim animation (`ffn-expand`):** a small MLP projecting the token vector up into a wider hidden layer, applying a ReLU nonlinearity that zeroes some units, then projecting back down.

### Step 5: Repeat the Block
- Depth lets later blocks build on earlier features. GPT-2 small repeats the decoder block 12 times. Each block reuses the same shape with its own weights.

### Normalization
- LayerNorm: $\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$.
- Post-norm: normalize after sub-layer output (original transformer).
- Pre-norm: normalize before sub-layer input (modern default, more stable at depth).
- RMSNorm: drop mean centering, divide by the root-mean-square.
- **Manim animation (`norm-demo`):** a vector with both an offset mean and varied scale; LayerNorm recenters to zero mean and rescales to unit variance, while RMSNorm only rescales (no recentering).

### The Residual Stream
- Placed after Step 6 so the full forward pass is in view before introducing the interpretability lens.
- Each sub-layer reads from the running vector, computes its contribution, and writes it back: $\mathbf{x} = \mathbf{x} + \text{sub-layer}(\text{norm}(\mathbf{x}))$.
- Because each block only adds to this running vector, the original embedding is never overwritten and gradients flow back through the addition without decaying.
- **Manim animation (`residual-stream`):** the residual stream is drawn as a horizontal "highway" from the embedding to the logits. Each attention and MLP sub-layer branches off via a vertical read arrow, computes an update, and writes it back at an addition node on the stream — the standard mechanistic-interpretability picture (cf. Elhage et al., "A Mathematical Framework for Transformer Circuits," 2021).

### Side Quest: Induction Heads
- **Induction heads** (Olsson et al., 2022, "In-context Learning and Induction Heads," <https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html>; mechanism first described in Elhage et al., 2021) implement copy-and-continue: given `[A] [B] ... [A]`, predict `[B]`.
- **Figure:** the two-head circuit on "Mr Dursley was proud . Later , Mr ?" (the "Mr D urs ley" example is the one used by Elhage et al.). (1) A previous-token head in an earlier layer writes into each position information about the token before it, so the Dursley position carries "I came after Mr". (2) The induction head at the second "Mr" forms a query that matches keys carrying "came after Mr", so it attends to Dursley. (3) Its value/output circuit copies Dursley's identity, raising the logit for "Dursley".
- Because the match is made on content in the context, the circuit works for token pairs never seen together in training.
- Olsson et al. observe a "phase change" early in training: induction heads form abruptly, at the same point in-context learning performance improves sharply.

### Step 6: Sampling
- The slide renames the former "logits to probabilities" step. Final layer norm, then the **unembedding** layer $W_{\text{head}}$ (also called the LM head) projects to vocabulary-sized logits &mdash; the mirror image of the input embedding lookup; softmax gives the next-token distribution.
- Weight tying: $W_{\text{head}} = W_E^{\top}$ saves parameters and ties input/output semantics, which is why the unembedding can be seen as reusing the embedding matrix in reverse.
- The decoding strategy from the earlier sampling section (greedy, temperature, top-k, top-p) then chooses the next token, which is appended and fed back in.

### Parameter Counting (GPT-2 small)
- $d = 768$, $N = 12$, $V = 50257$.
- Embeddings: $V \cdot d \approx 38.6$M.
- Attention per layer: $4d^2$ (Q/K/V projection + output projection) = ~2.36M. Total: $N \cdot 4d^2 \approx 28.3$M.
- FFN per layer: $8d^2$ (two linear layers, $d \to 4d \to d$) = ~4.71M. Total: $N \cdot 8d^2 \approx 56.6$M.
- Total: ~124M parameters.

## Generating Text: Sampling and Decoding

This section precedes the architecture walkthrough so that the walkthrough's final step can simply refer back to it.

### From Logits to Text
- A forward pass yields a probability distribution over the vocabulary for the next token: $p_i = e^{z_i} / \sum_j e^{z_j}$.
- Both $i$ and $j$ index the vocabulary: $z_i$ is the logit for the candidate token $i$ being scored, and the denominator sums $e^{z_j}$ over every token $j$ so the probabilities normalize to one.
- Decoding chooses one token, appends it, and repeats.

### Greedy Decoding
- Always select $\arg\max_i p_i$.
- Deterministic but produces repetitive, flat text.

### Temperature
- Scale logits by $T$ before softmax: $p_i \propto e^{z_i / T}$.
- $T < 1$: sharper, more conservative.
- $T > 1$: flatter, more random.

### Top-k and Top-p
- Top-k: keep only the $k$ highest-probability tokens.
- Top-p (nucleus): keep the smallest set whose cumulative probability exceeds $p$.
- Usually combined with temperature scaling.
- **Interactive (`sampling-explorer`):** twelve candidate tokens with illustrative logits (6.6 down to 1.5; realistic in shape, not measured from a model). Under each bar the widget prints the token, its score (logit divided by the temperature $T$), and the cumulative probability in rank order. An orange line plots that cumulative probability against a 0 to 100% axis, and when top-p is below 1 a dashed line marks $p$: top-p keeps tokens up to the first one whose cumulative probability reaches $p$. A dashed vertical cut marks where top-k or top-p truncates; everything right of it has probability 0 after renormalization. Order of operations matches common serving stacks: temperature, then top-k, then top-p, then renormalize.

### Beam Search
- Keep $b$ best partial sequences at each step.
- The slide draws this as a left-to-right tree with $b = 2$: the two surviving beams are solid orange paths, and the pruned (lower-scoring) branches are dashed.
- Best for constrained tasks (translation) where a single correct answer exists.
- Poor for open-ended generation, where the most likely sequence is usually vacuous or repetitive.

- **Interactive widget (`:::interactive widget="samplingExplorer"`):** twelve candidate continuations with fixed logits, run through the production order &mdash; divide by temperature, softmax, truncate by top-k and top-p, renormalize, sample. The outlined bars are the post-temperature distribution and the filled bars the post-truncation one, so the truncated tail is visibly set to zero rather than merely small. The readout reports the size of the kept set and the entropy $H = -\sum_i p_i \log_2 p_i$ of the final distribution. Drawing 200 samples overlays the empirical frequencies, which shows sampling noise against the distribution the model actually specified.

## Beyond the Vanilla Transformer

### Modern Block (Llama-style)
- RMSNorm instead of LayerNorm.
- SwiGLU instead of ReLU/GELU.
- RoPE for positional embeddings.
- Pre-norm architecture.
- Dropped bias terms in linear layers.

### Mixture of Experts
- Replace FFN with many expert FFNs + router.
- Router selects top-$k$ experts per token.
- Far more parameters, roughly same compute per token.
- Sparse activation also changes memory serving: inactive experts may be staged across CPU memory, GPU memory, or devices, while active experts for the current batch need low-latency access.
- Mixtral, DeepSeek.
- Router (Mixtral form): $g(\mathbf x) = \text{softmax}(\text{TopK}(\mathbf x W_g))$, where TopK sets all but the $k$ largest logits to $-\infty$ (Jiang et al., 2024, "Mixtral of Experts," arXiv:2401.04088). Mixtral 8x7B has 8 experts per layer and routes each token to 2; each token can reach 47B parameters but uses 13B active parameters. Earlier sparse MoE: Shazeer et al. (2017), "Outrageously Large Neural Networks"; Fedus, Zoph & Shazeer (2021), "Switch Transformers" ($k = 1$).
- **Diagram (`moe` slide):** one decoder block in the same style as the architecture overview: masked multi-head self-attention (dense: every token uses all of its weights), Add & Norm, then an MoE layer in place of the feed-forward network, Add & Norm, repeated $N$ times. A callout opens the MoE layer for a single token: the router scores 8 experts, only experts 3 and 6 run, and their outputs are summed with gate weights 0.62 and 0.38 (illustrative). Attention stays dense in Mixtral-style MoE models; only the FFN is replaced by experts (Jiang et al., 2024; Fedus et al., 2021).

### Sub-Quadratic Alternatives
- Mamba: state-space model with input-dependent gates. Linear in sequence length.
- Linear attention: kernelized softmax avoiding the $n \times n$ matrix.
- RWKV: combines recurrence with linear attention.
- Hybrids like Jamba mix attention and state-space layers.

## The Three Architectural Families

Placed last, after the full walkthrough and the modern-block survey, so the comparison can draw on everything seen so far.

### Encoder-Decoder
- Original transformer (Vaswani et al., 2017).
- Encoder: bidirectional attention, builds representations.
- Decoder: causal self-attention + cross-attention to encoder output.
- Natural fit: translation, summarization. T5, BART.

### Encoder-Only
- Bidirectional attention, no causal mask.
- Trained with masked language modeling.
- Produces representations, not generations.
- BERT (Devlin et al., 2018), RoBERTa.

### Decoder-Only
- Causal masking, autoregressive next-token prediction.
- GPT lineage: GPT, GPT-2, GPT-3, Llama.
- Advantages: single stack, dense training signal, subsumes understanding tasks through prompting, emergent in-context learning.

### The Honest Trade-offs
- Encoder-only is still more efficient for fixed-vector embeddings, retrieval, and classification.
- Encoder-decoder remains strong where input and output are clearly distinct (machine translation).
- Decoder-only excels at open-ended generation and many-task prompting; the price is that every task must be phrased as text completion.
- The choice depends on the problem, not on decoder-only being universally superior.

## Training: Match the Target Token
- Placed right before the exercise. The same forward pass as `parallel-forward` (GPT-2 medium on "The capital of France is"); the target is the real next token, " Paris".
- Left panel: the model's softmax over the vocabulary for the last row: " Paris" 0.174, " the" 0.054, " Lyon" 0.042, " not" 0.031, " a" 0.026 (the other 50,252 tokens share the remaining 0.673). Right panel: the target distribution $y$, one-hot on " Paris".
- Cross-entropy between target and prediction: $\mathcal L = -\sum_w y_w \log p_w$. With a one-hot target only the target term survives, so $\mathcal L = -\log p(\text{Paris}) = -\ln 0.174 \approx 1.75$ nats. This is the same cross-entropy from Module 1's information theory.
- The gradient of the loss with respect to the logits is $\mathbf p - \mathbf y$: it raises the target's logit and lowers every other logit in proportion to its probability. Backpropagation carries this into every weight, embedding rows included.
- In training every row of the $T \times V$ logits has a target (the input shifted by one position), so one forward pass gives $T$ loss terms, which are averaged. This is why rows 1 to 4 in the `parallel-forward` animation matter even though inference ignores them. Module 5 covers pretraining at scale.
