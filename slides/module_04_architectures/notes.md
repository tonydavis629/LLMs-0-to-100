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
- **Manim animation (`sampling-demo`):** starts from a next-token distribution over a small vocabulary, then shows temperature sharpening ($T<1$) and flattening ($T>1$), top-k keeping the three most likely tokens, and top-p keeping the nucleus whose mass first exceeds $p$. The point is that each method reshapes the same distribution differently.

### Beam Search
- Keep $b$ best partial sequences at each step.
- The slide draws this as a left-to-right tree with $b = 2$: the two surviving beams are solid orange paths, and the pruned (lower-scoring) branches are dashed.
- Best for constrained tasks (translation) where a single correct answer exists.
- Poor for open-ended generation, where the most likely sequence is usually vacuous or repetitive.

## Putting It All Together

A single prompt is followed through a decoder-only transformer. The section opens with an architecture-only slide (`decoder-only-arch`) showing the full stack bottom-to-top: input tokens, token + positional embedding, an $N\times$ block of masked multi-head self-attention and a feed-forward network with Add & Norm around each, then a final LayerNorm, a linear head, and softmax to next-token probabilities. The recurring pipeline locator has six boxes: **word &rarr; vector**, add position, multi-head attention, feed-forward network, repeat blocks, sampling. Tokenization is collapsed into the first box because it has its own earlier section; the first box represents the entire word-to-vector embedding stage. The key concept slides (embedding, feed-forward, normalization, residual stream) are each paired with a dedicated step-through Manim animation.

### Step 1: Words to Vectors
- Token ID $t_i$ indexes row $W_E[t_i]$ of the embedding matrix: $\mathbf{x}_i = W_E[t_i]$.
- The slide shows the full handoff from words to subword tokens, token IDs, and embedding rows. The transformer receives vectors, not raw text.
- **Manim animation (`embedding-lookup`):** "France" maps to id 4881, the id selects one row of the embedding matrix, and that row is pulled out as the token's vector.

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

### Step 6: Sampling
- The slide renames the former "logits to probabilities" step. Final layer norm, then $W_{\text{head}}$ projects to vocabulary-sized logits; softmax gives the next-token distribution.
- Weight tying: $W_{\text{head}} = W_E^{\top}$ saves parameters and ties input/output semantics.
- The decoding strategy from the earlier sampling section (greedy, temperature, top-k, top-p) then chooses the next token, which is appended and fed back in.

### The Residual Stream
- Placed after Step 6 so the full forward pass is in view before introducing the interpretability lens.
- Each sub-layer reads from the running vector, computes its contribution, and writes it back: $\mathbf{x} = \mathbf{x} + \text{sub-layer}(\text{norm}(\mathbf{x}))$.
- Because each block only adds to this running vector, the original embedding is never overwritten and gradients flow back through the addition without decaying.
- **Manim animation (`residual-stream`):** the residual stream is drawn as a horizontal "highway" from the embedding to the logits. Each attention and MLP sub-layer branches off via a vertical read arrow, computes an update, and writes it back at an addition node on the stream — the standard mechanistic-interpretability picture (cf. Elhage et al., "A Mathematical Framework for Transformer Circuits," 2021).

### Side Quest: Induction Heads
- **Induction heads** (Olsson et al., 2022) are attention heads implementing a copy-and-continue pattern: given `[A] [B] ... [A]`, predict `[B]`. Source: <https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html>.
- They are linked to the emergence of in-context and few-shot learning at scale: a head detects an earlier occurrence of the current token and predicts whatever followed it last time, with no weight updates.

### Parameter Counting (GPT-2 small)
- $d = 768$, $N = 12$, $V = 50257$.
- Embeddings: $V \cdot d \approx 38.6$M.
- Attention per layer: $4d^2$ (Q/K/V projection + output projection) = ~2.36M. Total: $N \cdot 4d^2 \approx 28.3$M.
- FFN per layer: $8d^2$ (two linear layers, $d \to 4d \to d$) = ~4.71M. Total: $N \cdot 8d^2 \approx 56.6$M.
- Total: ~124M parameters.

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
