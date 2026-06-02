# Module 4: LLM Architectures — Lecture Notes

Citations, math, and explanations for every claim in the presentation.

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

## Tokenization

### The Unit Problem
- Characters: tiny vocabulary, very long sequences, low information per token.
- Words: meaningful units, but vocabulary explodes and any unseen word is out-of-vocabulary.
- Subwords: frequent words stay whole, rare words split into reusable pieces.

### Byte-Pair Encoding
- BPE starts from characters and iteratively merges the most frequent adjacent pair.
- The learned list of merges defines the vocabulary. Every input can be decomposed into vocabulary tokens.
- GPT-2 uses byte-level BPE: the merge process runs over raw bytes, so every Unicode string is representable.
- GPT-2 vocabulary: ~50k tokens. Modern models use 100k&ndash;200k.

### Tokenization in Practice
- Token counts drive context limits and API cost.
- Non-English text often requires more tokens per sentence (the "token tax").
- Models struggle with letter counting, string reversal, and digit arithmetic because these operations are not natural at the token level.
- Tokenizers and models are trained separately. Glitch tokens (rare tokens with near-zero training frequency) can cause unpredictable behavior.

## Anatomy of a Transformer Block

### The Embedding Layer
- Token ID $t_i$ indexes row $W_E[t_i]$ of the embedding matrix.
- Positional embedding $W_P[i]$ adds position information: $\mathbf{x}_i = W_E[t_i] + W_P[i]$.

### The Residual Stream
- Each sub-layer reads from the stream, computes its contribution, and writes it back: $\mathbf{x} = \mathbf{x} + \text{sub-layer}(\text{norm}(\mathbf{x}))$.
- Residual connections create a gradient highway that makes deep stacks trainable.

### Multi-Head Self-Attention
- The only sub-layer where information moves between positions in the same layer.
- Wrapped in a residual connection and normalization.

### Position-Wise Feed-Forward Network
- Two linear layers with a nonlinearity, applied independently at every position.
- GPT-2 uses GELU; modern models use SwiGLU.
- Typically $d_{\text{ff}} \approx 4 \cdot d_{\text{model}}$.
- Stores much of the model's factual knowledge (Geva et al., 2021).

### Normalization
- LayerNorm: $\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$.
- Post-norm: normalize after sub-layer output (original transformer).
- Pre-norm: normalize before sub-layer input (modern default, more stable).
- RMSNorm: drop mean centering, divide by root-mean-square.

### Output Head and Weight Tying
- Final layer norm, then $W_{\text{head}}$ projects to vocabulary-sized logits.
- Weight tying: $W_{\text{head}} = W_E^T$. Saves parameters and ties input/output semantics.

### Parameter Counting (GPT-2 small)
- $d = 768$, $N = 12$, $V = 50257$.
- Embeddings: $V \cdot d \approx 38.6$M.
- Attention per layer: $4d^2$ (Q/K/V projection + output projection) = ~2.36M. Total: $N \cdot 4d^2 \approx 28.3$M.
- FFN per layer: $8d^2$ (two linear layers, $d \to 4d \to d$) = ~4.71M. Total: $N \cdot 8d^2 \approx 56.6$M.
- Total: ~124M parameters.

## The Three Architectural Families

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
- Mixtral, DeepSeek.

### Sub-Quadratic Alternatives
- Mamba: state-space model with input-dependent gates. Linear in sequence length.
- Linear attention: kernelized softmax avoiding the $n \times n$ matrix.
- RWKV: combines recurrence with linear attention.
- Hybrids like Jamba mix attention and state-space layers.

## Generating Text: Sampling and Decoding

### Greedy Decoding
- Always select $\arg\max_i p_i$.
- Deterministic but produces repetitive, flat text.

### Temperature
- Scale logits by $T$ before softmax: $p_i \propto \exp(z_i / T)$.
- $T < 1$: sharper, more conservative.
- $T > 1$: flatter, more random.

### Top-k and Top-p
- Top-k: keep only the $k$ highest-probability tokens.
- Top-p (nucleus): keep the smallest set whose cumulative probability exceeds $p$.
- Usually combined with temperature scaling.

### Beam Search
- Keep $b$ best partial sequences at each step.
- Best for constrained tasks (translation) where a single correct answer exists.
- Poor for open-ended generation.
