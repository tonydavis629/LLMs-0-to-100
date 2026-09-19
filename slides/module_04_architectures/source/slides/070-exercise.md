:::divider id="divider-exercise" title="Exercise" sub="Assemble GPT-2 and Generate Text"
:::

---

<!-- .slide: id="exercise-run" -->

## Running the Exercise

Open `module_04_architectures/exercise.py`, the only file you edit, and fill in the `NotImplementedError` lines. Everything already written for you lives in `src/`. Run after each step. <!-- .element: class="text-lg" -->

```bash
# Run every step; each is tagged CORRECT, INCORRECT, or INCOMPLETE
cd exercises
uv run python module_04_architectures/src/main.py

# Run a single step (1-6)
uv run python module_04_architectures/src/main.py --step 3
```

Step 4 loads the real GPT-2 weights into your model and saves a plot of next-token probabilities to `output/`. The first run downloads the weights from HuggingFace. <!-- .element: class="text-lg" style="margin-top: 15px;" -->

---

<!-- .slide: id="exercise-results" -->

## Reading the Results

For each step the runner prints what your code produced (token IDs, shapes, generated text), then runs the step's **tests** from `tests/`. Each test calls your code on small inputs whose correct answer is known. <!-- .element: class="text-lg" -->

- **CORRECT**: every test for the step passed
- **INCORRECT**: your code ran but a test failed; the expected and actual values are printed beneath
- **INCOMPLETE**: the function still raises `NotImplementedError`

When a step relies on other layers, its tests replace them with simple stand-ins, so each step is judged on its own code. Steps 4 and 5 also check your work against Hugging Face's GPT-2. <!-- .element: class="text-lg" style="margin-top: 15px;" -->

---

:::terminal id="exercise-results-example" title="What an INCORRECT Step Looks Like" cmd="uv run python module_04_architectures/src/main.py --step 3" maxw="920px" caption="Here <code>TransformerBlock.forward()</code> dropped both residual connections (<code>x = self.attn(self.ln1(x))</code>). The shapes still match, so nothing crashes. The test's stand-in layers keep the arithmetic readable: 2 &times; 10 = 20, where adding x back each time gives 33."
<span class="header">=== Step 3: TransformerBlock.forward() ===</span> <span class="t-fail">INCORRECT</span>
Input (1, 5, 768) -&gt; output (1, 5, 768)
Parameters: 7,087,872 per block, 4,722,432 of them in the FFN
  <span class="success">CORRECT</span>    keeps the shape: (2, 5, 8) in gives (2, 5, 8) out
  <span class="t-fail">INCORRECT</span>  adds each sub-layer back to its input: attn(x)=2x, ffn(x)=10x turn x=1 into 33
             expected 33
             got      20
             (20 means no residual at all: add x back after each sub-layer)
  <span class="t-fail">INCORRECT</span>  normalizes before each sub-layer: x + attn(ln1(x)), then x + ffn(ln2(x))
             largest difference from the reference is 3.5097
:::

---

<!-- .slide: id="exercise-overview" -->

## Exercise: Assemble GPT-2

Build the full decoder-only transformer from scratch, load pretrained weights, and generate text with different decoding strategies. <!-- .element: class="text-lg" -->

:::columns cols="2" gap="30px"
**Model assembly (steps 1 to 4)**

Implement embeddings, the FFN, a transformer block, and the full stack. The runner then loads the real GPT-2 weights into your model.
+++
**Generation (steps 5 and 6)**

Implement greedy decoding and the temperature scaling for top-k sampling. Observe how decoding strategy shapes the output.
:::

Each function is mostly written, so you only fill in the missing lines. Every step has its own tests in `tests/`. <!-- .element: class="text-lg" style="margin-top: 10px;" -->

---

:::step id="exercise-step1-code" title="Step 1: EmbeddingLayer.forward()"
The layer holds two tables, `self.token_embed` (one row per vocabulary token) and `self.pos_embed` (one row per position). Inside `forward(self, token_ids)`: <!-- .element: class="text-lg" style="margin-bottom: 10px;" -->

```python
        b, t = token_ids.size()
        # Create a tensor [0, 1, 2, ..., t-1] for the positions in this batch.
        position = torch.arange(t, dtype=torch.long, device=token_ids.device)
        # Broadcast so every row in the batch gets the same position indices.
        position = position.unsqueeze(0).expand(b, t)

        # TODO: Look up token embeddings and position embeddings, then add them.
        raise NotImplementedError("TODO: token embedding lookup + positional embedding")
```
+++
**Hint:** there are two embedding tables on `self` (one for tokens, one for positions); call each on its index tensor and add the two results.
+++
**Answer:**

```python
return self.token_embed(token_ids) + self.pos_embed(position)
```
:::

---

:::step id="exercise-step2-code" title="Step 2: FeedForward.forward()"
```python
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the two-layer FFN with GELU nonlinearity.

        GPT-2 was trained with the tanh approximation of GELU, so we pass
        approximate="tanh" to reproduce the pretrained model exactly. The
        default (exact) GELU is a slightly different curve and shifts the
        logits enough to notice.

        Args:
            x: Input tensor, shape (batch_size, seq_len, d_model).

        Returns:
            Output tensor, shape (batch_size, seq_len, d_model).
        """
        # TODO: Apply fc1, then F.gelu(..., approximate="tanh"), then fc2, then dropout.
        raise NotImplementedError("TODO: FFN forward pass")
```
+++
**Hint:** compose the layers in order; the tensor should end at d_model, not d_ff.
+++
**Answer:**

```python
return self.dropout(self.fc2(F.gelu(self.fc1(x), approximate="tanh")))
```
:::

---

:::terminal id="exercise-step2-output" title="Steps 1 and 2: Output" cmd="uv run python module_04_architectures/src/main.py" maxw="920px" caption="The tokenizer turns the prompt into 5 token IDs, and your layer maps each one to a 768-number vector. The third FFN test catches a missing <code>approximate=&quot;tanh&quot;</code>: the exact GELU differs from the tanh curve by up to 0.0005."
<span class="header">=== Step 1: EmbeddingLayer.forward() ===</span> <span class="success">CORRECT</span>
Prompt: "The capital of France is"
Tokens: ['The', ' capital', ' of', ' France', ' is']
Token IDs: [464, 3139, 286, 4881, 318]
Embeddings: (1, 5, 768), one 768-number vector per token
  <span class="success">CORRECT</span>    returns one vector per token: ids (2, 3) give shape (2, 3, d&#95;model=2)
  <span class="success">CORRECT</span>    adds token + position vectors: ids [2,1,2] give [[103,104],[201,202],[303,304]]
  <span class="success">CORRECT</span>    every sequence in the batch uses positions 0, 1, 2 (row 2: ids [0,0,0])

<span class="header">=== Step 2: FeedForward.forward() ===</span> <span class="success">CORRECT</span>
Input (1, 5, 768) -&gt; output (1, 5, 768), widening to d&#95;ff=3072 in between
Parameters: 4,722,432
  <span class="success">CORRECT</span>    keeps the shape: (1, 5, 4) in gives (1, 5, 4) out, even with d&#95;ff=8 inside
  <span class="success">CORRECT</span>    hand-set layers: x=[2, -2] gives fc2(GELU(fc1(x))) = [2.4092, 0.5000]
  <span class="success">CORRECT</span>    uses the tanh GELU that GPT-2 was trained with (approximate="tanh")
  <span class="success">CORRECT</span>    applies dropout: with p=0.5, training and eval outputs differ

<span class="header">=== Step 3: TransformerBlock.forward() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">TODO: transformer block forward pass</span>

<span class="skipped">...</span>
:::

---

:::step id="exercise-step3-code" title="Step 3: TransformerBlock.forward()"
```python
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-norm transformer block with residual connections.

        Args:
            x: Input tensor, shape (batch_size, seq_len, d_model).

        Returns:
            Output tensor, shape (batch_size, seq_len, d_model).
        """
        # TODO: Pre-norm attention with residual, then pre-norm FFN with residual.
        raise NotImplementedError("TODO: transformer block forward pass")
```
+++
**Hint:** use the same pattern twice: normalize, apply the sub-layer, then add the result back to x.
+++
**Answer:**

```python
x = x + self.attn(self.ln1(x))
x = x + self.ffn(self.ln2(x))
return x
```
:::

---

:::step id="exercise-step4-code" title="Step 4: GPT2Model.forward()"
```python
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Run a full forward pass from token IDs to logits.

        Args:
            token_ids: Integer token IDs, shape (batch_size, seq_len).

        Returns:
            Logits over the vocabulary, shape (batch_size, seq_len, vocab_size).
        """
        # TODO: embed -> run every block -> final layer norm -> LM head.
        raise NotImplementedError("TODO: full GPT-2 forward pass")
```
+++
**Hint:** embed first, then loop over `self.blocks`, then `self.ln_f`, then `self.lm_head`.
+++
**Answer:**

```python
x = self.embed(token_ids)
for block in self.blocks:
    x = block(x)
return self.lm_head(self.ln_f(x))
```
:::

---

<!-- .slide: id="exercise-weights" -->

## Loading Pretrained Weights (Provided)

In Step 4 the runner calls `load_gpt2_weights(model)` from `src/pretrained.py`. It fetches the official GPT-2 small checkpoint from HuggingFace and copies each tensor into the matching layer of your model. A name-mapping table bridges the different naming conventions:

| Custom name | Pretrained name |
|-------------|-----------------|
| `embed.token_embed.weight` | `transformer.wte.weight` |
| `embed.pos_embed.weight` | `transformer.wpe.weight` |
| `blocks.0.attn.c_attn.weight` | `transformer.h.0.attn.c_attn.weight` |
| `ln_f.weight` | `transformer.ln_f.weight` |
| `lm_head.weight` | `lm_head.weight` |

The shape of every tensor is checked before copying, and all 149 parameter tensors load. Step 4's last test then runs the prompt through your model and through Hugging Face's own GPT-2 and compares the logits.

---

:::terminal id="exercise-step4-output" title="Steps 3 and 4: Output" cmd="uv run python module_04_architectures/src/main.py" maxw="920px" caption="Your GPT-2 matches Hugging Face's logits to within 0.001. It has 163M parameters rather than the usual 124M because <code>lm_head</code> keeps its own copy of the 38.6M-parameter token embedding, which GPT-2 shares (see the extra credit)."
<span class="header">=== Step 1: EmbeddingLayer.forward() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 2: FeedForward.forward() ===</span> <span class="success">CORRECT</span>
<span class="skipped">...</span>

<span class="header">=== Step 3: TransformerBlock.forward() ===</span> <span class="success">CORRECT</span>
Input (1, 5, 768) -&gt; output (1, 5, 768)
Parameters: 7,087,872 per block, 4,722,432 of them in the FFN
  <span class="success">CORRECT</span>    keeps the shape: (2, 5, 8) in gives (2, 5, 8) out
  <span class="success">CORRECT</span>    adds each sub-layer back to its input: attn(x)=2x, ffn(x)=10x turn x=1 into 33
  <span class="success">CORRECT</span>    normalizes before each sub-layer: x + attn(ln1(x)), then x + ffn(ln2(x))

<span class="header">=== Step 4: GPT2Model.forward() ===</span> <span class="success">CORRECT</span>
Loading GPT-2 small weights from HuggingFace (downloaded on the first run) ...
Loaded 149 parameter tensors from pretrained GPT-2.
Parameters: 163,037,184 (124,439,808 if lm&#95;head shared the token embedding)
Logits: (1, 5, 50257), one score per vocabulary token at each position
Most likely next tokens: ' the' 8.5%, ' now' 4.8%, ' a' 4.6%, ' France' 3.2%, ' Paris' 3.2%
Saved token probability plot to module&#95;04&#95;architectures/output/token&#95;probs.png
  <span class="success">CORRECT</span>    returns logits over the vocabulary: ids (2, 3) give shape (2, 3, vocab&#95;size=7)
  <span class="success">CORRECT</span>    runs all 3 blocks once each, in order: block calls were [0, 1, 2]
  <span class="success">CORRECT</span>    feeds each block's output to the next, then applies ln&#95;f and lm&#95;head
  <span class="success">CORRECT</span>    real GPT-2: your logits match Hugging Face's GPT2LMHeadModel to within 0.001

<span class="header">=== Step 5: greedy&#95;decode() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">TODO: greedy argmax next token</span>
<span class="skipped">...</span>
:::

---

:::step id="exercise-step5-code" title="Step 5: greedy_decode()"
```python
    with torch.no_grad():
        for _ in range(max_new):
            # Run a forward pass to get logits for every position.
            logits = model(token_ids)
            # The logits for the very last position predict the next token.
            next_logits = logits[:, -1, :]

            # TODO: Pick the token with the highest logit (argmax).
            next_token = None
            if next_token is None:
                raise NotImplementedError("TODO: greedy argmax next token")

            # Append the new token to the sequence.
            token_ids = torch.cat([token_ids, next_token], dim=1)

    return tokenizer.decode(token_ids[0])
```
+++
**Hint:** use `torch.argmax(next_logits, dim=-1, keepdim=True)`.
+++
**Answer:**

```python
next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
```
:::

---

:::step id="exercise-step6-code" title="Step 6: sample_with_temperature_topk()"
```python
    with torch.no_grad():
        for _ in range(max_new):
            logits = model(token_ids)
            next_logits = logits[:, -1, :]

            # TODO: Scale logits by temperature before top-k filtering.
            next_logits = None
            if next_logits is None:
                raise NotImplementedError("TODO: temperature scaling")

            # Use the provided helper to filter to top-k and sample one token.
            next_token = _sample_topk_token(next_logits, top_k)
            token_ids = torch.cat([token_ids, next_token], dim=1)

    return tokenizer.decode(token_ids[0])
```
+++
**Hint:** temperature is applied before filtering; divide the logits by `temperature`.
+++
**Answer:**

```python
next_logits = next_logits / temperature
```
:::

---

:::terminal id="exercise-step6-output" title="Steps 5 and 6: Generation Output" cmd="uv run python module_04_architectures/src/main.py" maxw="920px" caption="Greedy decoding takes the top token every time, and GPT-2 small loops back to &quot;the capital&quot;. Both samples use the same random seed, so they start alike. At T=1.4 the flatter distribution drifts off and emits the end-of-text token."
<span class="header">=== Step 1: EmbeddingLayer.forward() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 2: FeedForward.forward() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 3: TransformerBlock.forward() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 4: GPT2Model.forward() ===</span> <span class="success">CORRECT</span>
<span class="skipped">...</span>

<span class="header">=== Step 5: greedy&#95;decode() ===</span> <span class="success">CORRECT</span>
Prompt:  "The capital of France is"
Greedy:  "The capital of France is the capital of the French Republic, and the capital"
  <span class="success">CORRECT</span>    follows the argmax: a counting model turns "1" into "1 2 3 4 5"
  <span class="success">CORRECT</span>    picks the largest of close logits [1.0, 1.2, 0.9, 1.1]: token 1, every time
  <span class="success">CORRECT</span>    real GPT-2: your loop matches Hugging Face's generate(do&#95;sample=False)

<span class="header">=== Step 6: sample&#95;with&#95;temperature&#95;topk() ===</span> <span class="success">CORRECT</span>
Prompt:  "The capital of France is"
T=0.8, top&#95;k=40:  "The capital of France is a place that has never had much of a French"
T=1.4, top&#95;k=40:  "The capital of France is a place that's pretty close...&lt;|endoftext|&gt;The New"
  <span class="success">CORRECT</span>    divides the logits by the temperature: [2, 4, 6] at T=2 become [1, 2, 3]
  <span class="success">CORRECT</span>    T=0.5 sharpens: logits [0, 1] pick the favorite ~88% of the time (73% at T=1)
  <span class="success">CORRECT</span>    T=2 flattens: logits [0, 1] pick the favorite ~62% of the time (73% at T=1)
:::

---

<!-- .slide: id="exercise-extra-credit" -->

## Extra Credit

- Implement **top-p (nucleus) sampling**: keep the smallest set of tokens whose cumulative probability exceeds $p$, then sample.
- **Tie the embedding and output weights**: set `self.lm_head.weight = self.embed.token_embed.weight` and verify the parameter count drops by about 38M (the size of the embedding matrix).
- **Swap the causal mask for a bidirectional one** in `CausalSelfAttention` and observe how open-ended generation breaks.
- Implement a **tiny BPE training loop** on a short string (a handful of merges) to see tokenization from the inside. <!-- .element: class="text-lg" style="margin-top: 15px;" -->
