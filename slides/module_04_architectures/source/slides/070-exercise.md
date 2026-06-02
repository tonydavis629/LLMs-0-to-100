:::divider id="divider-exercise" title="Exercise" sub="Assemble GPT-2 and Generate Text"
:::

---

<!-- .slide: id="exercise-run" -->

## Running the Exercise

Open `module_04_architectures/exercise.py` and fill in the `NotImplementedError` lines. Run after each step &mdash; unfinished functions are skipped automatically. <!-- .element: class="text-lg" -->

```bash
# Run the exercise (skips any not yet implemented)
cd exercises
uv run python module_04_architectures/src/main.py
```

The exercise wires attention into a complete decoder-only model, loads real GPT-2 weights, and generates text. Check the `output/` directory for plots after each run. <!-- .element: class="text-lg" style="margin-top: 15px;" -->

---

<!-- .slide: id="exercise-overview" -->

## Exercise: Assemble GPT-2

Build the full decoder-only transformer from scratch, load pretrained weights, and generate text with different decoding strategies. <!-- .element: class="text-lg" -->

:::columns cols="2" gap="30px"
**Model assembly (steps 1&ndash;5)**

Implement embeddings, the FFN, a transformer block, the full stack, and the weight-loading bridge to HuggingFace.
+++
**Generation (steps 6&ndash;7)**

Implement greedy decoding, temperature scaling, and top-k sampling. Observe how decoding strategy shapes the output.
:::

---

:::step id="exercise-step1-code" title="Step 1: EmbeddingLayer.forward()"
```python
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look up token and positional embeddings and add them."""
        b, t = token_ids.size()
        position = torch.arange(t, dtype=torch.long, device=token_ids.device)
        position = position.unsqueeze(0).expand(b, t)

        # TODO: Look up token embeddings and position embeddings, then add them.
        raise NotImplementedError("TODO: token embedding lookup + positional embedding")
```
+++
**Hint:** use `self.token_embed(token_ids) + self.pos_embed(position)`.
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
        """Apply the two-layer FFN with GELU nonlinearity."""
        # TODO: Apply fc1, then F.gelu(...), then fc2, then dropout.
        raise NotImplementedError("TODO: FFN forward pass")
```
+++
**Hint:** use `self.fc2(self.dropout(F.gelu(self.fc1(x))))`.
+++
**Answer:**

```python
return self.fc2(self.dropout(F.gelu(self.fc1(x))))
```
:::

---

:::step id="exercise-step3-code" title="Step 3: TransformerBlock.forward()"
```python
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-norm transformer block with residual connections."""
        # TODO: Pre-norm attention with residual, then pre-norm FFN with residual.
        raise NotImplementedError("TODO: transformer block forward pass")
```
+++
**Hint:** `x = x + self.attn(self.ln1(x))` then `x = x + self.ffn(self.ln2(x))`.
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
        """Run a full forward pass from token IDs to logits."""
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

:::terminal id="exercise-step4-output" title="Steps 1&ndash;4: Model Forward Pass" cmd="uv run python module_04_architectures/src/main.py" caption="The model produces logits of shape (batch, seq, vocab_size). Total parameters: ~124M, matching GPT-2 small."
<span class="header">============================================================
STEP 1: Embedding Layer
============================================================</span>
<span class="success">Output shape: torch.Size([1, 3, 768])</span>
<span class="success">Expected: (1, 3, 768)</span>

<span class="header">============================================================
STEP 2: Feed-Forward Network
============================================================</span>
<span class="success">Output shape: torch.Size([1, 3, 768])</span>
<span class="success">Expected: (1, 3, 768)</span>

<span class="header">============================================================
STEP 3: Transformer Block
============================================================</span>
<span class="success">Output shape: torch.Size([1, 3, 768])</span>
<span class="success">Expected: (1, 3, 768)</span>

<span class="header">============================================================
STEP 4: Full GPT2Model Forward Pass
============================================================</span>
<span class="success">Output shape: torch.Size([1, 3, 50257])</span>
<span class="success">Expected: (1, 3, 50257)</span>
<span class="success">Total parameters: 124,439,808</span>
:::

---

<!-- .slide: id="exercise-step5" -->

## Step 5: Load Pretrained Weights (Provided)

`load_gpt2_weights(model)` downloads the official GPT-2 small checkpoint from HuggingFace and copies each tensor into the matching layer of your model. A name-mapping table bridges the different naming conventions:

| Custom name | Pretrained name |
|-------------|-----------------|
| `embed.token_embed.weight` | `transformer.wte.weight` |
| `embed.pos_embed.weight` | `transformer.wpe.weight` |
| `blocks.0.attn.c_attn.weight` | `transformer.h.0.attn.c_attn.weight` |
| `ln_f.weight` | `transformer.ln_f.weight` |
| `lm_head.weight` | `lm_head.weight` |

The shape of every tensor is verified before copying. If your model architecture is correct, all 148 parameter tensors load successfully.

---

:::step id="exercise-step6-code" title="Step 6: greedy_decode()"
```python
    def greedy_decode(model, tokenizer, prompt, max_new=10):
        token_ids = tokenizer.encode(prompt, return_tensors="pt")
        for _ in range(max_new):
            logits = model(token_ids)
            next_logits = logits[:, -1, :]

            # TODO: Pick the token with the highest logit (argmax).
            next_token = None
            if next_token is None:
                raise NotImplementedError("TODO: greedy argmax next token")
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

:::step id="exercise-step7-code" title="Step 7: sample_with_temperature_topk()"
```python
    def sample_with_temperature_topk(model, tokenizer, prompt, max_new=10,
                                     temperature=1.0, top_k=50):
        token_ids = tokenizer.encode(prompt, return_tensors="pt")
        for _ in range(max_new):
            logits = model(token_ids)
            next_logits = logits[:, -1, :]

            # TODO: Scale logits by temperature, then apply top-k filtering, then sample.
            raise NotImplementedError("TODO: temperature scaling + top-k sampling")
        return tokenizer.decode(token_ids[0])
```
+++
**Hint:** divide by `temperature`, use `torch.topk` to zero out tokens below the k-th largest logit, then `F.softmax` and `torch.multinomial`.
+++
**Answer:**

```python
next_logits = next_logits / temperature
topk_vals, _ = torch.topk(next_logits, top_k, dim=-1)
threshold = topk_vals[:, -1].unsqueeze(-1)
next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))
probs = F.softmax(next_logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
token_ids = torch.cat([token_ids, next_token], dim=1)
```
:::

---

:::terminal id="exercise-step7-output" title="Steps 6&ndash;7: Generation Output" cmd="uv run python module_04_architectures/src/main.py" caption="Greedy output is deterministic. Sampling with temperature produces varied completions."
<span class="header">============================================================
STEP 6: Greedy Decoding
============================================================</span>
<span class="success">Prompt:  "The capital of France is"</span>
<span class="success">Output:  "The capital of France is Paris. The city is home to the"</span>

<span class="header">============================================================
STEP 7: Temperature and Top-k Sampling
============================================================</span>
<span class="success">Prompt:  "The capital of France is"</span>
<span class="success">Output (temp=0.8): "The capital of France is Paris, and the French"</span>
<span class="success">Output (temp=1.4): "The capital of France is known as the 'City of"</span>

<span class="header">============================================================
VISUALIZATIONS
============================================================</span>
<span class="success">Saved token probability plot to module_04_architectures/output/token_probs.png</span>
:::

---

<!-- .slide: id="exercise-extra-credit" -->

## Extra Credit

- Implement **top-p (nucleus) sampling**: keep the smallest set of tokens whose cumulative probability exceeds $p$, then sample.
- **Tie the embedding and output weights**: set `self.lm_head.weight = self.embed.token_embed.weight` and verify the parameter count drops by about 38M (the size of the embedding matrix).
- **Swap the causal mask for a bidirectional one** in `CausalSelfAttention` and observe how open-ended generation breaks.
- Implement a **tiny BPE training loop** on a short string (a handful of merges) to see tokenization from the inside. <!-- .element: class="text-lg" style="margin-top: 15px;" -->
