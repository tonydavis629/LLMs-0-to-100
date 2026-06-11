:::divider id="divider-exercise" title="Exercise" sub="Pretraining NanoGPT"
:::

---

<!-- .slide: id="exercise-run" -->

## Running the Exercise

Open `module_05_pretraining/exercise.py` and fill in the `NotImplementedError` lines. The model is provided &mdash; you write the training loop. Run after each step; unfinished steps are skipped automatically. <!-- .element: class="text-lg" -->

```bash
# Train the tiny model on the bundled Shakespeare corpus
cd exercises
uv run python module_05_pretraining/src/main.py

# Single-batch sanity check: the loss should crater toward zero
uv run python module_05_pretraining/src/main.py --overfit
```

The runner prints the model and dataset size, a sample before training, the loss at each checkpoint, the final perplexity and bits per token, a sample after training, and saves a loss-curve image. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="exercise-overview" -->

## Exercise: Pretraining NanoGPT

The architecture is fixed (it is the Module 4 model, scaled down). Everything you implement is the **pretraining loop**. <!-- .element: class="text-lg" -->

:::columns cols="2" gap="30px"
**The loop (steps 1&ndash;8)**

Encode text, split train/validation, build **shifted** batches, compute cross-entropy, step the optimizer under a warmup + cosine schedule, then read the loss as perplexity and bits per token.
+++
**Generation (step 10)**

Sample one token at a time and watch the output go from random characters to text-like Shakespeare.
:::

---

:::step id="exercise-step1" title="Step 1: encode()"
```python
def encode(text, stoi):
    # TODO: Return a 1-D LongTensor with one integer ID per character in `text`.
    raise NotImplementedError("TODO: encode text into a LongTensor of token IDs")
```
+++
**Hint:** look up each character in `stoi`, then wrap the list with `torch.tensor(..., dtype=torch.long)`.
+++
**Answer:**

```python
return torch.tensor([stoi[c] for c in text], dtype=torch.long)
```
:::

---

:::step id="exercise-step2" title="Step 2: train_val_split()"
```python
def train_val_split(data, val_fraction=0.1):
    # TODO: Return (train_data, val_data): the first (1 - val_fraction) of the
    #       tokens for training, and the remaining tail for validation.
    raise NotImplementedError("TODO: split the token stream into train and validation")
```
+++
**Hint:** compute `n_train = int(len(data) * (1 - val_fraction))`, then slice `data[:n_train]` and `data[n_train:]`.
+++
**Answer:**

```python
n_train = int(len(data) * (1.0 - val_fraction))
return data[:n_train], data[n_train:]
```
:::

---

:::terminal id="exercise-output-12" title="Steps 1&ndash;2: Data" cmd="uv run python module_05_pretraining/src/main.py" caption="A 1.1M-character corpus, 65 distinct characters, encoded and split. The model has 818K parameters."
<span class="header">MODEL</span>
TinyGPT: 4 layers, 4 heads, width 128, context 128
<span class="success">Parameters:  818,048</span>

<span class="header">STEP 1: Encode text into token IDs</span>
<span class="success">Encoded 1,115,394 tokens. First 20 IDs: [18, 47, 56, 57, 58, 1, 15, ...]</span>

<span class="header">STEP 2: Train / validation split</span>
<span class="success">Train tokens: 1,003,854   Validation tokens: 111,540</span>
:::

---

:::step id="exercise-step3" title="Step 3: get_batch() &mdash; the shift"
```python
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in ix])

    # TODO: Build y, the targets: the same blocks as x but shifted one step left,
    #       so y[:, t] is the token that should follow x[:, t].
    y = None
    if y is None:
        raise NotImplementedError("TODO: build the shifted target batch y")
    return x, y
```
+++
**Hint:** mirror the line that builds `x`, but start each slice one index later: `i + 1 ... i + 1 + block_size`.
+++
**Answer:**

```python
y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
```
:::

---

:::step id="exercise-step4" title="Step 4: compute_loss()"
```python
def compute_loss(logits, targets):
    B, T, V = logits.shape
    # TODO: Return the average cross-entropy between logits and targets.
    raise NotImplementedError("TODO: cross-entropy loss from logits and targets")
```
+++
**Hint:** flatten `logits` to `(B*T, V)` and `targets` to `(B*T,)`, then call `F.cross_entropy`.
+++
**Answer:**

```python
return F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
```
:::

---

:::step id="exercise-step5" title="Step 5: train_step()"
```python
    logits = model(x)
    loss = compute_loss(logits, y)

    # TODO: Clear last step's gradients, then backpropagate this step's loss.
    raise NotImplementedError("TODO: zero the gradients, then loss.backward()")

    # Provided: clip the global gradient norm for stability, then take the step.
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return loss.item()
```
+++
**Hint:** two calls: `optimizer.zero_grad(set_to_none=True)` and `loss.backward()`. The clip and step are already provided.
+++
**Answer:**

```python
optimizer.zero_grad(set_to_none=True)
loss.backward()
```
:::

---

:::step id="exercise-step6" title="Step 6: lr_at_step() &mdash; cosine decay"
```python
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    # TODO: Set coeff to follow a cosine from 1 (start of decay) down to 0 (end).
    coeff = None
    if coeff is None:
        raise NotImplementedError("TODO: cosine-decay coefficient")
    return min_lr + coeff * (max_lr - min_lr)
```
+++
**Hint:** `coeff = 0.5 * (1 + cos(pi * decay_ratio))`; use `math.cos` and `math.pi`.
+++
**Answer:**

```python
coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
```
:::

---

:::step id="exercise-step7" title="Step 7: estimate_loss()"
```python
    total = 0.0
    for _ in range(n_batches):
        x, y = get_batch(data, block_size, batch_size, generator)
        logits = model(x)
        # TODO: Add this batch's loss (as a Python float) to `total`.
        raise NotImplementedError("TODO: accumulate the batch loss into total")
    return total / n_batches
```
+++
**Hint:** `compute_loss(logits, y)` returns a tensor; use `.item()` to get a float.
+++
**Answer:**

```python
total += compute_loss(logits, y).item()
```
:::

---

:::step id="exercise-step8" title="Step 8: perplexity_and_bits()"
```python
def perplexity_and_bits(loss):
    # TODO: Return (perplexity, bits_per_token) from the average loss (in nats).
    raise NotImplementedError("TODO: perplexity and bits per token")
```
+++
**Hint:** `perplexity = exp(loss)`; `bits_per_token = loss / ln(2)`. Use `math.exp` and `math.log`.
+++
**Answer:**

```python
return math.exp(loss), loss / math.log(2)
```
:::

---

:::step id="exercise-step10" title="Step 10: generate()"
```python
        logits = logits[:, -1, :] / temperature

        # TODO: Turn these logits into probabilities and sample ONE token id.
        next_id = None
        if next_id is None:
            raise NotImplementedError("TODO: sample the next token id from the logits")

        idx = torch.cat([idx, next_id], dim=1)
```
+++
**Hint:** `probs = F.softmax(logits, dim=-1)`, then `torch.multinomial(probs, num_samples=1, generator=generator)`.
+++
**Answer:**

```python
probs = F.softmax(logits, dim=-1)
next_id = torch.multinomial(probs, num_samples=1, generator=generator)
```
:::

---

:::terminal id="exercise-output-train" title="Full Run: Loss Falling" cmd="uv run python module_05_pretraining/src/main.py" caption="Training loss falls from 4.18 to ~1.44; validation tracks it. Final perplexity 5.14, bits per token 2.36."
<span class="header">TRAINING</span>
  step         lr     train       val
     0   3.00e-05    4.1816    4.1803
   250   2.96e-03    2.2870    2.3055
   500   2.72e-03    1.8689    1.9827
  1000   1.76e-03    1.5905    1.7748
  1500   7.36e-04    1.4845    1.6649
  2000   3.00e-04    1.4379    1.6373

<span class="header">STEP 8: Perplexity and bits per token</span>
<span class="success">Final validation loss: 1.6373 nats</span>
<span class="success">Perplexity (exp loss): 5.14   (effective next-char choices)</span>
<span class="success">Bits per token (loss / ln 2): 2.3621</span>
:::

---

:::terminal id="exercise-output-overfit" title="Sanity Check: Overfit One Batch" cmd="uv run python module_05_pretraining/src/main.py --overfit" caption="On a single fixed batch the loss craters from 4.18 toward zero: the loop is wired correctly."
<span class="header">OVERFIT ONE BATCH (sanity check)</span>
Training repeatedly on ONE batch of shape (8, 128) for 300 steps:
  step      loss
     0    4.1782
    50    2.4023
   100    1.1647
   150    0.3843
   200    0.2334
   300    0.0704
:::

---

<!-- .slide: id="exercise-extra-credit" -->

## Extra Credit

- **Memorization vs generalization.** Make a tiny dataset with one phrase repeated many times, train, then check whether the model reproduces that exact phrase verbatim.
- **Prose vs code.** Swap the corpus for a file of source code and compare the samples &mdash; indentation, brackets, identifiers.
- **Gradient clipping.** The runner clips inside `train_step`. Raise the peak learning rate with clipping on vs off and watch for loss spikes.
- **Gradient accumulation.** Accumulate gradients over several micro-batches before each optimizer step, and compare the effective batch size to the per-step one. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

