:::divider id="divider-exercise" title="Exercise" sub="Pretraining NanoGPT"
:::

---

<!-- .slide: id="exercise-run" -->

## Running the Exercise

Open `module_05_pretraining/exercise.py` and fill in the `NotImplementedError` lines. The model is provided; you write the training loop. Run after each step. Step 7 trains for 2,000 steps, which takes several minutes on a laptop CPU, and saves a loss-curve image to `output/`. <!-- .element: class="text-lg" -->

```bash
# Run every step; each is tagged CORRECT, INCORRECT, or INCOMPLETE
cd exercises
uv run python module_05_pretraining/src/main.py

# Run a single step (1, 2, 3, 4, 5, 7, 8, or 10)
uv run python module_05_pretraining/src/main.py --step 4

# Single-batch sanity check: the loss should crater toward zero
uv run python module_05_pretraining/src/main.py --overfit
```

---

<!-- .slide: id="exercise-results" -->

## Reading the Results

For each step the runner prints what your code produced, then runs the step's **tests** from `tests/`. Each test calls your function on a tiny tensor whose correct answer is known. <!-- .element: class="text-lg" -->

- **CORRECT**: every test for the step passed
- **INCORRECT**: your code ran but a test failed; the expected and actual values are printed beneath
- **INCOMPLETE**: the function still raises `NotImplementedError`, or it needs an earlier step you have not finished

The tag sits on the step's header line, with the output and the individual test results after it. A falling loss is not proof that a step is right, so check the tags. <!-- .element: class="text-lg" style="margin-top: 15px;" -->

---

:::terminal id="exercise-results-example" title="What an INCORRECT Step Looks Like" cmd="uv run python module_05_pretraining/src/main.py --step 3" maxw="920px" caption="Here <code>get_batch()</code> built <code>y</code> with the same slice as <code>x</code>, so every target equals its input. The shape test still passes. The shift test prints <code>x[0]</code>, the <code>y[0]</code> it got, and the <code>y[0]</code> it expected. Training on these batches would still drive the loss down, because the model only has to copy its input."
<span class="skipped">...</span>
<span class="header">=== Step 3: get_batch() ===</span> <span class="t-fail">INCORRECT</span>
  One batch: x has shape (32, 128), y has shape (32, 128)
  x[0][:24] = ' guess who caused your f'
  y[0][:24] = ' guess who caused your f'
  <span class="success">CORRECT</span>    x and y both have shape (batch_size, block_size): (3, 4) here
  <span class="t-fail">INCORRECT</span>  on data = 0, 1, ..., 99 every target is its input plus one (y = x + 1)
             x[0] = [44, 45, 46, 47]
             got y[0] = [44, 45, 46, 47], expected [45, 46, 47, 48]
  <span class="t-fail">INCORRECT</span>  shortest stream: 0..4 with block_size 4 gives x = [0,1,2,3], y = [1,2,3,4]
             got x = [[0, 1, 2, 3]], y = [[0, 1, 2, 3]]
:::

---

<!-- .slide: id="exercise-overview" -->

## Exercise: Pretraining NanoGPT

The architecture is fixed (the Module 4 model, scaled down). You implement the **pretraining loop**. <!-- .element: class="text-lg" -->

:::columns cols="2" gap="30px"
**The loop (steps 1&ndash;8)**

- Encode text, split train/validation
- Build **shifted** batches
- Cross-entropy loss, optimizer steps (schedule provided)
- Read the loss as perplexity and bits per token
+++
**Generation (step 10)**

- Sample one token at a time
- Watch output go from random characters to text-like Shakespeare
:::

Each blank is one line or a short block, and every step has its own tests in `tests/`. <!-- .element: class="text-lg" style="margin-top: 10px;" -->

---

:::step id="exercise-step1" title="Step 1: encode()"
```python
def encode(text: str, stoi: dict[str, int]) -> torch.Tensor:
    """Turn a string into a 1-D LongTensor of token IDs.

    Args:
        text: The raw text.
        stoi: A "string-to-index" map from each character to its integer ID.

    Returns:
        A 1-D tensor of dtype torch.long with one ID per character.
    """
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
def train_val_split(data: torch.Tensor,
                    val_fraction: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a 1-D token stream into a training prefix and a validation suffix.

    Args:
        data: The full 1-D tensor of token IDs.
        val_fraction: Fraction of tokens to hold out for validation.

    Returns:
        (train_data, val_data) as two 1-D tensors.
    """
    # TODO: Return (train_data, val_data): the first (1 - val_fraction) of the
    #       tokens for training, and the remaining tail for validation.
    raise NotImplementedError("TODO: split the token stream into train and validation")
```
+++
**Hint:** the split index is the training fraction of `len(data)`, rounded down with `int()`; slice up to it for train and from it onward for val.
+++
**Answer:**

```python
n_train = int(len(data) * (1.0 - val_fraction))
return data[:n_train], data[n_train:]
```
:::

---

:::terminal id="exercise-output-12" title="Steps 1&ndash;2: Data" cmd="uv run python module_05_pretraining/src/main.py" maxw="920px" caption="A 1.1M-character corpus with 65 distinct characters, encoded and split 90/10. The model has 818K parameters. Every later step is tagged INCOMPLETE with the TODO it is waiting on."
Corpus:      tinyshakespeare.txt  (1,115,394 characters)
Vocabulary:  65 unique characters
TinyGPT:     4 layers, 4 heads, width 128, context 128
Parameters:  818,048

<span class="header">=== Step 1: encode() ===</span> <span class="success">CORRECT</span>
  Encoded 1,115,394 tokens. First 20 IDs: [18, 47, 56, 57, 58, 1, 15, 47, 58, 47, 64, 43, 52, 10, 0, 14, 43, 44, 53, 56]
  <span class="success">CORRECT</span>    looks up every character: 'abca' with a=0, b=1, c=2 gives [0, 1, 2, 0]
  <span class="success">CORRECT</span>    returns a 1-D torch.long tensor (the dtype nn.Embedding needs)
  <span class="success">CORRECT</span>    round trip: decoding the IDs of the first 1,000 characters gives the text back

<span class="header">=== Step 2: train_val_split() ===</span> <span class="success">CORRECT</span>
  Train tokens: 1,003,854   Validation tokens: 111,540
  <span class="success">CORRECT</span>    0..9 with val_fraction=0.2 gives train [0..7] and validation [8, 9]
  <span class="success">CORRECT</span>    the default val_fraction=0.1 splits 100 tokens into 90 and 10
  <span class="success">CORRECT</span>    train + validation is the original stream: nothing lost, repeated, or shuffled

<span class="header">=== Step 3: get_batch() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">TODO: build the shifted target batch y</span>

<span class="header">=== Step 4: compute_loss() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">TODO: cross-entropy loss from logits and targets</span>
<span class="skipped">...</span>
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
def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Average cross-entropy between predicted logits and the true next tokens.

    Args:
        logits: Model outputs, shape (batch, time, vocab_size).
        targets: True next-token IDs, shape (batch, time).

    Returns:
        A scalar loss tensor (negative log-likelihood averaged over all tokens).
    """
    batch_size, seq_len, vocab_size = logits.shape
    # TODO: Return the average cross-entropy between logits and targets.
    raise NotImplementedError("TODO: cross-entropy loss from logits and targets")
```
+++
**Hint:** use `.view` to flatten `logits` to `(batch_size * seq_len, vocab_size)` and `targets` to `(batch_size * seq_len,)`, then `F.cross_entropy`.
+++
**Answer:**

```python
return F.cross_entropy(logits.view(batch_size * seq_len, vocab_size),
                       targets.view(batch_size * seq_len))
```
:::

---

:::step id="exercise-step5" title="Step 5: train_step()"
```python
    logits = model(x)
    loss = compute_loss(logits, y)

    # TODO: Clear last step's gradients, then backpropagate this step's loss.
    raise NotImplementedError("TODO: clear old gradients and backpropagate the loss")

    # Provided: clip the global gradient norm for stability, then take the step.
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return loss.item()
```
+++
**Hint:** gradients accumulate across steps unless cleared &mdash; the optimizer has a method to reset them, and the loss tensor has a method that backpropagates.
+++
**Answer:**

```python
optimizer.zero_grad(set_to_none=True)
loss.backward()
```
:::

---

:::terminal id="exercise-output-35" title="Steps 3&ndash;5: Batches, Loss, One Step" cmd="uv run python module_05_pretraining/src/main.py" maxw="920px" caption="<code>y</code> is <code>x</code> shifted by one character. Untrained, the loss is near ln 65: a uniform guess."
<span class="header">=== Step 1: encode() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 2: train_val_split() ===</span> <span class="success">CORRECT</span>
<span class="skipped">  ...</span>

<span class="header">=== Step 3: get_batch() ===</span> <span class="success">CORRECT</span>
  One batch: x has shape (32, 128), y has shape (32, 128)
  x[0][:24] = ' guess who caused your f'
  y[0][:24] = 'guess who caused your fa'
  <span class="success">CORRECT</span>    x and y both have shape (batch_size, block_size): (3, 4) here
  <span class="success">CORRECT</span>    on data = 0, 1, ..., 99 every target is its input plus one (y = x + 1)
  <span class="success">CORRECT</span>    shortest stream: 0..4 with block_size 4 gives x = [0,1,2,3], y = [1,2,3,4]

<span class="header">=== Step 4: compute_loss() ===</span> <span class="success">CORRECT</span>
  Untrained model on 8 x 128 characters: loss 4.1857 nats
  A uniform guess over 65 characters costs ln 65 = 4.1744 nats
  <span class="success">CORRECT</span>    equal logits over 3 tokens cost ln 3 = 1.0986 nats
  <span class="success">CORRECT</span>    averages over positions: surprises ln 3 and ln 2 give 0.8959
  <span class="success">CORRECT</span>    matches -log_softmax at the true token, averaged, on a random (2, 4, 4) batch
  <span class="success">CORRECT</span>    returns a 0-dim tensor that autograd can backpropagate through

<span class="header">=== Step 5: train_step() ===</span> <span class="success">CORRECT</span>
  Five train_step() calls on one batch of 8 x 128 characters:
  loss 4.1860 -&gt; 3.8484 -&gt; 3.5361 -&gt; 3.3830 -&gt; 3.2890
  <span class="success">CORRECT</span>    returns the batch loss as a float: ln 5 = 1.6094 for a table of zeros
  <span class="success">CORRECT</span>    one SGD step (lr 0.5) moves the weights exactly as backward() + step() should
  <span class="success">CORRECT</span>    clears old gradients first: a leftover .grad of 100 does not leak into the step

<span class="header">=== Step 7: estimate_loss() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">TODO: accumulate the batch loss into total</span>
:::

---

:::step id="exercise-step7" title="Step 7: estimate_loss()"
```python
@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    n_batches: int,
    generator: torch.Generator | None = None,
) -> float:
    """Average the loss over several random batches (a less noisy estimate).

    Args:
        model: The model to evaluate.
        data: The token stream to sample evaluation batches from.
        block_size: Context length.
        batch_size: Examples per batch.
        n_batches: How many batches to average over.
        generator: Optional RNG for reproducible sampling.

    Returns:
        The mean loss across the batches, as a float.
    """
    model.eval()
    total = 0.0
    for _ in range(n_batches):
        x, y = get_batch(data, block_size, batch_size, generator)
        logits = model(x)
        # TODO: Add this batch's loss (as a Python float) to `total`.
        raise NotImplementedError("TODO: accumulate the batch loss into total")
    model.train()
    return total / n_batches
```
+++
**Hint:** reuse `compute_loss` on this batch; it returns a tensor, and `.item()` turns that into a float.
+++
**Answer:**

```python
total += compute_loss(logits, y).item()
```
:::

---

:::terminal id="exercise-output-train" title="Step 7: The Pretraining Run" cmd="uv run python module_05_pretraining/src/main.py" maxw="920px" caption="Loss falls from 4.18, a uniform guess, to 1.44 on train and 1.64 on validation. The last test checks the run itself."
<span class="header">=== Step 1: encode() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 2: train_val_split() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 3: get_batch() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 4: compute_loss() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 5: train_step() ===</span> <span class="success">CORRECT</span>
<span class="skipped">  ...</span>

<span class="header">=== Step 7: estimate_loss() ===</span> <span class="success">CORRECT</span>
Pretraining for 2,000 steps (warmup + cosine learning rate from src/schedules.py):
  step         lr     train       val
     0   3.00e-05    4.1816    4.1803
   250   2.96e-03    2.2870    2.3055
   500   2.72e-03    1.8689    1.9827
   750   2.29e-03    1.6868    1.8527
  1000   1.76e-03    1.5905    1.7748
  1250   1.21e-03    1.5325    1.7299
  1500   7.36e-04    1.4845    1.6649
  1750   4.14e-04    1.4398    1.6254
  2000   3.00e-04    1.4379    1.6373
Saved loss curve to module_05_pretraining/output/loss_curve.png
  <span class="success">CORRECT</span>    a uniform model scores ln 5 = 1.6094 on every batch, so the average is ln 5
  <span class="success">CORRECT</span>    matches the mean of compute_loss(logits, y) over the same 3 seeded batches
  <span class="success">CORRECT</span>    returns a plain Python float
  <span class="success">CORRECT</span>    pretraining lowers the validation loss from about ln 65 = 4.17 to under 2.0
:::

---

:::step id="exercise-step8" title="Step 8: loss_to_perplexity_and_bits()"
```python
def loss_to_perplexity_and_bits(loss: float) -> tuple[float, float]:
    """Convert an average cross-entropy loss (in nats) into two readouts.

    Perplexity is exp(loss): roughly the effective number of equally likely
    next-token choices. Bits per token is loss / ln(2): the same loss expressed
    in Shannon's units, i.e. the average number of bits to encode each token.

    Args:
        loss: Average cross-entropy loss in nats.

    Returns:
        (perplexity, bits_per_token).
    """
    # TODO: Return (perplexity, bits_per_token) from the average loss (in nats).
    raise NotImplementedError("TODO: perplexity and bits per token")
```
+++
**Hint:** perplexity exponentiates the loss; dividing the loss by ln(2) converts nats to bits. Use `math.exp` and `math.log`.
+++
**Answer:**

```python
perplexity = math.exp(loss)
bits_per_token = loss / math.log(2)
return perplexity, bits_per_token
```
:::

---

:::terminal id="exercise-output-step8" title="Step 8: Perplexity and Bits" cmd="uv run python module_05_pretraining/src/main.py" maxw="920px" caption="Before training the model is as unsure as a uniform guess over 65 characters. After 2,000 steps its perplexity is 5.14, and it needs 2.36 bits per character instead of 6.03."
<span class="skipped">...</span>
<span class="header">=== Step 5: train_step() ===</span> <span class="success">CORRECT</span>
<span class="skipped">  ...</span>
<span class="header">=== Step 7: estimate_loss() ===</span> <span class="success">CORRECT</span>
<span class="skipped">  ...</span>
  2000   3.00e-04    1.4379    1.6373
<span class="skipped">  ...</span>

<span class="header">=== Step 8: loss_to_perplexity_and_bits() ===</span> <span class="success">CORRECT</span>
  Validation loss, read three ways:
                            nats   perplexity   bits/token
  uniform over 65 chars   4.1744        65.00       6.0224
  before training         4.1803        65.38       6.0309
  after training          1.6373         5.14       2.3621
  <span class="success">CORRECT</span>    uniform over 65 characters: loss ln 65 gives perplexity 65 and 6.0224 bits
  <span class="success">CORRECT</span>    a loss of 0 (a perfect model) gives perplexity 1 and 0 bits
  <span class="success">CORRECT</span>    a loss of 1 nat gives perplexity e = 2.7183 and 1 / ln 2 = 1.4427 bits
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
**Hint:** use `F.softmax` over the `-1` dimension, then `torch.multinomial` to sample (pass `generator=` for reproducibility).
+++
**Answer:**

```python
probs = F.softmax(logits, dim=-1)
next_id = torch.multinomial(probs, num_samples=1, generator=generator)
```
:::

---

:::terminal id="exercise-output-step10" title="Step 10: Before and After Training" cmd="uv run python module_05_pretraining/src/main.py" maxw="920px" caption="The same seeded generator, before and after 2,000 steps. Every step is now CORRECT."
<span class="skipped">...</span>
<span class="header">=== Step 7: estimate_loss() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 8: loss_to_perplexity_and_bits() ===</span> <span class="success">CORRECT</span>
<span class="skipped">  ...</span>

<span class="header">=== Step 10: generate() ===</span> <span class="success">CORRECT</span>
  Sample before training (random weights):
    -pzlYaS ;czdeCpwEiT,YzrzlG3-aYeNB
    ijbo
    Lzzj$KUKS-A.U FisdJ'G HTobPPW;,Ue$zOAnsz-imHzPAkfEfYYgS;RTNBE:myOzk. qVh -LLcJJlZHPXBvZLUofg<span class="skipped">...</span>
  Sample after training (2,000 steps):
    FRIAR LAURENCE:
    What do tongue the cLARENCE:
    Your felsed hath you seed heart of me.

    FRIAR MARGARET:
    Pray, good Jint:
    And face where comman, I field I will be you wede no
    To die.
    <span class="skipped">...</span>
  <span class="success">CORRECT</span>    appends max_new_tokens tokens: a 2-token seed plus 5 new gives shape (1, 7)
  <span class="success">CORRECT</span>    at temperature 0.01 it matches argmax: the counting table gives 0 1 2 3 4 0 1
  <span class="success">CORRECT</span>    samples in proportion: a token with probability 0.6 comes up about 60% of the time
  <span class="success">CORRECT</span>    the same generator seed gives the same sample twice
:::

---

:::terminal id="exercise-output-overfit" title="Sanity Check: Overfit One Batch" cmd="uv run python module_05_pretraining/src/main.py --overfit" maxw="920px" caption="With <code>--overfit</code>, Steps 1&ndash;5 run as usual, then one fixed batch replaces the full run. Its loss craters from 4.18 toward zero: the loop is wired correctly."
<span class="header">=== Step 1: encode() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 2: train_val_split() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 3: get_batch() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 4: compute_loss() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 5: train_step() ===</span> <span class="success">CORRECT</span>
<span class="skipped">  ...</span>

<span class="header">=== Sanity check: overfit one batch (Steps 3-5 together) ===</span> <span class="success">CORRECT</span>
Training repeatedly on ONE batch of shape (8, 128) for 300 steps:
  step      loss
     0    4.1782
    50    2.4023
   100    1.1647
   150    0.3843
   200    0.2334
   250    0.1421
   300    0.0704
  <span class="success">CORRECT</span>    training on one fixed batch drives its loss from about 4.18 to under 0.5
:::

---

<!-- .slide: id="exercise-extra-credit" -->

## Extra Credit

- **Memorization vs generalization.** Make a tiny dataset with one phrase repeated many times, train, then check whether the model reproduces that exact phrase verbatim.
- **Prose vs code.** Swap the corpus for a file of source code and compare the samples &mdash; indentation, brackets, identifiers.
- **Gradient clipping.** The runner clips inside `train_step`. Raise the peak learning rate with clipping on vs off and watch for loss spikes.
- **Gradient accumulation.** Accumulate gradients over several micro-batches before each optimizer step, and compare the effective batch size to the per-step one. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

