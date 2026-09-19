:::divider id="divider-exercise" title="Exercise" sub="Finetune NanoGPT into an instruct model"
:::

---

<!-- .slide: id="exercise-run" -->

## Running the Exercise

- Open `module_06_finetuning/exercise.py`, fill in the ten `NotImplementedError` lines
- The model, LoRA plumbing, tokenizer, and dataset are provided in `src/`
- Run after each step; each step is tagged with its test results

```bash
# Run every step; each is tagged CORRECT, INCORRECT, or INCOMPLETE
cd exercises
uv run python module_06_finetuning/src/main.py

# Run a single step (1-10)
uv run python module_06_finetuning/src/main.py --step 3
```

Step 7 finetunes the bundled Module 5 base model (`data/base_model.pt`) and takes about two minutes on a laptop CPU. <!-- .element: class="text-md" style="margin-top: 22px;" -->

---

<!-- .slide: id="exercise-results" -->

## Reading the Results

For each step the runner prints what your code produced on the real model, then runs the step's **tests** from `tests/`: each calls your function on small inputs whose correct answer is known. <!-- .element: class="text-lg" -->

- **CORRECT**: every test for the step passed
- **INCORRECT**: your code ran but a test failed; the expected and actual values are printed beneath
- **INCOMPLETE**: the function still raises `NotImplementedError`, or an earlier step it needs is unfinished

The tag sits on the step's header line. A falling loss or a plausible sample does not prove a step is right; the tests do. <!-- .element: class="text-lg" style="margin-top: 15px;" -->

---

:::terminal id="exercise-results-example" title="What an INCORRECT Step Looks Like" cmd="uv run python module_06_finetuning/src/main.py --step 2" maxw="1000px" caption="Here <code>build_targets()</code> masked <code>prompt_span</code> positions instead of <code>prompt_span - 1</code>. The assistant marker no longer predicts <strong>p</strong>, so the model trains on <code>'ets&lt;|end|&gt;'</code> and never learns the first letter of an answer. The failing tests show the expected and actual targets and name the fix."
<span class="header">=== Step 2: build_targets() ===</span> <span class="t-fail">INCORRECT</span>
Example pair: 'reverse: step' -&gt; 'pets'
  21 tokens, the first 16 are the prompt (prompt_span = 16)
  masked to -100: 17 positions
  trained:        4 positions, targets 'ets&lt;|end|&gt;'
  <span class="t-fail">INCORRECT</span>  ids [10..16] with prompt_span=4 gives [-100, -100, -100, 14, 15, 16, -100]
             expected [-100, -100, -100, 14, 15, 16, -100]
             got      [-100, -100, -100, -100, 15, 16, -100]
  <span class="success">CORRECT</span>    returns one target per input position (10 ids give 10 targets)
  <span class="t-fail">INCORRECT</span>  prompt_span=3: the assistant marker's position predicts ids[3], the first response token
             expected 23, got -100 (mask prompt_span - 1 positions, not prompt_span)
  <span class="success">CORRECT</span>    every unmasked target is the next token: targets[t] == ids[t + 1]
:::

---

<!-- .slide: id="exercise-overview" -->

## Exercise: From Base to Assistant

The base checkpoint is the **frozen Module 5 model**. You implement the **finetuning loop**: chat template, loss mask, LoRA adapter, merge.

:::columns cols="2" gap="30px"
**The payoff**

- `uppercase: hello` flips from Shakespeare-style continuation (base) to `HELLO` (finetuned)
- Under **8%** of parameters trained
+++
**Ten one-line steps**

Format the template, mask the loss, compute masked cross-entropy, build the optimizer, implement the LoRA delta, freeze the base, run a step, count parameters, build the generation prompt, merge.
:::

Every step has its own tests in `tests/`. <!-- .element: class="text-lg" style="margin-top: 18px;" -->

---

:::step id="exercise-step1" title="Step 1: format_example()"
```python
def format_example(
    prompt: str,
    response: str,
    special: dict[str, int],
    encode_fn,
) -> list[int]:
    """Assemble a chat-template token sequence for one prompt-response pair.

    The template is:
        [user] + encode(prompt) + [end] + [assistant] + encode(response) + [end]

    Args:
        prompt: The user's instruction text.
        response: The assistant's desired response text.
        special: Mapping from special token strings to their integer IDs.
        encode_fn: Function that maps a plain string to a list of token IDs.

    Returns:
        A flat list of token IDs representing the full formatted example.
    """
    # TODO: Return the chat-template token ids:
    #       [user] + encode(prompt) + [end] + [assistant] + encode(response) + [end].
    raise NotImplementedError("TODO: assemble the chat-template token sequence")
```
+++
**Hint:** look up the marker ids in `special` (e.g. `special["<|user|>"]`) and call `encode_fn(prompt)` / `encode_fn(response)` for the text; join lists with `+`.
+++
**Answer:**

```python
return (
    [special["<|user|>"]]
    + encode_fn(prompt)
    + [special["<|end|>"]]
    + [special["<|assistant|>"]]
    + encode_fn(response)
    + [special["<|end|>"]]
)
```
:::

---

:::step id="exercise-step2" title="Step 2: build_targets() &mdash; the loss mask"
```python
def build_targets(ids: list[int], prompt_span: int) -> list[int]:
    """Build next-token targets where prompt predictions are set to -100 (ignored).

    Target t is the token that position t should predict, i.e. ids[t + 1]. We keep
    only the predictions of the response tokens. The first response token is
    predicted at position prompt_span - 1 (the assistant marker), so positions
    0 .. prompt_span - 2 are masked, the response targets are ids[prompt_span:],
    and the final position is -100 (no token follows the last one).

    Args:
        ids: The full formatted token sequence.
        prompt_span: Number of leading tokens that belong to the prompt (user turn + markers).

    Returns:
        A list of target IDs the same length as ids.
    """
    # TODO: Return next-token targets: the first prompt_span - 1 positions are -100
    #       (ignored), the response targets are ids[prompt_span:], and the final
    #       position is -100 (no token follows the last one).
    raise NotImplementedError("TODO: build the masked next-token targets")
```
+++
**Hint:** `[-100] * (prompt_span - 1)` masks the prompt predictions; `ids[prompt_span:]` are the response targets; append one more `-100` for the final position.
+++
**Answer:**

```python
return [-100] * (prompt_span - 1) + ids[prompt_span:] + [-100]
```
:::

---

:::step id="exercise-step3" title="Step 3: masked_cross_entropy()"
```python
def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Average cross-entropy over the response tokens only (-100 positions are ignored).

    Args:
        logits: Model outputs, shape (batch, time, vocab_size).
        targets: Target IDs with -100 for prompt positions, shape (batch, time).

    Returns:
        Scalar loss tensor.
    """
    vocab_size = logits.shape[-1]
    # TODO: Return the average cross-entropy over the response tokens, ignoring -100.
    raise NotImplementedError("TODO: masked cross-entropy with ignore_index=-100")
```
+++
**Hint:** flatten `logits` to `(-1, vocab_size)` and `targets` to `(-1)`, then call `F.cross_entropy` with `ignore_index=-100`.
+++
**Answer:**

```python
return F.cross_entropy(logits.view(-1, vocab_size),
                       targets.view(-1), ignore_index=-100)
```
:::

---

:::terminal id="exercise-output-steps1-3" title="Steps 1&ndash;3: Output" cmd="uv run python module_06_finetuning/src/main.py" maxw="1000px" caption="The template and the mask turn real pairs into a batch. On the response tokens, the untrained base model scores 6.11, worse than a uniform guess over 69 tokens (4.23): it learned Shakespeare, not this format."
<span class="header">=== Step 1: format_example() ===</span> <span class="success">CORRECT</span>
<span class="t-gray">  decoded: '&lt;|user|&gt;reverse: step&lt;|end|&gt;&lt;|assistant|&gt;pets&lt;|end|&gt;'</span>
<span class="header">=== Step 2: build_targets() ===</span> <span class="success">CORRECT</span>
<span class="t-gray">  trained:        5 positions, targets 'pets&lt;|end|&gt;'</span>
<span class="skipped">  ...</span>

<span class="header">=== Step 3: masked_cross_entropy() ===</span> <span class="success">CORRECT</span>
Base model on one batch of 16 pairs (99 response tokens scored)
  masked loss:    6.1116
  uniform guess:  4.2341 (ln 69)
  <span class="success">CORRECT</span>    uniform logits over 4 tokens cost ln 4 = 1.3863 per position
  <span class="success">CORRECT</span>    a -100 position adds nothing: [real, masked] costs the real position alone, 0.3408
  <span class="success">CORRECT</span>    matches F.cross_entropy over just the unmasked positions of a random (2, 6, 69) batch
  <span class="success">CORRECT</span>    returns a scalar tensor that can backpropagate (not a Python float)

<span class="header">=== Step 4: build_optimizer() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">TODO: build AdamW over the trainable adapter params</span>

<span class="skipped">...</span>
:::

---

:::step id="exercise-step4" title="Step 4: build_optimizer()"
```python
def build_optimizer(model: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
    """Return an AdamW optimizer over only the trainable (adapter) parameters.

    Args:
        model: The model with LoRA injected and base weights frozen.
        lr: The small finetuning learning rate.

    Returns:
        A torch.optim.Optimizer configured for the adapter parameters.
    """
    # TODO: Return an AdamW optimizer over only the trainable (requires_grad) params.
    raise NotImplementedError("TODO: build AdamW over the trainable adapter params")
```
+++
**Hint:** collect `[p for p in model.parameters() if p.requires_grad]`, then pass that list to `torch.optim.AdamW` with `lr=lr`.
+++
**Answer:**

```python
trainable = [p for p in model.parameters() if p.requires_grad]
return torch.optim.AdamW(trainable, lr=lr)
```
:::

---

:::step id="exercise-step5" title="Step 5: lora_forward_delta()"
```python
def lora_forward_delta(
    x: torch.Tensor,
    A: torch.nn.Parameter,
    B: torch.nn.Parameter,
    scale: float,
    dropout: torch.nn.Module,
) -> torch.Tensor:
    """Compute the low-rank update added to the frozen layer's output.

    The update is:  scale * (dropout(x) @ A.t() @ B.t())

    Args:
        x: The layer input, shape (..., in_features).
        A: LoRA A matrix, shape (r, in_features).
        B: LoRA B matrix, shape (out_features, r).
        scale: alpha / r.
        dropout: A dropout (or Identity) module applied to x.

    Returns:
        The low-rank delta, shape (..., out_features).
    """
    # TODO: Return the low-rank update scale * (dropout(x) @ A.t() @ B.t()).
    raise NotImplementedError("TODO: compute the LoRA low-rank delta")
```
+++
**Hint:** apply `dropout` to `x`, matrix-multiply by `A.t()` then `B.t()`, scale the result.
+++
**Answer:**

```python
return scale * (dropout(x) @ A.t() @ B.t())
```
:::

---

:::step id="exercise-step6" title="Step 6: freeze_base_param()"
```python
def freeze_base_param(p: torch.nn.Parameter) -> None:
    """Freeze a single base parameter so the optimizer never updates it."""
    # TODO: Freeze this parameter so it receives no gradient updates.
    raise NotImplementedError("TODO: freeze this base parameter")
```
+++
**Hint:** set the parameter's `requires_grad` attribute to `False`.
+++
**Answer:**

```python
p.requires_grad = False
```
:::

---

:::terminal id="exercise-output-steps4-6" title="Steps 4&ndash;6: Output" cmd="uv run python module_06_finetuning/src/main.py" maxw="1000px" caption="Every B starts at zero, so wrapping 16 layers with LoRA changes no logit. After the freeze, only the 32 adapter tensors can train."
<span class="header">=== Step 1: format_example() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 2: build_targets() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 3: masked_cross_entropy() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 4: build_optimizer() ===</span> <span class="success">CORRECT</span>
<span class="skipped">  ...</span>

<span class="header">=== Step 5: lora_forward_delta() ===</span> <span class="success">CORRECT</span>
LoRA (r=8, alpha=32, scale=4) wraps 16 linear layers
Every B starts at zero, so on 'uppercase: hello' the wrapped model matches the base model:
  max logit difference 0.00e+00
  <span class="success">CORRECT</span>    x=[1,2], A=[[1,1]], B=[[2],[0],[1]], scale=0.5 gives [3, 0, 1.5]
  <span class="success">CORRECT</span>    equals x @ (scale * B @ A).t() on a random (2, 5, 8) batch, output shape (2, 5, 6)
  <span class="success">CORRECT</span>    B = 0 (how LoRA starts) gives an all-zero update, so the adapter begins as a no-op
  <span class="success">CORRECT</span>    passes x through dropout first (nn.Dropout(p=1.0) drops every input, so the update is 0)

<span class="header">=== Step 6: freeze_base_param() ===</span> <span class="success">CORRECT</span>
Base tensors frozen:     52 of 52 (embeddings, layer norms, and every wrapped Linear)
LoRA tensors trainable:  32 of 32 (A and B in each of the 16 LoRA layers)
  <span class="success">CORRECT</span>    sets requires_grad to False on the parameter it is given
  <span class="success">CORRECT</span>    leaves the parameter's values unchanged
  <span class="success">CORRECT</span>    backward() then skips it: weight.grad stays None, the unfrozen bias still gets one
  <span class="success">CORRECT</span>    real model: all 52 base tensors are frozen and all 32 LoRA tensors (A and B) still train

<span class="header">=== Step 7: sft_train_step() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">TODO: zero the gradients and backpropagate the loss</span>

<span class="skipped">...</span>
:::

---

:::step id="exercise-step7" title="Step 7: sft_train_step()"
```python
    logits = model(x)
    loss = masked_cross_entropy(logits, y)

    # TODO: Clear last step's gradients, then backpropagate this step's loss.
    raise NotImplementedError("TODO: zero the gradients and backpropagate the loss")

    # Provided: clip the global gradient norm for stability, then take the step.
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return loss.item()
```
+++
**Hint:** the optimizer has a method to zero gradients (use `set_to_none=True`); the loss tensor has a method that backpropagates.
+++
**Answer:**

```python
optimizer.zero_grad(set_to_none=True)
loss.backward()
```
:::

---

:::terminal id="exercise-output-step7" title="Step 7: Finetuning" cmd="uv run python module_06_finetuning/src/main.py" maxw="1000px" caption="The masked loss falls from 6.11 to 0.34 by step 200 and then hovers around 0.3. The first three tests grade <code>sft_train_step()</code> on a 4-token toy model worked out by hand; the last one checks this run."
<span class="header">=== Step 1: format_example() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 2: build_targets() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 3: masked_cross_entropy() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 4: build_optimizer() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 5: lora_forward_delta() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 6: freeze_base_param() ===</span> <span class="success">CORRECT</span>
<span class="skipped">  ...</span>

<span class="header">=== Step 7: sft_train_step() ===</span> <span class="success">CORRECT</span>
  step      loss
     0    6.1116
   100    0.4182
   200    0.3391
<span class="skipped">  ...</span>
  1000    0.3012
  <span class="success">CORRECT</span>    returns the batch loss as a float: uniform logits over 4 tokens give ln 4 = 1.3863
  <span class="success">CORRECT</span>    one SGD step (lr=1) on a zero embedding moves row 1 to [-0.125, -0.125, 0.375, -0.125]
  <span class="success">CORRECT</span>    clears the previous step's gradients first (a leftover .grad of 100 changes nothing)
  <span class="success">CORRECT</span>    finetuning drives the masked loss below 0.5 (from 6.11 at step 0)

<span class="header">=== Step 8: count_trainable_params() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">TODO: count the trainable parameters</span>

<span class="skipped">...</span>
:::

---

:::step id="exercise-step8" title="Step 8: count_trainable_params()"
```python
def count_trainable_params(model: torch.nn.Module) -> int:
    """Count parameters where requires_grad is True."""
    # TODO: Return the number of parameters with requires_grad=True.
    raise NotImplementedError("TODO: count the trainable parameters")
```
+++
**Hint:** sum `p.numel()` over `model.parameters()` where `p.requires_grad` is True.
+++
**Answer:**

```python
return sum(p.numel() for p in model.parameters() if p.requires_grad)
```
:::

---

:::terminal id="exercise-output-step8" title="Step 8: Output" cmd="uv run python module_06_finetuning/src/main.py" maxw="1000px" caption="LoRA trains 65,536 of 884,096 numbers, 7.41% of the model. The real-model test checks that the count is exactly the size of the A and B matrices."
<span class="header">=== Step 1: format_example() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 2: build_targets() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 3: masked_cross_entropy() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 4: build_optimizer() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 5: lora_forward_delta() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 6: freeze_base_param() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 7: sft_train_step() ===</span> <span class="success">CORRECT</span>
<span class="skipped">  ...</span>

<span class="header">=== Step 8: count_trainable_params() ===</span> <span class="success">CORRECT</span>
Trainable (LoRA adapters):  65,536
Total (base + adapters):    884,096
Fraction trainable:         7.41%
  <span class="success">CORRECT</span>    counts every element: nn.Linear(3, 2) has 2*3 + 2 = 8 trainable numbers
  <span class="success">CORRECT</span>    skips frozen tensors: a rank-2 adapter on a frozen 4 -&gt; 3 layer gives 2*4 + 3*2 = 14
  <span class="success">CORRECT</span>    a fully frozen layer has 0 trainable parameters
  <span class="success">CORRECT</span>    real model: the count is exactly the LoRA A and B matrices, 65,536 numbers

<span class="header">=== Step 9: build_generation_prompt() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">TODO: assemble the generation prompt up to the assistant marker</span>

<span class="skipped">...</span>
:::

---

:::step id="exercise-step9" title="Step 9: build_generation_prompt()"
```python
def build_generation_prompt(
    prompt: str,
    special: dict[str, int],
    encode_fn,
) -> list[int]:
    """Assemble the token stream up to (and including) the assistant marker.

    Template:
        [user] + encode(prompt) + [end] + [assistant]

    Args:
        prompt: The user's instruction text.
        special: Mapping from special token strings to their integer IDs.
        encode_fn: Function mapping a plain string to a list of token IDs.

    Returns:
        A flat list of token IDs ready for autoregressive generation.
    """
    # TODO: Return the token ids up to the assistant marker (no response yet):
    #       [user] + encode(prompt) + [end] + [assistant].
    raise NotImplementedError("TODO: assemble the generation prompt up to the assistant marker")
```
+++
**Hint:** same as `format_example` but stop right after `special["<|assistant|>"]`.
+++
**Answer:**

```python
return (
    [special["<|user|>"]]
    + encode_fn(prompt)
    + [special["<|end|>"]]
    + [special["<|assistant|>"]]
)
```
:::

---

:::terminal id="exercise-output-step9" title="Step 9: The Behavioral Flip" cmd="uv run python module_06_finetuning/src/main.py" maxw="1000px" caption="Same prompt, same sampling seed. The base model continues Shakespeare-style text. The finetuned model answers <code>HELLO</code> and closes its turn with <code>&lt;|end|&gt;</code>; it keeps generating past that marker, so the runner cuts the response at the first one."
<span class="header">=== Step 1: format_example() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 2: build_targets() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 3: masked_cross_entropy() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 4: build_optimizer() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 5: lora_forward_delta() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 6: freeze_base_param() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 7: sft_train_step() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 8: count_trainable_params() ===</span> <span class="success">CORRECT</span>
<span class="skipped">  ...</span>

<span class="header">=== Step 9: build_generation_prompt() ===</span> <span class="success">CORRECT</span>
prompt: 'uppercase: hello'
Base model (before finetuning):
  full:     '&lt;|user|&gt;uppercase: hello&lt;|end|&gt;&lt;|assistant|&gt;ers\nIn the father '
  response: 'ers\nIn the father '   &lt;- ignores the instruction
Finetuned model (after Step 7):
  full:     '&lt;|user|&gt;uppercase: hello&lt;|end|&gt;&lt;|assistant|&gt;HELLO&lt;|end|&gt;d doge&lt;|end|&gt;t&lt;|end|&gt;Mbl'
  response: 'HELLO'   &lt;- answers the instruction
  <span class="success">CORRECT</span>    prompt 'ab' gives [user, a, b, end, assistant], with no response yet
  <span class="success">CORRECT</span>    ends on the assistant marker, so the next generated token starts the answer
  <span class="success">CORRECT</span>    real vocabulary: 'uppercase: hello' decodes to '&lt;|user|&gt;uppercase: hello&lt;|end|&gt;&lt;|assistant|&gt;'

<span class="header">=== Step 10: merge_lora_weight() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">TODO: merge the LoRA update into the base weight</span>
:::

---

:::step id="exercise-step10" title="Step 10: merge_lora_weight()"
```python
def merge_lora_weight(base_W: torch.Tensor, A: torch.Tensor, B: torch.Tensor, scale: float) -> torch.Tensor:
    """Return base_W + scale * (B @ A).

    Args:
        base_W: The frozen pretrained weight matrix.
        A: LoRA A matrix (r x in).
        B: LoRA B matrix (out x r).
        scale: alpha / r.

    Returns:
        The merged weight matrix of the same shape as base_W.
    """
    # TODO: Return the merged weight base_W + scale * (B @ A).
    raise NotImplementedError("TODO: merge the LoRA update into the base weight")
```
+++
**Hint:** matrix-multiply `B @ A` (shape out x in), scale it, add to `base_W`.
+++
**Answer:**

```python
return base_W + scale * (B @ A)
```
:::

---

:::terminal id="exercise-output-all" title="All Ten Steps: Output" cmd="uv run python module_06_finetuning/src/main.py" maxw="1000px" caption="Merging folds each <code>scale * (B @ A)</code> back into its weight. The model is back to 818,560 parameters with no adapters, and its logits move by only 6.68e-06, float32 rounding."
<span class="header">=== Step 1: format_example() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 2: build_targets() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 3: masked_cross_entropy() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 4: build_optimizer() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 5: lora_forward_delta() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 6: freeze_base_param() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 7: sft_train_step() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 8: count_trainable_params() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 9: build_generation_prompt() ===</span> <span class="success">CORRECT</span>
<span class="t-gray">Base model (before finetuning):</span>
<span class="t-gray">  response: 'ers\nIn the father '   &lt;- ignores the instruction</span>
<span class="t-gray">Finetuned model (after Step 7):</span>
<span class="t-gray">  response: 'HELLO'   &lt;- answers the instruction</span>
<span class="skipped">  ...</span>

<span class="header">=== Step 10: merge_lora_weight() ===</span> <span class="success">CORRECT</span>
Merged 16 LoRA layers into plain nn.Linear weights: 884,096 -&gt; 818,560 parameters
  Max logit difference (adapter vs merged): 6.68e-06
  <span class="success">CORRECT</span>    W = I, A = [[1, 2]], B = [[1], [0]], scale = 0.5 gives [[1.5, 1], [0, 1]]
  <span class="success">CORRECT</span>    returns a new tensor and leaves base_W itself unchanged
  <span class="success">CORRECT</span>    one merged layer gives the same output as base layer + adapter (random 8 -&gt; 6 layer, r = 4)
  <span class="success">CORRECT</span>    real model: merged and adapter logits agree (max difference under 1e-4)
:::

---

<!-- .slide: id="exercise-extra-credit" -->

## Extra Credit

- **Full finetuning vs LoRA.** Unfreeze the whole model (skip the freeze step), compare the trainable-parameter count and the sample quality against the LoRA run.
- **Vary the rank `r`.** Try `r = 1, 2, 4, 16`. Watch the quality-vs-size trade-off: how small can the adapter get before the flip stops working?
- **Catastrophic-forgetting probe.** After finetuning, feed a raw base-style prompt (e.g. `To be, or not to be`) and check whether the model still continues text or only answers instructions.
- **Loss-mask ablation.** Build targets **without** masking the prompt, retrain, and watch the model start hallucinating its own `<|user|>` prompts. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

