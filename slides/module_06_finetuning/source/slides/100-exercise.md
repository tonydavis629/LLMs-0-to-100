:::divider id="divider-exercise" title="Exercise" sub="Finetune NanoGPT into an instruct model"
:::

---

<!-- .slide: id="exercise-run" -->

## Running the Exercise

- Open `module_06_finetuning/exercise.py`, fill in the ten `NotImplementedError` lines
- The model, LoRA plumbing, tokenizer, and dataset are provided
- Run after each step; unfinished steps are skipped automatically

```bash
# Finetune the bundled Module 5 base model on toy instruction data
cd exercises
uv run python module_06_finetuning/src/main.py
```

The base checkpoint lives at `module_06_finetuning/data/base_model.pt`. <!-- .element: class="text-md" style="margin-top: 22px;" -->

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

---

:::step id="exercise-step1" title="Step 1: format_example()"
```python
def format_example(prompt, response, special, encode_fn) -> list[int]:
    """Assemble a chat-template token sequence for one prompt-response pair."""
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
    """Next-token targets; prompt predictions are -100 (ignored)."""
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
def masked_cross_entropy(logits, targets) -> torch.Tensor:
    """Average cross-entropy over response tokens only (-100 ignored)."""
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

:::step id="exercise-step4" title="Step 4: build_optimizer()"
```python
def build_optimizer(model, lr) -> torch.optim.Optimizer:
    """AdamW over only the trainable (adapter) parameters."""
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
def lora_forward_delta(x, A, B, scale, dropout) -> torch.Tensor:
    """The low-rank update added to the frozen layer's output."""
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
def freeze_base_param(p) -> None:
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

:::step id="exercise-step8" title="Step 8: count_trainable_params()"
```python
def count_trainable_params(model) -> int:
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

:::step id="exercise-step9" title="Step 9: build_generation_prompt()"
```python
def build_generation_prompt(prompt, special, encode_fn) -> list[int]:
    """Token stream up to (and including) the assistant marker."""
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

:::step id="exercise-step10" title="Step 10: merge_lora_weight()"
```python
def merge_lora_weight(base_W, A, B, scale) -> torch.Tensor:
    """Return base_W + scale * (B @ A)."""
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

:::terminal id="exercise-output-before" title="Before Finetuning + Parameter Counts" cmd="uv run python module_06_finetuning/src/main.py" caption="The base model ignores the instruction and continues Shakespeare-style text. Only 7.41% of parameters are trainable: the LoRA adapters."
<span class="header">MODULE 6: Finetuning NanoGPT</span>
TinyGPT: 4 layers, 4 heads, width 128
Base parameters: 818,560   Vocabulary: 69 (4 special tokens)
Instruction pairs loaded: 350

<span class="header">SAMPLE BEFORE FINETUNING (base model)</span>
  prompt:   'uppercase: hello'
  full:     '<|user|>uppercase: hello<|end|><|assistant|>ers\nIn the father '
  response: 'ers\nIn the father '   &lt;- base model ignores the instruction

<span class="header">PARAMETER COUNTS (LoRA)</span>
  Trainable (LoRA adapters):  65,536
  Total (base + adapters):    884,096
  <span class="success">Fraction trainable:         7.41%</span>
:::

---

:::terminal id="exercise-output-after" title="Finetuning + the Behavioral Flip" cmd="uv run python module_06_finetuning/src/main.py" caption="The masked loss falls fast; the same prompt now answers 'HELLO'; and merging the adapter into the weights changes the logits by ~1e-5."
<span class="header">FINETUNING</span>
  step      loss
     0    6.1116
   100    0.4130
   200    0.3377
   400    0.3187
   600    0.2797
   800    0.2945
  1000    0.3003

<span class="header">SAMPLE AFTER FINETUNING (instruct model)</span>
  prompt:   'uppercase: hello'
  full:     '<|user|>uppercase: hello<|end|><|assistant|>HELLO<|end|>d dog\nA<|end|>VEL<|end|>'
  <span class="success">response: 'HELLO'   &lt;- finetuned model answers the instruction</span>

<span class="header">MERGE-EQUALITY CHECK</span>
  Max logit difference (adapter vs merged): 1.26e-05
  <span class="success">Merged model matches the adapter model: PASS</span>
:::

---

<!-- .slide: id="exercise-extra-credit" -->

## Extra Credit

- **Full finetuning vs LoRA.** Unfreeze the whole model (skip the freeze step), compare the trainable-parameter count and the sample quality against the LoRA run.
- **Vary the rank `r`.** Try `r = 1, 2, 4, 16`. Watch the quality-vs-size trade-off: how small can the adapter get before the flip stops working?
- **Catastrophic-forgetting probe.** After finetuning, feed a raw base-style prompt (e.g. `To be, or not to be`) and check whether the model still continues text or only answers instructions.
- **Loss-mask ablation.** Build targets **without** masking the prompt, retrain, and watch the model start hallucinating its own `<|user|>` prompts. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

