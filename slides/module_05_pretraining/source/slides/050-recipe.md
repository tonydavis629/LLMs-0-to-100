:::divider id="divider-recipe" title="The Training Recipe" sub="The engineering that keeps a long run stable"
:::

---

<!-- .slide: id="loop-overview" -->

## The Training Loop in Code

:::columns grid="1.45fr 1fr" gap="30px"
```python []
model = GPT(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-3, weight_decay=0.1)

for step in range(max_steps):
    lr = lr_at_step(step)
    for group in optimizer.param_groups:
        group["lr"] = lr

    x, y = get_batch(train_data, block_size, batch_size)

    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```
+++
**Module 2's loop, plus two additions**

- Lines 1-2: build the model and optimizer, once
- Lines 5-7: set this step's learning rate
- Line 9: sample a batch
- Lines 11-12: forward pass and loss
- Lines 14-15: backward pass
- Lines 17-18: clip, then update

The schedule (5-7) and clipping (17) are what pretraining adds. The exercise runs this loop 800 times.
:::

---

<!-- .slide: id="loop-setup" -->

## Loop Step 1: Model and Optimizer

:::columns grid="1.45fr 1fr" gap="30px"
```python [1-2]
model = GPT(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-3, weight_decay=0.1)

for step in range(max_steps):
    lr = lr_at_step(step)
    for group in optimizer.param_groups:
        group["lr"] = lr

    x, y = get_batch(train_data, block_size, batch_size)

    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```
+++
- `model` starts with random weights, so its predictions are close to uniform: loss $\approx \ln V$
- The exercise's 65-character vocabulary starts near $\ln 65 = 4.17$
- `AdamW` keeps two running averages for **every** weight, so its state is twice the size of the model
:::

---

<!-- .slide: id="loop-lr" -->

## Loop Step 2: Set the Learning Rate

:::columns grid="1.45fr 1fr" gap="30px"
```python [5-7]
model = GPT(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-3, weight_decay=0.1)

for step in range(max_steps):
    lr = lr_at_step(step)
    for group in optimizer.param_groups:
        group["lr"] = lr

    x, y = get_batch(train_data, block_size, batch_size)

    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```
+++
- The rate changes every step: warm up, then cosine decay (two slides ahead)
- PyTorch stores the rate in `optimizer.param_groups`
- Overwriting it there each step is how a schedule is applied
:::

---

<!-- .slide: id="loop-batch" -->

## Loop Step 3: Sample a Batch

:::columns grid="1.45fr 1fr" gap="30px"
```python [9]
model = GPT(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-3, weight_decay=0.1)

for step in range(max_steps):
    lr = lr_at_step(step)
    for group in optimizer.param_groups:
        group["lr"] = lr

    x, y = get_batch(train_data, block_size, batch_size)

    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```
+++
- Pick `batch_size` random offsets into the token stream and slice `block_size` tokens at each
- `x` and `y` both have shape $(B, T)$
- `y` is `x` shifted left by one, so `y[b, t]` is the token that follows `x[b, t]`
- One batch holds $B \times T$ next-token problems
:::

---

<!-- .slide: id="loop-forward" -->

## Loop Step 4: Forward Pass and Loss

:::columns grid="1.45fr 1fr" gap="30px"
```python [11-12]
model = GPT(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-3, weight_decay=0.1)

for step in range(max_steps):
    lr = lr_at_step(step)
    for group in optimizer.param_groups:
        group["lr"] = lr

    x, y = get_batch(train_data, block_size, batch_size)

    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```
+++
- `model(x)` returns logits of shape $(B, T, V)$: a score for every vocabulary token at every position
- `cross_entropy` expects one row of scores per prediction, so `.view` flattens logits to $(BT, V)$ and targets to $(BT,)$
- The result is one number: the mean of $-\log p(\text{true next token})$ over all $BT$ positions
:::

---

<!-- .slide: id="loop-backward" -->

## Loop Step 5: Backward Pass

:::columns grid="1.45fr 1fr" gap="30px"
```python [14-15]
model = GPT(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-3, weight_decay=0.1)

for step in range(max_steps):
    lr = lr_at_step(step)
    for group in optimizer.param_groups:
        group["lr"] = lr

    x, y = get_batch(train_data, block_size, batch_size)

    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```
+++
- `loss.backward()` runs the chain rule (Module 2) from the loss back through every layer
- It fills `p.grad` for every parameter `p`
- PyTorch **adds** into `.grad` instead of overwriting it, so `zero_grad()` must clear last step's gradients first
- Skip it and every update uses the sum of all past gradients
:::

---

<!-- .slide: id="loop-update" -->

## Loop Step 6: Clip and Update

:::columns grid="1.45fr 1fr" gap="30px"
```python [17-18]
model = GPT(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-3, weight_decay=0.1)

for step in range(max_steps):
    lr = lr_at_step(step)
    for group in optimizer.param_groups:
        group["lr"] = lr

    x, y = get_batch(train_data, block_size, batch_size)

    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```
+++
- `clip_grad_norm_` measures the norm of all gradients together; above 1.0, it scales every gradient down by the same factor
- The direction of the step is unchanged, only its length is capped
- `optimizer.step()` moves each weight using its gradient and its AdamW running averages
- Then the loop repeats with the next batch
:::

---

<!-- .slide: id="adamw" -->

## AdamW: The Standard Optimizer

**AdamW** = Adam's adaptive step sizes + **decoupled weight decay**.

:::columns cols="2" gap="34px"
**Adam part**

- Running averages of the gradient and its square
- Each parameter gets its own effective learning rate
- Robust to the very different gradient scales in a deep model
+++
**Weight-decay part**

- Pulls weights toward zero each step (regularization, Module 2)
- "Decoupled": applied directly to the weights, separate from the adaptive gradient term
- Works better in practice
:::

---

<!-- .slide: id="lr-schedule" -->

## Learning-Rate Schedule: Warmup, Then Cosine Decay

:::columns cols="2" gap="34px"
**Warmup**

- Start near zero, ramp up over the first few hundred steps
- Early weights are random and gradients large; small steps avoid a blow-up
+++
**Cosine decay**

- After the peak, decay smoothly to a small floor
- Big steps early to explore, small steps late to settle
:::

---

:::manim id="lr-anim" scene="lr-schedule"
:::

---

<!-- .slide: id="grad-clipping" -->

## Gradient Clipping

:::columns grid="1.15fr 1fr" gap="34px" valign="center"
<div class="stage-flow"><svg viewBox="0 0 560 390" role="img" aria-label="A long gradient arrow is shortened to the clipping circle of radius c, keeping its direction; a short gradient inside the circle is unchanged"><defs><marker id="gcr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#e74c3c"></path></marker><marker id="gcg" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#3fb950"></path></marker><marker id="gcb" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#4a9eff"></path></marker></defs><circle cx="200" cy="240" r="110" fill="rgba(136,146,164,0.06)" stroke="#8892a4" stroke-width="1.5" stroke-dasharray="6 5"></circle><text x="200" y="378" text-anchor="middle" font-size="15" fill="#8892a4">circle: norm = c</text><line x1="200" y1="240" x2="500" y2="40" stroke="#e74c3c" stroke-width="2.5" stroke-dasharray="7 5" marker-end="url(#gcr)"></line><text x="490" y="28" text-anchor="end" font-size="16" fill="#e74c3c">raw gradient, norm 3.3c</text><line x1="200" y1="240" x2="291.5" y2="179.0" stroke="#3fb950" stroke-width="4" marker-end="url(#gcg)"></line><text x="307.5" y="203.0" font-size="16" fill="#3fb950">clipped: same direction, norm c</text><line x1="200" y1="240" x2="140" y2="178" stroke="#4a9eff" stroke-width="3" marker-end="url(#gcb)"></line><text x="20" y="100" font-size="15" fill="#4a9eff">small gradient: unchanged</text><circle cx="200" cy="240" r="4" fill="#e8eaf0"></circle></svg></div>
+++
$$\mathbf g \leftarrow \mathbf g \cdot \min\left(1, \frac{c}{\lVert \mathbf g \rVert}\right)$$

- A rare batch can produce a huge gradient, and one step along it is a **loss spike**
- Clipping caps the step length at $c$ without changing its direction
- The norm covers all parameters at once. GPT-3 used $c = 1.0$

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```
:::

---

<!-- .slide: id="grad-accumulation" -->

## Gradient Accumulation

Bigger batches average over more examples, so the gradient is less noisy and big runs train more stably. But a batch of millions of tokens does not fit in one GPU's memory.

<div class="stage-flow"><svg viewBox="0 0 900 300" role="img" aria-label="Four micro-batches each run forward and backward; their gradients add into p.grad; one optimizer step follows"><defs><marker id="gab" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#4a9eff"></path></marker><marker id="gaa" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#f5a623"></path></marker></defs><rect x="25" y="14" width="190" height="62" rx="10" fill="rgba(74,158,255,0.10)" stroke="rgba(74,158,255,0.55)" stroke-width="1.4"></rect><text x="120" y="40" text-anchor="middle" font-size="16" fill="#e8eaf0">micro-batch 1</text><text x="120" y="62" text-anchor="middle" font-size="13" fill="#8892a4">forward, backward</text><line x1="120" y1="80" x2="270" y2="140" stroke="#4a9eff" stroke-width="2" marker-end="url(#gab)"></line><rect x="245" y="14" width="190" height="62" rx="10" fill="rgba(74,158,255,0.10)" stroke="rgba(74,158,255,0.55)" stroke-width="1.4"></rect><text x="340" y="40" text-anchor="middle" font-size="16" fill="#e8eaf0">micro-batch 2</text><text x="340" y="62" text-anchor="middle" font-size="13" fill="#8892a4">forward, backward</text><line x1="340" y1="80" x2="390" y2="140" stroke="#4a9eff" stroke-width="2" marker-end="url(#gab)"></line><rect x="465" y="14" width="190" height="62" rx="10" fill="rgba(74,158,255,0.10)" stroke="rgba(74,158,255,0.55)" stroke-width="1.4"></rect><text x="560" y="40" text-anchor="middle" font-size="16" fill="#e8eaf0">micro-batch 3</text><text x="560" y="62" text-anchor="middle" font-size="13" fill="#8892a4">forward, backward</text><line x1="560" y1="80" x2="510" y2="140" stroke="#4a9eff" stroke-width="2" marker-end="url(#gab)"></line><rect x="685" y="14" width="190" height="62" rx="10" fill="rgba(74,158,255,0.10)" stroke="rgba(74,158,255,0.55)" stroke-width="1.4"></rect><text x="780" y="40" text-anchor="middle" font-size="16" fill="#e8eaf0">micro-batch 4</text><text x="780" y="62" text-anchor="middle" font-size="13" fill="#8892a4">forward, backward</text><line x1="780" y1="80" x2="630" y2="140" stroke="#4a9eff" stroke-width="2" marker-end="url(#gab)"></line><rect x="200" y="146" width="500" height="58" rx="10" fill="rgba(245,166,35,0.10)" stroke="rgba(245,166,35,0.6)" stroke-width="1.6"></rect><text x="450" y="172" text-anchor="middle" font-size="17" fill="#e8eaf0">p.grad: the four gradients add up</text><text x="450" y="193" text-anchor="middle" font-size="13" fill="#8892a4">dividing each loss by 4 makes the sum the mean</text><line x1="450" y1="208" x2="450" y2="236" stroke="#f5a623" stroke-width="2.5" marker-end="url(#gaa)"></line><rect x="270" y="242" width="360" height="48" rx="10" fill="rgba(63,185,80,0.12)" stroke="rgba(63,185,80,0.6)" stroke-width="1.6"></rect><text x="450" y="272" text-anchor="middle" font-size="16" fill="#e8eaf0">one optimizer.step(), then zero_grad()</text></svg></div>

:::columns cols="2" gap="34px"
- GPT-3 used 3.2 million tokens per step
- Run the batch as small **micro-batches**: `.grad` adds up across `backward()` calls on its own
+++
- Same update as one big batch, paid for in time instead of memory
- Effective batch = micro-batch × steps × number of GPUs
:::

---

<!-- .slide: id="number-formats" -->

## Mixed Precision: Number Formats

<div class="stage-flow">
<svg viewBox="0 0 1000 210" role="img" aria-label="Bit layouts: fp32 has 1 sign, 8 exponent, 23 mantissa bits; fp16 has 1, 5, 10; bf16 has 1, 8, 7"><text x="70" y="41" text-anchor="end" font-size="17" fill="#e8eaf0">fp32</text><rect x="90" y="20" width="19" height="30" rx="3" fill="rgba(136,146,164,0.30)" stroke="#8892a4" stroke-width="1"></rect><rect x="112" y="20" width="19" height="30" rx="3" fill="rgba(74,158,255,0.30)" stroke="#4a9eff" stroke-width="1"></rect><rect x="134" y="20" width="19" height="30" rx="3" fill="rgba(74,158,255,0.30)" stroke="#4a9eff" stroke-width="1"></rect><rect x="156" y="20" width="19" height="30" rx="3" fill="rgba(74,158,255,0.30)" stroke="#4a9eff" stroke-width="1"></rect><rect x="178" y="20" width="19" height="30" rx="3" fill="rgba(74,158,255,0.30)" stroke="#4a9eff" stroke-width="1"></rect><rect x="200" y="20" width="19" height="30" rx="3" fill="rgba(74,158,255,0.30)" stroke="#4a9eff" stroke-width="1"></rect><rect x="222" y="20" width="19" height="30" rx="3" fill="rgba(74,158,255,0.30)" stroke="#4a9eff" stroke-width="1"></rect><rect x="244" y="20" width="19" height="30" rx="3" fill="rgba(74,158,255,0.30)" stroke="#4a9eff" stroke-width="1"></rect><rect x="266" y="20" width="19" height="30" rx="3" fill="rgba(74,158,255,0.30)" stroke="#4a9eff" stroke-width="1"></rect><rect x="288" y="20" width="19" height="30" rx="3" fill="rgba(245,166,35,0.25)" stroke="#f5a623" stroke-width="1"></rect><rect x="310" y="20" width="19" height="30" rx="3" fill="rgba(245,166,35,0.25)" stroke="#f5a623" stroke-width="1"></rect><rect x="332" y="20" width="19" height="30" rx="3" fill="rgba(245,166,35,0.25)" stroke="#f5a623" stroke-width="1"></rect><rect x="354" y="20" width="19" height="30" rx="3" fill="rgba(245,166,35,0.25)" stroke="#f5a623" stroke-width="1"></rect><rect x="376" y="20" width="19" height="30" rx="3" fill="rgba(245,166,35,0.25)" stroke="#f5a623" stroke-width="1"></rect><rect x="398" y="20" width="19" height="30" rx="3" fill="rgba(245,166,35,0.25)" stroke="#f5a623" stroke-width="1"></rect><rect x="420" y="20" width="19" height="30" rx="3" fill="rgba(245,166,35,0.25)" stroke="#f5a623" stroke-width="1"></rect><rect x="442" y="20" width="19" height="30" rx="3" fill="rgba(245,166,35,0.25)" stroke="#f5a623" stroke-width="1"></rect><rect x="464" y="20" width="19" height="30" rx="3" fill="rgba(245,166,35,0.25)" stroke="#f5a623" stroke-width="1"></rect><rect x="486" y="20" width="19" height="30" rx="3" fill="rgba(245,166,35,0.25)" stroke="#f5a623" stroke-width="1"></rect><rect x="508" y="20" width="19" height="30" rx="3" fill="rgba(245,166,35,0.25)" stroke="#f5a623" stroke-width="1"></rect><rect x="530" y="20" width="19" height="30" rx="3" fill="rgba(245,166,35,0.25)" stroke="#f5a623" stroke-width="1"></rect><rect x="552" y="20" width="19" height="30" rx="3" fill="rgba(245,166,35,0.25)" stroke="#f5a623" stroke-width="1"></rect><rect x="574" y="20" width="19" height="30" rx="3" fill="rgba(245,166,35,0.25)" stroke="#f5a623" stroke-width="1"></rect><rect x="596" y="20" width="19" height="30" rx="3" fill="rgba(245,166,35,0.25)" stroke="#f5a623" stroke-width="1"></rect><rect x="618" y="20" width="19" height="30" rx="3" fill="rgba(245,166,35,0.25)" stroke="#f5a623" stroke-width="1"></rect><rect x="640" y="20" width="19" height="30" rx="3" fill="rgba(245,166,35,0.25)" stroke="#f5a623" stroke-width="1"></rect><rect x="662" y="20" width="19" height="30" rx="3" fill="rgba(245,166,35,0.25)" stroke="#f5a623" stroke-width="1"></rect><rect x="684" y="20" width="19" height="30" rx="3" fill="rgba(245,166,35,0.25)" stroke="#f5a623" stroke-width="1"></rect><rect x="706" y="20" width="19" height="30" rx="3" fill="rgba(245,166,35,0.25)" stroke="#f5a623" stroke-width="1"></rect><rect x="728" y="20" width="19" height="30" rx="3" fill="rgba(245,166,35,0.25)" stroke="#f5a623" stroke-width="1"></rect><rect x="750" y="20" width="19" height="30" rx="3" fill="rgba(245,166,35,0.25)" stroke="#f5a623" stroke-width="1"></rect><rect x="772" y="20" width="19" height="30" rx="3" fill="rgba(245,166,35,0.25)" stroke="#f5a623" stroke-width="1"></rect><text x="806" y="41" font-size="14" fill="#8892a4">max ≈ 3.4e38</text><text x="70" y="103" text-anchor="end" font-size="17" fill="#e8eaf0">fp16</text><rect x="90" y="82" width="19" height="30" rx="3" fill="rgba(136,146,164,0.30)" stroke="#8892a4" stroke-width="1"></rect><rect x="112" y="82" width="19" height="30" rx="3" fill="rgba(74,158,255,0.30)" stroke="#4a9eff" stroke-width="1"></rect><rect x="134" y="82" width="19" height="30" rx="3" fill="rgba(74,158,255,0.30)" stroke="#4a9eff" stroke-width="1"></rect><rect x="156" y="82" width="19" height="30" rx="3" fill="rgba(74,158,255,0.30)" stroke="#4a9eff" stroke-width="1"></rect><rect x="178" y="82" width="19" height="30" rx="3" fill="rgba(74,158,255,0.30)" stroke="#4a9eff" stroke-width="1"></rect><rect x="200" y="82" width="19" height="30" rx="3" fill="rgba(74,158,255,0.30)" stroke="#4a9eff" stroke-width="1"></rect><rect x="222" y="82" width="19" height="30" rx="3" fill="rgba(245,166,35,0.25)" stroke="#f5a623" stroke-width="1"></rect><rect x="244" y="82" width="19" height="30" rx="3" fill="rgba(245,166,35,0.25)" stroke="#f5a623" stroke-width="1"></rect><rect x="266" y="82" width="19" height="30" rx="3" fill="rgba(245,166,35,0.25)" stroke="#f5a623" stroke-width="1"></rect><rect x="288" y="82" width="19" height="30" rx="3" fill="rgba(245,166,35,0.25)" stroke="#f5a623" stroke-width="1"></rect><rect x="310" y="82" width="19" height="30" rx="3" fill="rgba(245,166,35,0.25)" stroke="#f5a623" stroke-width="1"></rect><rect x="332" y="82" width="19" height="30" rx="3" fill="rgba(245,166,35,0.25)" stroke="#f5a623" stroke-width="1"></rect><rect x="354" y="82" width="19" height="30" rx="3" fill="rgba(245,166,35,0.25)" stroke="#f5a623" stroke-width="1"></rect><rect x="376" y="82" width="19" height="30" rx="3" fill="rgba(245,166,35,0.25)" stroke="#f5a623" stroke-width="1"></rect><rect x="398" y="82" width="19" height="30" rx="3" fill="rgba(245,166,35,0.25)" stroke="#f5a623" stroke-width="1"></rect><rect x="420" y="82" width="19" height="30" rx="3" fill="rgba(245,166,35,0.25)" stroke="#f5a623" stroke-width="1"></rect><text x="454" y="103" font-size="14" fill="#8892a4">max 65,504</text><text x="70" y="165" text-anchor="end" font-size="17" fill="#e8eaf0">bf16</text><rect x="90" y="144" width="19" height="30" rx="3" fill="rgba(136,146,164,0.30)" stroke="#8892a4" stroke-width="1"></rect><rect x="112" y="144" width="19" height="30" rx="3" fill="rgba(74,158,255,0.30)" stroke="#4a9eff" stroke-width="1"></rect><rect x="134" y="144" width="19" height="30" rx="3" fill="rgba(74,158,255,0.30)" stroke="#4a9eff" stroke-width="1"></rect><rect x="156" y="144" width="19" height="30" rx="3" fill="rgba(74,158,255,0.30)" stroke="#4a9eff" stroke-width="1"></rect><rect x="178" y="144" width="19" height="30" rx="3" fill="rgba(74,158,255,0.30)" stroke="#4a9eff" stroke-width="1"></rect><rect x="200" y="144" width="19" height="30" rx="3" fill="rgba(74,158,255,0.30)" stroke="#4a9eff" stroke-width="1"></rect><rect x="222" y="144" width="19" height="30" rx="3" fill="rgba(74,158,255,0.30)" stroke="#4a9eff" stroke-width="1"></rect><rect x="244" y="144" width="19" height="30" rx="3" fill="rgba(74,158,255,0.30)" stroke="#4a9eff" stroke-width="1"></rect><rect x="266" y="144" width="19" height="30" rx="3" fill="rgba(74,158,255,0.30)" stroke="#4a9eff" stroke-width="1"></rect><rect x="288" y="144" width="19" height="30" rx="3" fill="rgba(245,166,35,0.25)" stroke="#f5a623" stroke-width="1"></rect><rect x="310" y="144" width="19" height="30" rx="3" fill="rgba(245,166,35,0.25)" stroke="#f5a623" stroke-width="1"></rect><rect x="332" y="144" width="19" height="30" rx="3" fill="rgba(245,166,35,0.25)" stroke="#f5a623" stroke-width="1"></rect><rect x="354" y="144" width="19" height="30" rx="3" fill="rgba(245,166,35,0.25)" stroke="#f5a623" stroke-width="1"></rect><rect x="376" y="144" width="19" height="30" rx="3" fill="rgba(245,166,35,0.25)" stroke="#f5a623" stroke-width="1"></rect><rect x="398" y="144" width="19" height="30" rx="3" fill="rgba(245,166,35,0.25)" stroke="#f5a623" stroke-width="1"></rect><rect x="420" y="144" width="19" height="30" rx="3" fill="rgba(245,166,35,0.25)" stroke="#f5a623" stroke-width="1"></rect><text x="454" y="165" font-size="14" fill="#8892a4">max ≈ 3.4e38</text><rect x="90" y="192" width="14" height="14" rx="2" fill="rgba(136,146,164,0.30)" stroke="#8892a4"></rect><text x="112" y="204" font-size="14" fill="#e8eaf0">sign</text><rect x="180" y="192" width="14" height="14" rx="2" fill="rgba(74,158,255,0.30)" stroke="#4a9eff"></rect><text x="202" y="204" font-size="14" fill="#e8eaf0">exponent: sets the range</text><rect x="420" y="192" width="14" height="14" rx="2" fill="rgba(245,166,35,0.25)" stroke="#f5a623"></rect><text x="442" y="204" font-size="14" fill="#e8eaf0">mantissa: sets the precision</text></svg>
</div>

:::columns cols="2" gap="34px"
- Half the bits = half the memory and bandwidth per number
- GPU tensor cores are built for 16-bit math. An A100 does 312 TFLOPS in bf16 against 19.5 in plain fp32
- fp16 has a small range: tiny gradients underflow to zero unless the loss is scaled up first
+++
- **bf16** keeps fp32's 8 exponent bits, so it has the same range and needs no loss scaling
- The price is precision: 7 mantissa bits. In bf16, $1 + 0.001$ rounds back to $1$
- A small weight update can vanish entirely, so the weights themselves cannot live in bf16
:::

---

<!-- .slide: id="mixed-precision" -->

## Mixed Precision: In the Loop

:::columns grid="1fr 1fr" gap="34px"
```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```
+++
**"Mixed" means two precisions at once**

- Inside `autocast`, matrix multiplies run in bf16: the forward and backward passes, where nearly all the compute is
- The **master weights** and AdamW's running averages stay in fp32, so small updates are not rounded away
- A few sensitive ops (softmax, layer norm, the loss) also stay in fp32
:::

:::note
Pushing further: DeepSeek-V3 (2024) ran most matmuls in FP8. **Quantization-aware training (QAT)** simulates low-bit weights (e.g. 4-bit) during training so the model still works after it is quantized for serving (Module 10).
:::

---

<!-- .slide: id="loss-scaling" -->

## Why fp16 Needs Loss Scaling

<div class="stage-flow"><svg viewBox="-40 0 940 270" role="img" aria-label="fp16 number line: raw gradients below 6e-8 round to zero; multiplying by 65,536 shifts every gradient into the range fp16 can store, below the overflow limit of 65,504"><defs><marker id="lsa" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#8892a4"></path></marker></defs><rect x="60.0" y="30" width="123.5" height="180" fill="rgba(231,76,60,0.12)"></rect><rect x="718.5" y="30" width="141.5" height="180" fill="rgba(231,76,60,0.12)"></rect><text x="121.7" y="22" text-anchor="middle" font-size="14" fill="#e74c3c">rounds to 0</text><text x="451.0" y="22" text-anchor="middle" font-size="14" fill="#e8eaf0">fp16 can store these</text><text x="789.3" y="22" text-anchor="middle" font-size="14" fill="#e74c3c">overflow: inf</text><text x="52" y="84" text-anchor="end" font-size="14" fill="#8892a4">raw</text><text x="52" y="174" text-anchor="end" font-size="14" fill="#8892a4">× 65,536</text><line x1="82.2" y1="88" x2="296.3" y2="160" stroke="#8892a4" stroke-width="1.2" stroke-dasharray="3 3" marker-end="url(#lsa)"></line><circle cx="82.2" cy="80" r="7" fill="#e74c3c"></circle><circle cx="296.3" cy="170" r="7" fill="#3fb950"></circle><line x1="122.2" y1="88" x2="336.3" y2="160" stroke="#8892a4" stroke-width="1.2" stroke-dasharray="3 3" marker-end="url(#lsa)"></line><circle cx="122.2" cy="80" r="7" fill="#e74c3c"></circle><circle cx="336.3" cy="170" r="7" fill="#3fb950"></circle><line x1="157.8" y1="88" x2="371.8" y2="160" stroke="#8892a4" stroke-width="1.2" stroke-dasharray="3 3" marker-end="url(#lsa)"></line><circle cx="157.8" cy="80" r="7" fill="#e74c3c"></circle><circle cx="371.8" cy="170" r="7" fill="#3fb950"></circle><line x1="197.8" y1="88" x2="411.8" y2="160" stroke="#8892a4" stroke-width="1.2" stroke-dasharray="3 3" marker-end="url(#lsa)"></line><circle cx="197.8" cy="80" r="7" fill="#4a9eff"></circle><circle cx="411.8" cy="170" r="7" fill="#3fb950"></circle><line x1="233.3" y1="88" x2="447.4" y2="160" stroke="#8892a4" stroke-width="1.2" stroke-dasharray="3 3" marker-end="url(#lsa)"></line><circle cx="233.3" cy="80" r="7" fill="#4a9eff"></circle><circle cx="447.4" cy="170" r="7" fill="#3fb950"></circle><line x1="264.4" y1="88" x2="478.5" y2="160" stroke="#8892a4" stroke-width="1.2" stroke-dasharray="3 3" marker-end="url(#lsa)"></line><circle cx="264.4" cy="80" r="7" fill="#4a9eff"></circle><circle cx="478.5" cy="170" r="7" fill="#3fb950"></circle><line x1="295.6" y1="88" x2="509.6" y2="160" stroke="#8892a4" stroke-width="1.2" stroke-dasharray="3 3" marker-end="url(#lsa)"></line><circle cx="295.6" cy="80" r="7" fill="#4a9eff"></circle><circle cx="509.6" cy="170" r="7" fill="#3fb950"></circle><line x1="60" y1="215" x2="860.0" y2="215" stroke="#2a3450" stroke-width="1.5"></line><line x1="148.9" y1="210" x2="148.9" y2="220" stroke="#8892a4"></line><text x="148.9" y="238" text-anchor="middle" font-size="13" fill="#8892a4">1e-8</text><line x1="326.7" y1="210" x2="326.7" y2="220" stroke="#8892a4"></line><text x="326.7" y="238" text-anchor="middle" font-size="13" fill="#8892a4">1e-4</text><line x1="504.4" y1="210" x2="504.4" y2="220" stroke="#8892a4"></line><text x="504.4" y="238" text-anchor="middle" font-size="13" fill="#8892a4">1</text><line x1="682.2" y1="210" x2="682.2" y2="220" stroke="#8892a4"></line><text x="682.2" y="238" text-anchor="middle" font-size="13" fill="#8892a4">1e4</text><line x1="860.0" y1="210" x2="860.0" y2="220" stroke="#8892a4"></line><text x="860.0" y="238" text-anchor="middle" font-size="13" fill="#8892a4">1e8</text><text x="183.5" y="258" text-anchor="middle" font-size="12" fill="#e74c3c">6e-8</text><text x="718.5" y="258" text-anchor="middle" font-size="12" fill="#e74c3c">65,504</text></svg></div>

:::columns cols="3" gap="24px"
**The problem**

Many gradients are smaller than fp16's smallest number, so they round to 0 and the update is lost
+++
**The fix**

Multiply the loss by a large $S$. Every gradient grows by $S$ too. Divide by $S$ again before the update
+++
**Keeping it safe**

If any gradient overflows past 65,504, skip the step and halve $S$
:::

:::note
fp16 needs this; bf16 does not, because it has fp32's range. That is why bf16 replaced fp16 as the default for LLM training.
:::

---

<!-- .slide: id="runs-are-messy" -->

## Real Runs Are Messy

:::columns cols="2" gap="34px"
**Spikes and divergence**

- Long runs can spike or diverge
- Usual suspects: learning rate too high, weak clipping, batch size, a bad data shard
+++
**Loss is not the whole story**

- Lower loss does not perfectly predict every capability
- Real runs also track **downstream benchmarks** at checkpoints
- Generated samples build intuition, but are not a metric
:::

---

<!-- .slide: id="bloom-176b-log" -->

## A Real Run: BLOOM 176B

:::columns grid="2.3fr 1fr" gap="28px" valign="center"
<div class="run-figure">
  <img src="images/bloom176b_training.png" alt="BLOOM 176B training loss falling from about 4.4 to 1.93 over 367 billion tokens, with a loss spike to 5.1 at step 31,219 and grad-norm spikes up to 960">
</div>
+++
- Mostly a smooth curve: loss falls to 1.93 over 367B tokens
- At step 31,219 one batch sent the gradient norm to 960 (typical: 0.14) and the loss to 5.1. It recovered within about 80 steps
- The job restarted from an earlier checkpoint 50 times over four months
:::

---

<!-- .slide: id="bloom-104b-log" -->

## A Run That Diverged: BigScience 104B

:::columns grid="2.3fr 1fr" gap="28px" valign="center"
<div class="run-figure">
  <img src="images/bloom104b_spikes.png" alt="Six attempts of the BigScience 104B prototype: loss jumps from 3.4 to above 7, losses become NaN, and the fp16 loss scale collapses to 1; restarts from earlier checkpoints diverge again">
</div>
+++
- Each colored line is a restart from an earlier checkpoint. Every one that got past step 8,700 diverged
- **Bottom:** the loss scale $S$. Each blow-up drove it to 1: the gradients overflowed fp16 even unscaled
- BLOOM 176B switched to bf16, which has fp32's range
:::

---

<!-- .slide: id="side-quest-llama3-noon" -->

## Side Quest: Llama 3 Trained Slower at Noon

Meta's log of a 54-day stretch of Llama 3 405B pretraining on 16K H100 GPUs:

:::columns cols="2" gap="34px"
**Things broke constantly**

- 466 job interruptions: 47 planned, 419 unexpected. Roughly one every three hours
- About 78% traced to hardware. Faulty GPUs and their HBM3 memory topped the list
- Automation restarted the job; only 3 failures needed serious human help
+++
**And some were just strange**

- Throughput rose and fell 1 to 2% with the time of day: midday heat made the GPUs lower their clock speeds
- When every GPU pauses at once, say for a checkpoint, the data center's power draw swings by tens of megawatts, straining the power grid
:::

Even so, over 90% of the time went to actual training.

---

<!-- .slide: id="overfit-check" -->

## Sanity Check: Overfit One Batch

Before a long run, train repeatedly on a **single batch**.

:::columns cols="2" gap="34px"
**Why it works**

- Any model with enough capacity can memorize one tiny batch
- Its loss should crater toward zero
+++
**What it catches**

- Loss does not crater = broken loop
- Detached gradient, wrong target shift, frozen parameter, bad learning rate
- Cheap to run, fails loudly
:::

The exercise runs exactly this check.
