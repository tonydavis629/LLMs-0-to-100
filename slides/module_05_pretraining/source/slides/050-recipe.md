:::divider id="divider-recipe" title="The Training Recipe" sub="The engineering that keeps a long run stable"
:::

---

<!-- .slide: id="recipe-intro" -->

## One Loop, Repeated

- Module 2 gave the core: SGD, Adam, mini-batches, learning rates
- Pretraining adds a few standard choices that keep **long, large** runs stable
- The loop never changes: forward, loss, backward, update &mdash; millions of times

---

:::manim id="training-loop-anim" scene="training-loop"
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

<!-- .slide: id="stability-tricks" -->

## Stability and Scale

:::columns cols="3" gap="20px"
**Gradient clipping**

- Cap the global gradient norm
- One bad batch cannot wreck the weights
- Prevents loss spikes
+++
**Gradient accumulation**

- Sum gradients over several small batches before stepping
- Simulates a larger **effective** batch than one GPU holds
+++
**Mixed precision**

- Most math in `bf16`, not `fp32`
- Fewer bits = faster compute, larger models in memory
- More in Module 9
:::

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
