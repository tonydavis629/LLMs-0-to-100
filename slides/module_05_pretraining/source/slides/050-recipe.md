:::divider id="divider-recipe" title="The Training Recipe" sub="The engineering that keeps a long run stable"
:::

---

<!-- .slide: id="recipe-intro" -->

## One Loop, Repeated

Module 2 already gave us the core: SGD, Adam, mini-batches, and learning rates. Real pretraining wraps the same loop in a small set of standard choices that make **long, large** runs stable and efficient.

The loop itself never changes: forward, loss, backward, update &mdash; millions of times.

---

:::manim id="training-loop-anim" scene="training-loop"
:::

---

<!-- .slide: id="adamw" -->

## AdamW: The Standard Optimizer

Pretraining almost always uses **AdamW**: Adam's per-parameter adaptive step sizes, plus **decoupled weight decay**.

:::columns cols="2" gap="34px"
**Adam part**

Keeps running averages of the gradient and its square, so each parameter gets its own effective learning rate. Robust to the wildly different gradient scales across a deep model.
+++
**Weight-decay part**

Gently pulls weights toward zero each step &mdash; the regularization idea from Module 2i. "Decoupled" means the decay is applied directly to the weights, separately from the adaptive gradient term, which works better in practice.
:::

---

<!-- .slide: id="lr-schedule" -->

## Learning-Rate Schedule: Warmup, Then Cosine Decay

A constant learning rate is rarely best. The standard schedule has two phases:

:::columns cols="2" gap="34px"
**Warmup**

Start near zero and ramp up over the first few hundred steps. Early on the weights are random and gradients are large; a small step size avoids an immediate blow-up.
+++
**Cosine decay**

After the peak, lower the learning rate smoothly along a cosine curve down to a small floor. Big steps early to explore, small steps late to settle into a good minimum.
:::

---

:::manim id="lr-anim" scene="lr-schedule"
:::

---

<!-- .slide: id="stability-tricks" -->

## Stability and Scale

Three more standard ingredients, each solving a concrete problem:

:::columns cols="3" gap="20px"
**Gradient clipping**

Cap the global gradient norm. A single bad batch can produce a huge gradient that wrecks the weights; clipping bounds the damage and prevents loss spikes.
+++
**Gradient accumulation**

Sum gradients over several small device batches before stepping. This simulates a much larger **effective** batch size than one GPU could hold at once.
+++
**Mixed precision**

Do most math in `bf16` instead of `fp32`. Fewer bits per number means faster compute and larger models in memory. (More in Module 9.)
:::

---

<!-- .slide: id="overfit-check" -->

## Sanity Check: Overfit One Batch

Before launching a long run, train repeatedly on a **single batch** until the loss gets very low.

:::columns cols="2" gap="34px"
**Why it works**

A model with enough capacity should be able to memorize one tiny batch and drive its loss toward zero. There is nothing to generalize to &mdash; it is pure memorization.
+++
**What it catches**

If the loss does **not** crater, the training loop is broken: a detached gradient, a wrong target shift, a frozen parameter, a bad learning rate. Cheap to run, and it fails loudly.
:::

You will run exactly this check in the exercise.
