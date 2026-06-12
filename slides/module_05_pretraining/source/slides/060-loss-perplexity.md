:::divider id="divider-loss" title="Reading the Loss" sub="Loss, perplexity, and what progress looks like"
:::

---

<!-- .slide: id="loss-intro" -->

## Cross-Entropy Measures Surprise

The loss is the model's average **surprise** at the true next token. Lower loss means the model assigned more probability to what actually came next &mdash; it is less surprised by held-out text.

The same number wears three hats:

- **loss** in nats, the raw objective
- **perplexity** $= \exp(\text{loss})$, the effective number of next-token choices
- **bits per token** $= \text{loss} / \ln 2$, the compression view from Module 1

:::note
A **nat** is the natural-log sibling of the bit: the bit measures surprise with $\log_2$, the nat with $\ln$. Cross-entropy uses $\ln$ because that is what calculus and PyTorch's `log` give you, so the raw loss comes out in nats. One nat $= 1/\ln 2 \approx 1.44$ bits &mdash; same quantity, different base.
:::

---

:::manim id="perplexity-anim" scene="perplexity"
:::

---

<!-- .slide: id="progress-curves" -->

## What Progress Looks Like

<div class="loss-figure">
  <img src="images/loss_curve.png" alt="Training and validation loss falling over 2000 steps">
</div>

Training loss should fall; validation loss should fall too. A **widening gap** between them is the warning sign &mdash; the model is starting to memorize the training set instead of learning reusable patterns. (This curve is the actual output of the Module 5 exercise.)

---

<!-- .slide: id="runs-are-messy" -->

## Real Runs Are Messy

Loss does not always glide down smoothly:

:::columns cols="2" gap="34px"
**Spikes and divergence**

A long run can spike or diverge outright when the recipe is unstable. Usual suspects: learning rate too high, weak gradient clipping, batch size, or a bad shard of data.
+++
**Loss is not the whole story**

Lower next-token loss does not perfectly predict every capability. Real runs also track **downstream benchmarks** at checkpoints. Generated samples are useful for intuition but are not a reliable metric on their own.
:::

---

<!-- .slide: id="before-after-demo" -->

## The Demo: Before vs After

The clearest way to feel pretraining work: sample from the **same** model before and after training. Both samples below are real output from the exercise's tiny character-level model.

:::columns cols="2" gap="30px"
**Before** (random weights, loss 4.18)

```text
-pzlYaS ;czdeCpwEiT,YzrzlG3-aYeNB
ijbo
Lzzj$KUKS-A.U FisdJ'G HTobPPW;,Ue$
```
+++
**After** (2000 steps, val loss 1.64)

```text
FRIAR LAURENCE:
What do tongue the cLARENCE:
Your felsed hath you seed heart of me.
```
:::

It is still nonsense up close &mdash; but it learned the **shape** of the data: character names in caps, colons, line breaks, plausible letter sequences. Perplexity fell from ~65 to ~5; bits per token from ~6.0 to ~2.4.
