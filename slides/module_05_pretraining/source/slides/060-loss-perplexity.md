:::divider id="divider-loss" title="Reading the Loss" sub="Loss, perplexity, and what progress looks like"
:::

---

<!-- .slide: id="loss-intro" -->

## Cross-Entropy Measures Surprise

Loss = the model's average **surprise** at the true next token. Lower loss = more probability on what actually came next.

The same number wears three hats:

- **loss** in nats, the raw objective
- **perplexity** $= \exp(\text{loss})$, the effective number of next-token choices
- **bits per token** $= \text{loss} / \ln 2$, the compression view from Module 1

:::note
A **nat** is the natural-log sibling of the bit: $\ln$ instead of $\log_2$. PyTorch's `log` is natural, so raw loss comes out in nats. One nat $= 1/\ln 2 \approx 1.44$ bits.
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

- Both curves should fall
- A **widening gap** = memorizing the training set, not learning reusable patterns
- This curve is real output from the Module 5 exercise

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

<!-- .slide: id="before-after-demo" -->

## The Demo: Before vs After

Same model, before and after training. Both samples are real output from the exercise's tiny character-level model.

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

- Still nonsense up close, but it learned the **shape**: names in caps, colons, line breaks, plausible letters
- Perplexity: ~65 to ~5. Bits per token: ~6.0 to ~2.4
