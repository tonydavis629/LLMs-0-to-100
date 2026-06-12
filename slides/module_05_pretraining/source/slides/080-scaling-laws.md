:::divider id="divider-scaling" title="Scaling Laws" sub="Predicting performance before spending the compute"
:::

---

<!-- .slide: id="scaling-predictable" -->

## Loss Improves Predictably

Kaplan, McCandlish, and collaborators (2020) found that language-model loss falls as a smooth **power law** in three quantities, each spanning many orders of magnitude:

:::columns cols="3" gap="20px"
**Parameters** $N$

a bigger model
+++
**Data** $D$

more training tokens
+++
**Compute** $C$

more total FLOPs
:::

On log-log axes the loss-versus-compute curve is nearly a straight line. That predictability is powerful: you can estimate a large model's loss from small-scale runs, **before** committing the budget.

---

:::manim id="scaling-anim" scene="scaling-laws"
:::

---

<!-- .slide: id="compute-6nd" -->

## A Computable Handle: C &approx; 6ND

For a dense transformer, total pretraining compute is well approximated by

$$C \approx 6 N D$$

where $N$ is the parameter count and $D$ is the number of training tokens (Kaplan et al., 2020).

:::columns cols="2" gap="34px"
**Where the 6 comes from**

Roughly 2 FLOPs per parameter per token for the forward pass, and about twice that for the backward pass &mdash; about 6 in total.
+++
**Why it is useful**

It turns "how big, how long?" into arithmetic. Fix a compute budget $C$, and $N$ and $D$ trade off directly against each other.
:::

---

<!-- .slide: id="chinchilla" -->

## Chinchilla: Compute-Optimal Balance

The original scaling story encouraged ever-**larger** models. Hoffmann and the Chinchilla team (2022) refined it: for a **fixed compute budget**, many earlier models were too large for the number of tokens they were trained on.

:::columns cols="2" gap="34px"
**The lesson**

Parameters and tokens should grow **together**. A smaller model trained on more tokens can beat a larger model starved of data.
+++
**The rule of thumb**

About **20 training tokens per parameter** is compute-optimal. A 1B-parameter model wants roughly 20B tokens.
:::

---

<!-- .slide: id="serving-optimal" -->

## Compute-Optimal Is Not Serving-Optimal

Chinchilla optimizes the **training** budget. But a model is trained once and then served billions of times.

- A model you will run constantly should be **cheap at inference**, which means **smaller**
- So Llama-style models deliberately train **smaller models on far more tokens** than Chinchilla suggests &mdash; spending extra training compute to save much more inference compute later
- Compute-optimal training balances parameters, training tokens, **and** deployment cost &mdash; not parameter count alone

:::note
This is why pretraining is an engineering problem as much as a modeling one: model size, data size, batch size, sequence length, hardware, and wall-clock time all interact. Brown and the GPT-3 team (2020) had already shown the payoff of scale &mdash; large-scale autoregressive pretraining producing strong **few-shot** behavior through prompting.
:::

---

<!-- .slide: id="side-quest-emergence" -->

## Side Quest: Emergent Abilities, Real or Mirage?

:::columns cols="2" gap="34px"
**Sharp emergence (Wei et al., 2022)**

Some capabilities appear to switch on **suddenly** past a scale threshold &mdash; near-zero performance, then a jump.
+++
**A measurement artifact? (Schaeffer et al., 2023)**

The jump can be an artifact of a **thresholded or nonlinear metric**. Under a smoother metric, the same capability often improves gradually.
:::

The critical-thinking takeaway: a curve that looks discontinuous may say more about how you measured it than about the model. Always ask what the metric is doing.
