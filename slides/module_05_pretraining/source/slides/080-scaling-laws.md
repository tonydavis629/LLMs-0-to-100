:::divider id="divider-scaling" title="Scaling Laws" sub="Predicting performance before spending the compute"
:::

---

<!-- .slide: id="scaling-predictable" -->

## Loss Improves Predictably

Kaplan, McCandlish, and collaborators (2020): loss falls as a smooth **power law** in three quantities, across many orders of magnitude:

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

- On log-log axes, loss vs compute is nearly a straight line
- So small-scale runs predict a large model's loss **before** committing the budget

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

- Forward pass: ~2 FLOPs per parameter per token
- Backward pass: about twice that
+++
**Why it is useful**

- "How big, how long?" becomes arithmetic
- Fix $C$, and $N$ and $D$ trade off directly
:::

---

<!-- .slide: id="chinchilla" -->

## Chinchilla: Compute-Optimal Balance

Hoffmann and the Chinchilla team (2022): for a **fixed compute budget**, many earlier models were too large for their token counts.

:::columns cols="2" gap="34px"
**The lesson**

- Parameters and tokens should grow **together**
- A smaller model on more tokens beats a larger model starved of data
+++
**The rule of thumb**

- ~**20 training tokens per parameter** is compute-optimal
- A 1B-parameter model wants ~20B tokens
:::

---

:::interactive id="scaling-planner" widget="scalingPlanner" title="Spend One Compute Budget"
:::

---

<!-- .slide: id="serving-optimal" -->

## Compute-Optimal Is Not Serving-Optimal

Chinchilla optimizes **training**. A model is trained once, then served billions of times.

- Constant serving means inference cost dominates, so **smaller** wins
- Llama-style models train **smaller models on far more tokens** than Chinchilla suggests
- Extra training compute buys much larger inference savings

:::note
Model size, data size, batch size, sequence length, hardware, and wall-clock time all interact; pretraining is engineering as much as modeling. Brown and the GPT-3 team (2020) had already shown the payoff of scale: strong **few-shot** behavior through prompting.
:::

---

<!-- .slide: id="side-quest-emergence" -->

## Side Quest: Emergent Abilities, Real or Mirage?

:::columns cols="2" gap="34px"
**Sharp emergence (Wei et al., 2022)**

- Some capabilities appear to switch on **suddenly** past a scale threshold
- Near-zero performance, then a jump
+++
**A measurement artifact? (Schaeffer et al., 2023)**

- The jump can come from a **thresholded or nonlinear metric**
- Under a smoother metric, the same capability often improves gradually
:::

A discontinuous-looking curve may describe the metric, not the model.
