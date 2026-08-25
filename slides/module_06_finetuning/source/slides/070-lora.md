:::divider id="divider-lora" title="Parameter-Efficient Finetuning" sub="LoRA and the adapter family"
:::

---

<!-- .slide: id="full-ft-cost" -->

## The Problem With Full Finetuning

Updating **all** parameters is expensive three ways:

:::columns cols="3" gap="22px"
**Optimizer state**

- AdamW: two moment buffers per parameter
- Roughly **3x** model memory just to train (Module 5)
+++
**A full copy per task**

- Every finetune is a full-size checkpoint
- Ten tasks: ten copies of a 70B model
+++
**Hardware**

- Model + gradients + optimizer state
- Needs a cluster, not a laptop
:::

---

<!-- .slide: id="lora-insight" -->

## The LoRA Insight

Hu et al. (2021): the finetuning **update** to a weight matrix has **low intrinsic rank**. So freeze $W$ and learn a low-rank correction:

$$W' = W + \frac{\alpha}{r} BA, \qquad B \in \mathbb{R}^{d_{\text{out}} \times r}, \quad A \in \mathbb{R}^{r \times d_{\text{in}}}, \quad r \ll d_{\text{in}}, d_{\text{out}}$$

$BA$ has exactly $W$'s shape: $d_{\text{out}} \times d_{\text{in}}$. <!-- .element: class="text-lg" -->

:::columns cols="2" gap="34px"
- Only $A$ and $B$ train &mdash; often **under 1%** of parameters
- The frozen base $W$ is **shared** across tasks
- $B$ starts at **zero**: the adapter begins as a no-op
+++
- $\alpha/r$ is fixed, so changing $r$ needs no learning-rate retune
- Optimizer state is tiny: moments for $A$ and $B$ only
- Less perturbation of the base: **less catastrophic forgetting**
:::

---

:::interactive id="lora-calculator" widget="loraCalculator" title="What Rank Buys You"
:::

---

:::manim id="lora-anim" scene="lora"
:::

---

<!-- .slide: id="lora-merge-swap" -->

## Two Key Properties

:::columns cols="2" gap="34px"
**Merge: zero added latency**

- Compute $W + \frac{\alpha}{r}BA$ once, store the result
- Same shape as $W$: merged model runs at **exactly** base speed
+++
**Swap: many behaviors, one base**

- Keep adapters separate over one frozen base
- Load a different megabyte-sized adapter to switch behaviors
:::

Checkpoints shrink from **gigabytes to megabytes**. Finetuning fits on **consumer hardware**. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="qlora-peft" -->

## QLoRA and the PEFT Family

:::columns cols="2" gap="34px"
**QLoRA** (Dettmers et al., 2023)

- Quantize the frozen base to **4-bit**, train LoRA adapters on top
- A **65B** model finetunes on a **single GPU**
- Quantization itself: Module 9
+++
**The broader PEFT family**

- **Adapters** (Houlsby et al.): small trainable layers inserted between frozen ones
- **Prefix / prompt tuning**: learn soft tokens, leave weights frozen
- **(IA)^3**: learn to rescale activations
:::

**Trade-off:** PEFT is slightly less expressive than full finetuning. For most instruction tuning the gap is small and the savings are large. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="figure-hu" -->

:::figure img="images/hu.jpg" name="Edward Hu" kicker="LoRA: Low-Rank Adaptation of Large Language Models (Microsoft, 2021)"
- Introduced **low-rank adaptation**, making finetuning of huge models cheap and modular
- Showed a tiny low-rank update can match full finetuning quality on many tasks
- Turned task-specific finetunes into **megabyte-sized** adapters over one shared base
:::

---

<!-- .slide: id="figure-dettmers" -->

:::figure img="images/dettmers.jpg" name="Tim Dettmers" kicker="QLoRA: Efficient Finetuning of Quantized LLMs (2023)"
- Drove practical **quantization** and **QLoRA**, combining 4-bit base weights with LoRA adapters
- Put finetuning of very large models on a **single consumer GPU**
- Made finetuning possible without a datacenter
:::

---

<!-- .slide: id="side-quest-adapters-as-diffs" -->

## Side Quest: Adapters as Diffs for Weights

A LoRA adapter is a **diff** against the frozen base: a code patch for model weights.

:::columns cols="2" gap="34px"
- Task-specific behaviors as **megabyte-sized files** you load and unload
- Ship a 3 MB adapter, not a 14 GB model
+++
- Model hubs host thousands of community adapters per popular base
- Adapters can be **composed** or averaged, like merging branches
:::
