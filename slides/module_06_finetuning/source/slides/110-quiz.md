:::divider id="divider-quiz" title="Quiz"
:::

---

:::quiz id="quiz-small-lr" title="Why a Smaller Learning Rate?"
Why does finetuning use a **smaller learning rate** and **far fewer steps** than pretraining?
+++
- Pretraining already found a good region of weight space
- Finetuning only **nudges** behavior; it does not learn language from scratch
- Large steps or many epochs on the small, skewed set overwrite broad features: **catastrophic forgetting**
- A small learning rate and 1&ndash;3 epochs teach format while preserving capability
:::

---

:::quiz id="quiz-why-mask" title="Why Mask the Prompt?"
Why do we mask the **prompt** tokens out of the SFT loss instead of training on the whole sequence?
+++
- We want the model to **answer** prompts, not **write** them
- Loss on prompt positions trains it to predict user tokens; at generation time it invents its own questions
- Masking (targets = -100) rewards only the **response**
- The prompt stays as context the model reads; it is just not a target
:::

---

:::quiz id="quiz-lora-params" title="How LoRA Saves Parameters"
How does LoRA reduce the number of **trainable** parameters without changing the model's **architecture at inference**?
+++
- LoRA freezes $W$ and trains a low-rank update $\frac{\alpha}{r}BA$ with $r$ far smaller than either dimension
- Trainable count: $r(d_{\text{in}} + d_{\text{out}})$ instead of $d_{\text{in}} d_{\text{out}}$, often under 1%
- At inference, merge $W + \frac{\alpha}{r}BA$ once; the result has the **same shape** as $W$
- Forward pass and architecture are identical; only the training-time count changed
:::

---

:::quiz id="quiz-1p3b-beats-175b" title="Smaller Beat Bigger"
Why was a **1.3B** InstructGPT model preferred over **175B** GPT-3 despite being far smaller?
+++
- Scale buys knowledge and fluency, not the **behavior** of following intent
- GPT-3 was a base model: it continues text, which often is not what the user wanted
- InstructGPT was finetuned on demonstrations and preferences to **satisfy requests**
- Raters judge usefulness, not parameter count, so the aligned model won
:::

---

:::quiz id="quiz-preference-vs-sft" title="The SFT Ceiling"
A team keeps improving their SFT model by collecting **more and higher-quality** demonstrations. Progress stalls: it still occasionally produces fluent, confident answers that are subtly **wrong**, and more demonstrations barely help. Why does this ceiling exist, and what kind of training signal is needed to push past it?
+++
- SFT learns only from **positive** examples: "produce responses like these"
- It pulls probability **toward** demonstrations, never **away** from plausible-but-wrong answers
- More demonstrations cannot teach **relative quality**, so the ceiling is the demonstrations themselves
- Pushing past it needs a **comparative** signal: preference rankings, reward modeling (Module 7)
- **Exposure bias** compounds this: trained on ground-truth prefixes, generating over its own imperfect text
:::

---

:::quiz id="quiz-forgetting" title="Forgetting and Adapters"
What is **catastrophic forgetting**, and why does freezing the base and training a small **adapter** help avoid it?
+++
- Catastrophic forgetting: finetuning on narrow data overwrites the weights that encoded general capability
- LoRA freezes the pretrained matrix; those values **cannot change**
- The adapter is low-capacity and starts at zero, so it can only make a modest adjustment
- The base model's knowledge is structurally preserved
:::

---

:::quiz id="quiz-merge-latency" title="Why Merging Adds No Latency"
After **merging** a LoRA update $BA$ into the frozen weight $W$, why is there **no extra** inference latency?
+++
- During training the layer computes $Wx + \frac{\alpha}{r}B(Ax)$: two extra multiplies
- $W + \frac{\alpha}{r}BA$ precomputes into a **single** matrix $W'$, same shape as $W$
- Merged, the layer computes $W'x$: same FLOPs, same memory as the base model
- The adapter is folded in; nothing extra runs at inference
:::
