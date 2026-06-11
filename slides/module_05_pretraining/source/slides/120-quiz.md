:::divider id="divider-quiz" title="Quiz"
:::

---

:::quiz id="quiz-one-loss-per-token" title="One Loss Per Token"
Causal language modeling produces one loss term per **token**, not one per sequence. Why?
+++
Because of causal masking, every position simultaneously makes a next-token prediction from only the tokens before it. A sequence of length $T$ is therefore $T$ separate prediction problems at once, each contributing its own cross-entropy term. This dense signal &mdash; a gradient from every position &mdash; is a major reason causal LM is so sample-efficient compared to masked LM, which only learns from the ~15% of positions it masks.
:::

---

:::quiz id="quiz-base-instruction" title="Base Models and Instructions"
Why can a base model **continue** an instruction without reliably **obeying** it?
+++
A base model is trained only to predict the most likely continuation of text in its training distribution. An instruction like "List three colors" is, statistically, just as likely to be followed by more instructions (as in a worksheet) as by an answer. The base model has no objective that says "satisfy the request" &mdash; it matches patterns. Teaching it to treat a prompt as something to fulfill is the job of instruction finetuning (Module 6).
:::

---

:::quiz id="quiz-val-loss" title="Why Validation Loss"
What does **validation** loss catch that training loss can hide?
+++
Training loss can keep falling because the model is **memorizing** the exact training batches rather than learning transferable structure. Validation loss is measured on held-out text the model never updated on, so it only improves when the model learns patterns that **generalize**. A training loss that keeps dropping while validation loss flattens or rises is the classic signature of overfitting or memorization.
:::

---

:::quiz id="quiz-dedup" title="Deduplication Removes Tokens, Yet Helps"
Deduplication throws away training tokens. Why can it still **improve** generalization?
+++
Duplicated passages bias the objective toward memorizing those exact strings: the model spends capacity reproducing text it has seen many times instead of learning reusable patterns. Removing duplicates rebalances the data toward diversity, so the same gradient steps teach more general structure. The tokens lost were largely redundant; the patterns gained transfer. (It also reduces verbatim memorization of sensitive or copyrighted text.)
:::

---

:::quiz id="quiz-chinchilla" title="Smaller Can Beat Bigger"
Under a **fixed compute budget**, why might a smaller model trained on more tokens beat a larger model trained on fewer?
+++
Compute is roughly $C \approx 6ND$, so for a fixed $C$ the parameters $N$ and tokens $D$ trade off directly. A very large model exhausts the budget after seeing too few tokens and never learns enough; a smaller model can afford many more tokens within the same budget. Chinchilla showed many early models were oversized for their token counts, and that roughly 20 tokens per parameter is compute-optimal &mdash; so the smaller, better-fed model wins.
:::

---

:::quiz id="quiz-units" title="Loss, Perplexity, Bits"
Two models report the same loss in **nats**. How do you convert that loss into perplexity and into bits per token?
+++
Perplexity is $\exp(\text{loss})$ &mdash; the exponential of the loss in nats, read as the effective number of equally likely next-token choices. Bits per token is $\text{loss} / \ln 2$ &mdash; the same loss converted from nats to bits (since $1$ nat $= 1/\ln 2$ bits). All three are the same quantity in different units, which is why a single training run can report whichever is most convenient.
:::

---

:::quiz id="quiz-warmup" title="Why Warmup Helps"
Why does learning-rate **warmup** reduce the chance of early training instability?
+++
At initialization the weights are random, so the first gradients can be large and poorly conditioned, and Adam's running averages of the gradient and its square have not yet stabilized. Taking full-size steps immediately can throw the weights into a bad region and spike the loss. Ramping the learning rate up slowly lets the optimizer statistics settle and keeps the first updates small while the model is most fragile.
:::

---

:::quiz id="quiz-parallelism" title="Two Kinds of Parallelism"
What is the difference between **data parallelism** and **model / tensor / pipeline** parallelism?
+++
Data parallelism replicates the **whole model** on each GPU and splits the **batch** across them, then all-reduces the gradients so every replica stays identical &mdash; it scales throughput when the model fits on one device. Model, tensor, and pipeline parallelism instead split the **model itself** &mdash; individual weight matrices (tensor) or whole layers (pipeline) &mdash; across GPUs, which is necessary when the parameters or activations are too large to fit on a single device. Large runs combine both.
:::

---

:::quiz id="quiz-emergence" title="Emergence or Mirage?"
Why might an apparent **emergent ability** be a measurement artifact rather than a sudden new capability?
+++
"Emergence" is often measured with a harsh metric &mdash; for example exact-match accuracy, which scores a task as 0 until every step is correct. A model improving smoothly in its underlying probabilities can stay at near-zero on such a metric for a long time, then appear to "switch on" once it crosses the threshold. Under a smoother, continuous metric the same capability frequently rises gradually. So a discontinuous-looking curve may reflect the metric, not a real phase change in the model.
:::
