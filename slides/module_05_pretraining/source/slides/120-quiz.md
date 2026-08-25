:::divider id="divider-quiz" title="Quiz"
:::

---

:::quiz id="quiz-one-loss-per-token" title="One Loss Per Token"
Causal language modeling produces one loss term per **token**, not one per sequence. Why?
+++
- Causal masking lets every position predict its next token from earlier tokens only
- A sequence of length $T$ = $T$ prediction problems, each with its own cross-entropy term
- This dense signal makes causal LM sample-efficient; masked LM learns only from the ~15% it masks
:::

---

:::quiz id="quiz-base-instruction" title="Base Models and Instructions"
Why can a base model **continue** an instruction without reliably **obeying** it?
+++
- A base model predicts the most likely continuation, nothing more
- "List three colors" is statistically as likely to be followed by more instructions (a worksheet) as by an answer
- No objective says "satisfy the request"; that is instruction finetuning (Module 6)
:::

---

:::quiz id="quiz-val-loss" title="Why Validation Loss"
What does **validation** loss catch that training loss can hide?
+++
- Training loss can fall from **memorizing** exact batches
- Validation loss uses held-out text, so it only improves when patterns **generalize**
- Training loss falling while validation flattens or rises = the classic overfitting signature
:::

---

:::quiz id="quiz-dedup" title="Deduplication Removes Tokens, Yet Helps"
Deduplication throws away training tokens. Why can it still **improve** generalization?
+++
- Duplicates push the model to memorize exact strings instead of reusable patterns
- Removing them rebalances toward diversity: the same gradient steps teach more general structure
- The lost tokens were redundant; the gained patterns transfer
- Bonus: less verbatim memorization of sensitive or copyrighted text
:::

---

:::quiz id="quiz-chinchilla" title="Smaller Can Beat Bigger"
Under a **fixed compute budget**, why might a smaller model trained on more tokens beat a larger model trained on fewer?
+++
- $C \approx 6ND$: for fixed $C$, parameters $N$ and tokens $D$ trade off directly
- A too-large model exhausts the budget after too few tokens
- A smaller model affords many more tokens in the same budget
- Chinchilla: ~20 tokens per parameter is compute-optimal; the smaller, better-fed model wins
:::

---

:::quiz id="quiz-units" title="Loss, Perplexity, Bits"
How do you convert a loss in **nats** into **perplexity** and into **bits per token**?
+++
- Perplexity $= \exp(\text{loss})$: the effective number of equally likely next-token choices
- Bits per token $= \text{loss} / \ln 2$: nats converted to bits
- All three are the same quantity in different units
:::

---

:::quiz id="quiz-warmup" title="Why Warmup Helps"
Why does learning-rate **warmup** reduce the chance of early training instability?
+++
- At initialization, gradients are large and poorly conditioned
- Adam's running averages have not yet stabilized
- Full-size steps can throw the weights into a bad region and spike the loss
- Ramping up slowly keeps early updates small while the model is most fragile
:::

---

:::quiz id="quiz-parallelism" title="Two Kinds of Parallelism"
What is the difference between **data parallelism** and **model / tensor / pipeline** parallelism?
+++
- Data parallelism: replicate the **whole model** per GPU, split the **batch**, all-reduce gradients. Requires the model to fit on one device
- Model / tensor / pipeline parallelism: split the **model itself** (weight matrices or whole layers) when it does not fit
- Large runs combine both
:::

---

:::quiz id="quiz-emergence" title="Emergence or Mirage?"
Why might an apparent **emergent ability** be a measurement artifact rather than a sudden new capability?
+++
- Harsh metrics (e.g. exact match) score 0 until every step is correct
- A model improving smoothly in its probabilities stays near zero, then appears to "switch on" at a threshold
- Under a smoother metric the same capability rises gradually
- The discontinuity can live in the metric, not the model
:::
