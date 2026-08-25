:::divider id="divider-objective" title="The Training Objective" sub="Causal language modeling, and the alternatives"
:::

---

<!-- .slide: id="causal-lm" -->

## Causal Language Modeling

The GPT-style objective: predict token $x_t$ from the tokens before it.

The training pair is the **same sequence, shifted by one**:

:::columns cols="2" gap="30px"
**input** $x$

$$x_0,\ x_1,\ \ldots,\ x_{T-1}$$
+++
**target** $y$

$$x_1,\ x_2,\ \ldots,\ x_{T}$$
:::

- One logit vector at **every** position: $T$ predictions, $T$ loss terms per sequence
- Labels are free: just shift

---

:::manim id="next-token-anim" scene="next-token"
:::

---

<!-- .slide: id="objective-equation" -->

## The Cross-Entropy Objective

Minimize the average negative log-probability of the true next token:

$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T}\log p_\theta\left(x_t \mid x_{<t}\right)$$

This is **cross-entropy** against the true next token, averaged over every position.

:::columns cols="3" gap="20px"
**loss** (nats)

the raw objective above
+++
**perplexity**

$$\exp(\mathcal{L})$$
the effective number of choices
+++
**bits per token**

$$\mathcal{L} / \ln 2$$
Shannon's units (Module 1)
:::

Three units, one quantity. Detail in a few slides.

---

:::figure img="images/radford.jpg" name="Alec Radford" kicker="Improving Language Understanding by Generative Pre-Training (2018)"
- Led the GPT line of generative pretraining at OpenAI
- Showed that a single decoder trained only to predict the next token transfers to many tasks
- GPT-2 (2019) made large-scale causal language modeling the dominant recipe
:::

---

<!-- .slide: id="other-objectives" -->

## Other Pretraining Objectives

Two alternatives change **which tokens are predicted from which context**:

:::columns cols="2" gap="30px"
**Masked language modeling (BERT)**

- Hide ~15% of tokens; predict them from **both sides**
- Builds rich bidirectional representations
- Not a natural generator; only masked positions give signal
+++
**Denoising / span corruption (T5)**

- Corrupt a passage (drop or shuffle spans)
- Train an encoder-decoder to **reconstruct** it
- Flexible "text-to-text", but heavier machinery
:::

---

:::figure img="images/devlin.jpg" name="Devlin, Chang, Lee, and Toutanova" kicker="BERT: Pre-training of Deep Bidirectional Transformers (2018)"
- Introduced masked language modeling as the reference point for bidirectional pretraining
- Each masked position is predicted from left and right context at once
- Excellent for understanding tasks; not designed for open-ended generation
:::

---

<!-- .slide: id="why-causal-won" -->

## Why Decoder-Only Causal LM Won

:::columns cols="2" gap="30px"
- **Dense signal.** Every position is a prediction: $T$ examples per sequence. MLM learns only from the ~15% it masks.
- **Trivial data construction.** Input and target are the same stream, shifted by one.
+++
- **Natural generation.** The objective *is* generation: sample, append, repeat.
- **Prompt compatibility.** Classification, translation, and Q&A all become text completion.
:::

---

<!-- .slide: id="side-quest-compression" -->

## Side Quest: Compression Is Prediction

A model that predicts text well is a good **compressor**: encoding the next token takes about $-\log_2 p(\text{token})$ bits.

:::columns cols="2" gap="30px"
- Shannon's through-line from Module 1: **cross-entropy, perplexity, and bits per token** measure compression
- Lower loss = fewer bits per token = tighter description of the data
+++
- Minimizing loss and maximizing compression are the **same objective**
- Ilya Sutskever frames next-token prediction as compression; the **Hutter Prize** rewards compressing Wikipedia; DeepMind's **"Language Modeling Is Compression"** (2023) makes the equivalence precise
:::
