:::divider id="divider-objective" title="The Training Objective" sub="Causal language modeling, and the alternatives"
:::

---

<!-- .slide: id="causal-lm" -->

## Causal Language Modeling

The GPT-style objective, and the focus of this module: predict token $x_t$ using only the tokens before it.

The training pair is the **same sequence, shifted by one**:

:::columns cols="2" gap="30px"
**input** $x$

$$x_0,\ x_1,\ \ldots,\ x_{T-1}$$
+++
**target** $y$

$$x_1,\ x_2,\ \ldots,\ x_{T}$$
:::

The model outputs one vocabulary-sized logit vector at **every** position, so a single sequence of length $T$ produces $T$ predictions and $T$ loss terms at once. Constructing the labels is free: just shift.

---

:::manim id="next-token-anim" scene="next-token"
:::

---

<!-- .slide: id="objective-equation" -->

## The Objective, Precisely

Minimize the average negative log-probability of the true next token over the sequence:

$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T}\log p_\theta\left(x_t \mid x_{<t}\right)$$

This is exactly **cross-entropy** between the model's predicted distribution and the one-hot true next token, averaged over every position.

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

We will read these three numbers in detail in a few slides. They are the same quantity in different units.

---

:::figure img="images/radford.jpg" name="Alec Radford" kicker="Improving Language Understanding by Generative Pre-Training (2018)"
- Led the GPT line of generative pretraining at OpenAI
- Showed that a single decoder trained only to predict the next token transfers to many tasks
- GPT-2 (2019) made large-scale causal language modeling the dominant recipe
:::

---

<!-- .slide: id="other-objectives" -->

## Other Pretraining Objectives

Causal LM is not the only way to learn from raw text. Two important alternatives change **which tokens are predicted from which context**:

:::columns cols="2" gap="30px"
**Masked language modeling (BERT)**

Hide ~15% of tokens and predict them using context on **both sides**. Bidirectional, so it builds rich representations &mdash; but it is not a natural text generator, and only the masked fraction produces a training signal.
+++
**Denoising / span corruption (T5)**

Corrupt a passage (drop or shuffle spans) and train an encoder-decoder to **reconstruct** the original. A flexible "text-to-text" bridge, but it needs the heavier encoder-decoder machinery.
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

For building general-purpose generative models, the causal objective has decisive practical advantages:

:::columns cols="2" gap="30px"
- **Dense signal.** Every position is a prediction, so one sequence trains $T$ examples. MLM only learns from the ~15% it masks.
- **Trivial data construction.** Inputs and targets are the same stream, shifted by one. No pairing, no corruption scheme.
+++
- **Natural generation.** The training objective *is* generation: sample the next token, append, repeat.
- **Prompt compatibility.** "Understanding" tasks become text completion, so the same model handles classification, translation, and Q&A through prompting.
:::

---

<!-- .slide: id="side-quest-compression" -->

## Side Quest: Compression Is Prediction

A model that predicts text well is, literally, a good **compressor**. To encode the next token you need about $-\log_2 p(\text{token})$ bits; a model that assigns high probability to what actually comes next needs fewer bits to store the text.

:::columns cols="2" gap="30px"
- This is the Shannon through-line from Module 1: **cross-entropy, perplexity, and bits per token** are not decorative metrics &mdash; they measure compression.
- Lower pretraining loss = fewer bits per token = a tighter description of the data.
+++
- Ilya Sutskever has framed next-token prediction as compression; the **Hutter Prize** rewards compressing Wikipedia; DeepMind's **"Language Modeling Is Compression"** (2023) makes the equivalence precise.
- Minimizing loss and maximizing compression are the **same objective**.
:::
