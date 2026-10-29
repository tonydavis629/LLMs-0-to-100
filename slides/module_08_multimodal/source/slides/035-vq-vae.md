:::divider id="divider-vqvae" title="From Vectors to Ids" sub="VQ-VAE and discrete image tokens"
:::

---

<!-- .slide: id="vq-why" -->

## Why Make Image Tokens Discrete?

A ViT patch embedding is a continuous vector. The LLM can attend to it, so it works for **reading** an image. It does not work for **writing** one.

:::columns cols="2" gap="34px"
**An LLM's output is a choice from a list**

- The last layer produces logits over a fixed vocabulary
- Softmax picks one id, and cross-entropy scores that pick
- A continuous vector has no list to put a softmax over
+++
**So give images a vocabulary**

- Learn a finite set of $K$ vectors, the **codebook**
- Describe each region of the image by the index of its closest codebook vector
- Those indices are tokens in the same sense as word ids
:::

This is vector quantization. The **VQ-VAE** (van den Oord et al., 2017) learns the codebook together with an encoder and a decoder. <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="vq-diagram" -->

## VQ-VAE: Snap Each Vector to a Code

<div class="img-figure"><img src="images/vqvae.svg" alt="VQ-VAE: an encoder turns the image into a 3 by 3 grid of continuous vectors, each is replaced by the id of its nearest codebook vector, and a decoder rebuilds the image from the codebook vectors those ids point to" style="max-height: 300px;"></div>

$$k\_i = \arg\min\_j \lVert z\_i - e\_j \rVert \qquad z^q\_i = e\_{k\_i}$$

The encoder vector $z\_i$ is replaced by its nearest codebook vector $e\_{k\_i}$. The image becomes a grid of integers, and the decoder rebuilds the picture from the codebook vectors those integers point to. <!-- .element: class="text-lg" -->

---

<!-- .slide: id="vq-tradeoff" -->

## Continuous or Discrete?

<div class="compare-table">
<table>
<thead><tr><th>Continuous patch embeddings</th><th>Discrete codebook ids</th></tr></thead>
<tbody>
<tr><td>Any pixel values give a new vector, so fine detail survives</td><td>Every region becomes one of $K$ codes; with $K = 8{,}192$ that is 13 bits per id</td></tr>
<tr><td>The LLM can read images but cannot write them</td><td>Ids join the vocabulary, so the LLM can read <strong>and</strong> write images</td></tr>
<tr><td>LLaVA, BLIP-2, Flamingo (next sections)</td><td>Chameleon's image tokens, codec audio (later in this module)</td></tr>
</tbody>
</table>
</div>

The loss of detail is real: the Chameleon authors report that their tokenizer struggles to reconstruct images with a lot of text. Neural audio codecs such as EnCodec apply the same idea to sound, with a stack of codebooks called residual vector quantization. <!-- .element: class="text-lg" -->
