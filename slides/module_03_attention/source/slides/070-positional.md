:::divider id="divider-positional" title="Positional Embeddings" sub="Teaching the model that order matters"
:::

---

<!-- .slide: id="permutation-equivariance" -->

## Self-Attention Is Permutation Equivariant

If you shuffle the input tokens, the attention output is the same shuffle. The operation has **no notion of order**.

:::note
**This is a problem.** "Dog bites man" and "man bites dog" have the same bag of tokens. Without position information, the model cannot distinguish them. Attention learns *what to look at* but not *where things are*.
:::

---

<!-- .slide: id="learned-vs-sinusoidal" -->

## Learned vs. Sinusoidal Positional Encodings

:::columns cols="2" gap="40px"
**Learned positional embeddings**

A trainable embedding table with one vector per position, added to the token embeddings.

$$X_{\text{pos}} = X + P$$

where $P$ is a learned matrix of shape (max_len, $d_{\text{model}}$).

Simple and effective. Used in BERT, GPT-2.
+++
**Sinusoidal positional encodings**

Fixed patterns based on sine and cosine at different frequencies:

$$PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

No parameters to learn. Extrapolates to unseen lengths. Used in the original Transformer.
:::

---

<!-- .slide: id="rope" -->

## Relative Position and RoPE

Both learned and sinusoidal encodings are **absolute** &mdash; they encode "position 5" as a fixed vector. But what often matters is **relative** position: "two tokens to the left."

**RoPE** (Rotary Position Embedding, Su et al. 2021) encodes relative position by rotating query and key vectors:

$$\mathbf{q}_m^T \mathbf{k}_n = f(m)^T f(n) = g(m - n)$$

The dot product depends only on the **relative offset** $m - n$, not the absolute positions. RoPE is used in LLaMA, Mistral, and most modern LLMs.
