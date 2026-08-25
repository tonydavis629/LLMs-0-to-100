:::divider id="divider-tokenization" title="Tokenization" sub="What is the atomic unit of a language model?"
:::

---

<!-- .slide: id="unit-problem" -->

## The Unit Problem

What should a language model read?

:::columns cols="3" gap="25px"
**Characters**

- Tiny vocabulary (~256 bytes)
- Very long sequences
- Each token carries almost no meaning
+++
**Words**

- Meaningful units
- Vocabulary explodes to hundreds of thousands
- Typos and new words become out-of-vocabulary
+++
**Subwords**

- Frequent words stay whole; rare words split into pieces
- "tokenization" &rarr; "token" + "ization"
- Typos and new coinages stay representable
:::

---

:::manim id="bpe-algorithm" scene="bpe-training" title="Byte-Pair Encoding"
:::

---

<!-- .slide: id="byte-level-bpe" -->

## Byte-Level BPE and the Vocabulary Table

<div class="byte-bpe-diagram">
  <div class="byte-lane">
    <h3>Text</h3>
    <div class="token-strip">
      <span>The</span><span>&nbsp;</span><span>model</span><span>&nbsp;</span><span>reads</span><span>&nbsp;</span><span>bytes</span>
    </div>
  </div>
  <div class="byte-arrow">&darr;</div>
  <div class="byte-lane">
    <h3>Raw bytes</h3>
    <div class="byte-grid">
      <span>54</span><span>68</span><span>65</span><span>20</span><span>6d</span><span>6f</span><span>64</span><span>65</span><span>6c</span><span>20</span><span>72</span><span>65</span><span>61</span><span>64</span><span>73</span>
    </div>
  </div>
  <div class="byte-arrow">&darr;</div>
  <div class="byte-lane">
    <h3>Learned token IDs</h3>
    <div class="token-strip token-strip-accent">
      <span>464</span><span>2746</span><span>1100</span><span>9048</span>
    </div>
  </div>
</div>

<div class="vocab-tradeoff">
  <div><strong>Small vocabulary</strong><span>longer sequences</span></div>
  <div><strong>Large vocabulary</strong><span>bigger embedding table</span></div>
  <div><strong>Byte-level base</strong><span>no true out-of-vocabulary text</span></div>
</div>

---

<!-- .slide: id="tokenization-practice" -->

## Why Tokenization Matters in Practice

<div class="token-practice-grid">
  <div class="token-case">
    <h3>English prose</h3>
    <div class="token-strip"><span>the</span><span>&nbsp;cat</span><span>&nbsp;sat</span></div>
    <p>Frequent pieces stay compact.</p>
  </div>
  <div class="token-case">
    <h3>Less represented scripts</h3>
    <div class="token-strip"><span>日</span><span>本</span><span>語</span><span>の</span><span>文</span><span>章</span></div>
    <p>More pieces can mean higher cost for the same idea.</p>
  </div>
  <div class="token-case">
    <h3>Strings and numbers</h3>
    <div class="token-strip"><span>12345</span><span>&nbsp;reverse</span><span>&nbsp;me</span></div>
    <p>The model sees token IDs, not guaranteed character access.</p>
  </div>
</div>

- Tokenized text is a list of integer IDs; each ID selects one embedding row
- The transformer sees only vectors, never the original characters
- This is why LLMs struggle to reverse strings or count letters

---

<!-- .slide: id="side-quest-glitch-tokens" -->

## Side Quest: Glitch Tokens

The tokenizer and the model are trained **separately**, and that seam can crack:

- A rare string earns a vocabulary entry in the tokenizer's data but rarely appears in the model's training data
- Its embedding row gets almost no gradient and stays near random initialization
- Classic case: <code>&nbsp;SolidGoldMagikarp</code>, a Reddit username; early GPT models refused it, swapped it, or produced garbage

<div class="glitch-example">
  <div class="glitch-turn user"><span>Prompt</span>Please repeat the string "SolidGoldMagikarp" back to me.</div>
  <div class="glitch-turn model"><span>Early GPT</span>"distribute"</div>
</div>

A real exchange: the model swaps the glitch token for an unrelated word.

These **under-trained tokens** (Land and Bartolo, 2024) prove the boundary: the model never sees `S-o-l-i-d-...`, only one token ID and its embedding vector.
