:::divider id="divider-tokenization" title="Tokenization" sub="What is the atomic unit of a language model?"
:::

---

<!-- .slide: id="unit-problem" -->

## The Unit Problem

What should a language model read? Three natural choices, each with a fatal flaw:

:::columns cols="3" gap="25px"
**Characters**

Tiny vocabulary (maybe 256 bytes), but sequences become very long. The model wastes capacity relearning spelling and morphology. Each token carries almost no meaning.
+++
**Words**

Meaningful units, but the vocabulary explodes to hundreds of thousands. Any unseen word is out-of-vocabulary with no way to represent it. Misspellings, typos, and brand-new words break the model.
+++
**Subwords**

The compromise: frequent words stay whole, rare words split into reusable pieces. "tokenization" becomes "token" + "ization". Morphology, typos, and new coinages are all representable.
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

<div class="handoff-diagram">
  <div class="token-strip"><span>token</span><span>ID</span><span>9048</span></div>
  <div class="byte-arrow">&rarr;</div>
  <div class="matrix-card">embedding row<br><strong>vector only</strong></div>
  <div class="byte-arrow">&rarr;</div>
  <div class="matrix-card">transformer<br><strong>never sees raw text</strong></div>
</div>
