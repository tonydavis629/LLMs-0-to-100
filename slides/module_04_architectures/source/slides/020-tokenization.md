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

Once text is tokenized, it is just a list of integer IDs. Each ID selects one row of the embedding table, and from that point on the transformer only ever sees vectors &mdash; it has no direct access to the original characters. This is why reversing a string or counting the letters in a word is surprisingly hard for an LLM: it never sees the letters, only the tokens they were grouped into.

---

<!-- .slide: id="side-quest-glitch-tokens" -->

## Side Quest: Glitch Tokens

The tokenizer and the model are trained **separately**, and that seam can crack. A rare string may appear often enough in the tokenizer's training data to earn its own vocabulary entry, yet show up almost never in the model's training data. Its embedding row then receives little or no gradient signal and stays close to its random initialization.

The classic example is the token <code>&nbsp;SolidGoldMagikarp</code> (originally a Reddit username). Prompts asking early GPT models to repeat or define it produced refusals, unrelated word substitutions, or garbled output &mdash; the model had simply never learned what that vector means.

<div class="glitch-example">
  <div class="glitch-turn user"><span>Prompt</span>Please repeat the string "SolidGoldMagikarp" back to me.</div>
  <div class="glitch-turn model"><span>Early GPT</span>"distribute"</div>
</div>

A real exchange: the glitch token is silently swapped for an unrelated word, because its embedding row never received a meaningful training signal.

These **under-trained tokens** (Land and Bartolo, 2024) make the abstraction boundary concrete: the model never sees the characters `S-o-l-i-d-...`. It receives a single **token ID**, looks up one **embedding vector**, and reasons only from there.
