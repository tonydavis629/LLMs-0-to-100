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
      <span>The</span><span>&nbsp;</span><span>model</span><span>&nbsp;</span><span>reads</span><span>&nbsp;</span><span>caf<b class="mb-char">é</b></span>
    </div>
  </div>
  <div class="byte-arrow">&darr;</div>
  <div class="byte-lane">
    <h3>Raw bytes: UTF-8, written in hex</h3>
    <div class="byte-grid">
      <span>54</span><span>68</span><span>65</span><span>20</span><span>6d</span><span>6f</span><span>64</span><span>65</span><span>6c</span><span>20</span><span>72</span><span>65</span><span>61</span><span>64</span><span>73</span><span>20</span><span>63</span><span>61</span><span>66</span><span class="mb">c3</span><span class="mb">a9</span>
    </div>
    <p class="byte-note">UTF-8 uses 1 byte for ASCII and 2 to 4 for other characters: <b class="mb-char">é</b> = <b class="mb-char">c3 a9</b>. There are only 256 possible bytes, so the base vocabulary is 256 symbols.</p>
  </div>
  <div class="byte-arrow">&darr;</div>
  <div class="byte-lane">
    <h3>Learned token IDs (GPT-2)</h3>
    <div class="token-strip token-strip-accent">
      <span>464</span><span>2746</span><span>9743</span><span>40304</span>
    </div>
  </div>
</div>

<div class="vocab-tradeoff">
  <div><strong>Small vocabulary</strong><span>longer sequences</span></div>
  <div><strong>Large vocabulary</strong><span>bigger embedding table</span></div>
  <div><strong>Byte-level base</strong><span>no true out-of-vocabulary text</span></div>
</div>

---

<!-- .slide: id="special-tokens" -->

## Special Tokens

<div class="st-strip">
<span class="st-chip special"><b>&lt;|begin_of_text|&gt;</b><i>128000</i></span>
<span class="st-chip special"><b>&lt;|start_header_id|&gt;</b><i>128006</i></span>
<span class="st-chip"><b>user</b><i>882</i></span>
<span class="st-chip special"><b>&lt;|end_header_id|&gt;</b><i>128007</i></span>
<span class="st-chip"><b>\n\n</b><i>271</i></span>
<span class="st-chip"><b>What</b><i>3923</i></span>
<span class="st-chip"><b>is</b><i>374</i></span>
<span class="st-chip"><b>the</b><i>279</i></span>
<span class="st-chip"><b>capital</b><i>6864</i></span>
<span class="st-chip"><b>of</b><i>315</i></span>
<span class="st-chip"><b>France</b><i>9822</i></span>
<span class="st-chip"><b>?</b><i>30</i></span>
<span class="st-chip special"><b>&lt;|eot_id|&gt;</b><i>128009</i></span>
<span class="st-chip special"><b>&lt;|start_header_id|&gt;</b><i>128006</i></span>
<span class="st-chip"><b>assistant</b><i>78191</i></span>
<span class="st-chip special"><b>&lt;|end_header_id|&gt;</b><i>128007</i></span>
<span class="st-chip"><b>\n\n</b><i>271</i></span>
<span class="st-chip"><b>Paris</b><i>60704</i></span>
<span class="st-chip"><b>.</b><i>13</i></span>
<span class="st-chip special"><b>&lt;|eot_id|&gt;</b><i>128009</i></span>
</div>
<div class="st-strip st-strip-tool">
<span class="st-chip special"><b>&lt;|python_tag|&gt;</b><i>128010</i></span>
<span class="st-chip"><b>br</b><i>1347</i></span>
<span class="st-chip"><b>ave</b><i>525</i></span>
<span class="st-chip"><b>_search</b><i>10947</i></span>
<span class="st-chip"><b>.call</b><i>8692</i></span>
<span class="st-chip"><b>(query</b><i>10974</i></span>
<span class="st-chip"><b>=&quot;</b><i>429</i></span>
<span class="st-chip"><b>Paris</b><i>60704</i></span>
<span class="st-chip"><b>weather</b><i>9282</i></span>
<span class="st-chip"><b>&quot;)</b><i>909</i></span>
<span class="st-chip special"><b>&lt;|eom_id|&gt;</b><i>128008</i></span>
</div>
<p class="st-caption">Top: one chat turn in Llama 3's template. Bottom: a Llama 3.1 tool call. Orange tokens are reserved IDs that ordinary text never produces.</p>

<table class="st-table">
<tr><th>token (Llama 3.1)</th><th>what it does</th><th>ID</th><th>same idea elsewhere</th></tr>
<tr><td><code class="st-tok">&lt;|begin_of_text|&gt;</code></td><td>start of every sequence (BOS)</td><td class="st-id">128000</td><td>BERT <code>[CLS]</code></td></tr>
<tr><td><code class="st-tok">&lt;|end_of_text|&gt;</code></td><td>end of a document (EOS)</td><td class="st-id">128001</td><td>GPT-2 <code>&lt;|endoftext|&gt;</code></td></tr>
<tr><td><code class="st-tok">&lt;|start_header_id|&gt;</code></td><td>opens a role name: system, user, assistant, ipython</td><td class="st-id">128006</td><td>ChatML <code>&lt;|im_start|&gt;</code></td></tr>
<tr><td><code class="st-tok">&lt;|end_header_id|&gt;</code></td><td>closes the role name</td><td class="st-id">128007</td><td>&mdash;</td></tr>
<tr><td><code class="st-tok">&lt;|eot_id|&gt;</code></td><td>end of turn: the model stops here</td><td class="st-id">128009</td><td>ChatML <code>&lt;|im_end|&gt;</code></td></tr>
<tr><td><code class="st-tok">&lt;|python_tag|&gt;</code></td><td>the assistant starts a tool call</td><td class="st-id">128010</td><td>Qwen <code>&amp;lt;tool_call&amp;gt;</code></td></tr>
<tr><td><code class="st-tok">&lt;|eom_id|&gt;</code></td><td>pause so the tool result can come back</td><td class="st-id">128008</td><td>Qwen <code>&amp;lt;/tool_call&amp;gt;</code></td></tr>
<tr><td><code class="st-tok">&lt;|finetune_right_pad_id|&gt;</code></td><td>padding, ignored by attention and loss</td><td class="st-id">128004</td><td>BERT <code>[PAD]</code></td></tr>
</table>

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
