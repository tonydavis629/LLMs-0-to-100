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

<!-- .slide: id="bpe-algorithm" -->

## Byte-Pair Encoding

BPE starts from individual characters or bytes, then **repeatedly merges the most frequent adjacent pair** into a new token:

1. Start with character vocabulary
2. Count every adjacent pair in the training corpus
3. Merge the most frequent pair; add the merged symbol to the vocabulary
4. Repeat until the vocabulary reaches the desired size

:::columns cols="2" gap="30px"
**Example**

Corpus: `low low lower lowest`

Start: `l o w` (plus space, etc.)

Most frequent pair: `lo` (appears 4 times)

Merge: `lo w lo w lo wer lo west`

Next: `low` (appears 4 times)

Result: vocabulary contains `low`, `er`, `est`, and the individual characters.
+++
**What this means**

"lower" becomes `["low", "er"]`. "lowest" becomes `["low", "est"]`. The shared prefix "low" is reused. Rare words are decomposed into familiar pieces; frequent words are kept intact. The model never sees an unrepresentable token.
:::

---

:::figure img="images/sennrich_haddow_birch.jpg" name="Rico Sennrich, Barry Haddow & Alexandra Birch" kicker="Adapted BPE for Neural Machine Translation (2016)"
- Sennrich, Haddow, and Birch adapted Byte-Pair Encoding &mdash; originally a 1994 compression scheme by Philip Gage &mdash; into subword tokenization for NMT
- The idea: let the data decide which character sequences deserve their own tokens
- This became the dominant tokenization method for transformer models
- Sennrich et al., <https://arxiv.org/abs/1508.07909>
:::

---

<!-- .slide: id="byte-level-bpe" -->

## Byte-Level BPE and the Vocabulary Table

GPT-2 runs BPE over **raw bytes** instead of Unicode characters. Every possible string &mdash; including emoji, code, non-Latin scripts, and arbitrary whitespace &mdash; is representable with no true out-of-vocabulary tokens.

The vocabulary and the token-to-ID table are learned together during the merge process. Typical sizes:

- GPT-2: about 50k tokens
- Modern models: 100k&ndash;200k tokens

A larger vocabulary means shorter sequences (fewer tokens per sentence) but a bigger embedding matrix and a larger output projection layer. This is a direct memory-parameter trade-off.

Other schemes exist: **WordPiece** (BERT), **Unigram** and **SentencePiece** (T5, Llama). They differ in how the merge or segmentation objective is defined, but the core idea is the same: break rare words into reusable pieces.

---

<!-- .slide: id="tokenization-practice" -->

## Why Tokenization Matters in Practice

Token counts drive both context limits and API cost:

- A model's "context window" is measured in tokens, not words. A 4K context holds roughly 3,000 English words but often fewer in other languages.

- Non-English text frequently costs more tokens. The same sentence in Japanese or Arabic may require 1.5&ndash;2x more tokens than English. This is the **"token tax"** on multilingual use.

- Tokenization explains why models struggle to count letters, reverse strings, or do digit arithmetic. The model never sees the string "12345"; it sees a token ID for the entire number "12345" and has no access to its individual digits.

- A tokenizer and a model are trained separately. The model only ever sees token IDs. Glitch tokens &mdash; rare tokens that appear in the vocabulary but almost never in training data &mdash; can trigger bizarre behavior because the model learned no reliable representation for them.

The handoff to the model is simple: each token ID indexes one row of the embedding matrix. That vector is the only thing the transformer ever sees.
