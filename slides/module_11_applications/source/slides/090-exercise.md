:::divider id="divider-exercise" title="Exercise" sub="Build two retrievers and find where each one wins"
:::

---

<!-- .slide: id="exercise-run" -->

## Running the Exercise

Open `module_11_applications/exercise.py` and fill in the eight `NotImplementedError` lines. The corpus, labeled queries, encoder, plotting, and runner are provided. Run after each step; unfinished steps are skipped. <!-- .element: class="text-lg" -->

```bash
# Two retrievers over one support corpus
cd exercises
uv run python module_11_applications/src/main.py
```

- Sparse retriever works after step 5; dense joins after step 6; report after step 7
- The runner prints two worked examples, then the per-category table
- Bar chart saved to `output/retrieval_comparison.png` <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="exercise-overview" -->

## Exercise: Nothing Here Trains a Model

The corpus: 48 short support articles for fictional PX-series printers. You build **two complete retrievers** and score them on 30 labeled queries.

:::columns cols="2" gap="30px"
**You write the retrieval math**

- Tokenization, IDF, TF-IDF vectors, cosine similarity, top-k ranking: one full retriever from scratch
- One pooling function turns the bundled MiniLM encoder into the second retriever, reusing your ranking code
- recall@k and MRR score both
+++
**The payoff is the table**

- Sparse: perfect on exact identifiers, near useless on paraphrases
- Dense: the mirror image
- Overall MRR is **identical to two decimal places**; only the per-category table tells them apart
:::

---

<!-- .slide: id="exercise-data" -->

## The Data

<div class="bench-table">
<table>
<thead><tr><th>File</th><th>Contents</th><th>Role</th></tr></thead>
<tbody>
<tr><td><code>articles.jsonl</code></td><td>48 support articles: 10 error codes (E-341, E-520, ...), 6 part numbers (DR-4410, TN-2211, ...), 32 how-to and troubleshooting pages</td><td>The corpus both retrievers index</td></tr>
<tr><td><code>queries.jsonl</code></td><td>30 queries, each labeled with its relevant article and a category: <code>keyword</code> (contains an exact identifier), <code>paraphrase</code> (shares almost no vocabulary with its answer), <code>verbatim</code> (reuses the article's own wording)</td><td>The labeled evaluation set</td></tr>
<tr><td><code>encoder/</code></td><td>all-MiniLM-L6-v2: 23M parameters, 384 dimensions, fp16, ~44MB, copied from <code>huggingface.co/sentence-transformers/all-MiniLM-L6-v2</code></td><td>The dense retriever's encoder; runs on CPU, no network needed</td></tr>
</tbody>
</table>
</div>

The categories are the experiment design: one favors each retriever, the third favors both. The labels make the comparison quantitative. <!-- .element: class="text-lg" -->

---

:::step id="exercise-step1" title="Step 1: tokenize()"
```python
def tokenize(text: str) -> list[str]:
    """Turn raw text into a list of normalized terms."""
    # Lowercase, then map every non-alphanumeric character to a space.
    cleaned = "".join(ch if ch.isalnum() else " " for ch in text.lower())
    # TODO: Return the list of terms in `cleaned`, split on whitespace.
    raise NotImplementedError("TODO: split the cleaned text into terms")
```
+++
**Hint:** `.split()` with no argument splits on any run of whitespace and drops the empty pieces.
+++
**Answer:**

```python
return cleaned.split()
```

Note what this does to "E-341": it becomes the terms `e` and `341`, and the rare term `341` is what the sparse retriever will latch onto.
:::

---

:::step id="exercise-step2" title="Step 2: inverse_document_frequency()"
```python
def inverse_document_frequency(tokenized_docs: list[list[str]]) -> dict[str, float]:
    """Weight each term by how rare it is across the corpus."""
    total_docs = len(tokenized_docs)
    # How many documents contain each term. set() collapses repeats first, so a
    # term used ten times in one article still counts that article only once.
    document_frequency: Counter[str] = Counter()
    for tokens in tokenized_docs:
        document_frequency.update(set(tokens))
    # TODO: Return a dict mapping each term to log(total_docs / its document
    #       frequency).
    raise NotImplementedError("TODO: compute the IDF weight for each term")
```
+++
**Hint:** a dict comprehension over `document_frequency.items()`; `math.log`.
+++
**Answer:**

```python
return {term: math.log(total_docs / df)
        for term, df in document_frequency.items()}
```

This is Module 1's surprisal: the information content of "a document contains this term". A term in every document scores log(1) = 0 and stops counting.
:::

---

:::step id="exercise-step3" title="Step 3: tfidf_vector()"
```python
def tfidf_vector(tokens: list[str], idf: dict[str, float],
                 vocab_index: dict[str, int]) -> np.ndarray:
    """Build one TF-IDF vector: term counts, each weighted by the term's IDF."""
    counts = Counter(tokens)
    vector = np.zeros(len(vocab_index), dtype=np.float64)
    for term, count in counts.items():
        if term not in vocab_index:
            continue  # a query term the corpus never uses
        # TODO: Set the vector entry for this term: its count in this text
        #       times its IDF weight.
        raise NotImplementedError("TODO: fill in the term's TF-IDF entry")
    return vector
```
+++
**Hint:** `vocab_index[term]` is the slot; `idf[term]` is the weight.
+++
**Answer:**

```python
vector[vocab_index[term]] = count * idf[term]
```

A rare term mentioned twice now outweighs a common term mentioned five times. Queries go through this same function, with the corpus's IDF table.
:::

---

:::terminal id="exercise-output-1" title="After Step 3: The Sparse Index Exists" cmd="uv run python module_11_applications/src/main.py" caption="Actual output. 48 articles became 48 vectors with one dimension per vocabulary term, almost all zero. Ranking them needs steps 4 and 5."
<span class="header">MODULE 11: two retrievers over one support corpus</span>
Corpus and queries
  corpus            48 support articles (fictional PX-series printers)
  queries           30 labeled queries: 10 keyword, 10 paraphrase, 10 verbatim
  sparse retriever  TF-IDF vectors, built from scratch (steps 1-5)
  dense retriever   MiniLM embeddings, 384 dims, mean-pooled (step 6)
  scoring           recall@1, recall@3, MRR, per category (step 7)

<span class="header">1. SPARSE INDEX (TF-IDF)</span>
  Indexed 48 articles: one vector each, with one
  dimension per vocabulary term (811 terms). On average only
  38 of the 811 entries are nonzero, which is why these
  vectors are called sparse.
  <span class="skipped">[ranking needs cosine_similarity() and rank_documents(), steps 4-5]</span>

<span class="skipped">2. DENSE INDEX  [skipped: implement mean_pool(), step 6]</span>
<span class="skipped">3. WORKED EXAMPLES  [skipped: no retriever is runnable yet]</span>
<span class="skipped">4. THE REPORT  [skipped: no retriever is runnable yet]</span>
:::

---

:::step id="exercise-step4" title="Step 4: cosine_similarity()"
```python
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Score how similar two vectors are by the angle between them."""
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 0.0  # a query with no known terms matches nothing
    # TODO: Return the dot product of a and b, divided by `denominator`.
    raise NotImplementedError("TODO: compute the cosine similarity")
```
+++
**Hint:** `np.dot(a, b)` is the dot product; wrap the result in `float()`.
+++
**Answer:**

```python
return float(np.dot(a, b) / denominator)
```

Dividing out both lengths means a long rambling document cannot outscore a short focused one just by having bigger numbers. Both retrievers will use this one function.
:::

---

:::step id="exercise-step5" title="Step 5: rank_documents()"
```python
def rank_documents(query_vector: np.ndarray, doc_vectors: np.ndarray,
                   k: int) -> list[int]:
    """Score every document against the query and return the top k indices."""
    scores = [cosine_similarity(query_vector, doc_vector)
              for doc_vector in doc_vectors]
    # TODO: Return the indices of the k highest scores, highest first.
    raise NotImplementedError("TODO: rank the documents and keep the top k")
```
+++
**Hint:** `sorted(range(len(scores)), key=..., reverse=True)` sorts document indices by their score; slice the first k.
+++
**Answer:**

```python
return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
```

This function never looks at what the vectors mean. Steps 1-5 plus a different vectorizer equals a different retriever, which is exactly what step 6 exploits.
:::

---

:::terminal id="exercise-output-2" title="After Step 5: A Complete Retriever" cmd="uv run python module_11_applications/src/main.py" caption="Actual output. The keyword query works: the rare term 341 dominates the score. The paraphrase query fails completely: not one content word is shared with the right article, so it never ranks."
<span class="header">3. WORKED EXAMPLES</span>
  [keyword] 'pages take ages to come out and now it says E-341'
  labeled answer: err-e341
    sparse:
      <span class="success">1. 0.360  err-e341        Error E-341: fuser temperature fault    [relevant]</span>
      2. 0.104  art-slow        Long delay before the first page
      3. 0.073  art-blank       Pages print blank

  [paraphrase] 'my pages come out crumpled'
  labeled answer: art-wrinkled
    sparse:
      <span class="t-fail">1. 0.133  art-toner-low   Toner low warning: what it means</span>
      <span class="t-fail">2. 0.130  art-blank       Pages print blank</span>
      <span class="t-fail">3. 0.110  part-dr4410     Drum unit DR-4410: when and how to rep</span>

<span class="skipped">4. THE REPORT  [skipped: implement recall_at_k() and reciprocal_rank(), step 7]</span>
:::

---

:::step id="exercise-step6" title="Step 6: mean_pool()"
```python
def mean_pool(token_vectors: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Average an encoder's per-token vectors into one vector for the text."""
    # Give the mask a second axis, (tokens, 1), so it broadcasts across the
    # 384 vector dimensions when multiplied with token_vectors.
    mask = attention_mask.astype(np.float64)[:, None]
    # TODO: Return the sum of the masked token vectors divided by the number
    #       of real tokens.
    raise NotImplementedError("TODO: average the real token vectors")
```
+++
**Hint:** `(token_vectors * mask).sum(axis=0)` sums the real token vectors; `mask.sum()` counts the real tokens.
+++
**Answer:**

```python
return (token_vectors * mask).sum(axis=0) / mask.sum()
```

The encoder emits one 384-dimensional vector per token; retrieval needs one per document. Texts are encoded in batches and padded to the longest, so the mask keeps padding tokens out of the average.
:::

---

:::terminal id="exercise-output-3" title="After Step 6: The Second Retriever Joins" cmd="uv run python module_11_applications/src/main.py" caption="Actual output. The failures have swapped. Dense nails the paraphrase (crumpled and wrinkled are neighbors in embedding space) and loses the error code: the symptom words drag the embedding toward the wrong articles, and E-341 is not a rare token to an encoder that splits it into subwords."
<span class="header">2. DENSE INDEX (MiniLM EMBEDDINGS)</span>
  Encoding 48 articles with the bundled MiniLM (CPU, a few seconds)...
  Indexed 48 articles: one 384-dimensional embedding
  each, every entry nonzero. Same corpus, entirely different geometry.

<span class="header">3. WORKED EXAMPLES</span>
  [keyword] 'pages take ages to come out and now it says E-341'
    sparse:
      <span class="success">1. 0.360  err-e341        Error E-341: fuser temperature fault    [relevant]</span>
    dense:
      <span class="t-fail">1. 0.447  art-slow        Long delay before the first page</span>
      <span class="t-fail">2. 0.427  art-blank       Pages print blank</span>
      <span class="t-fail">3. 0.415  err-e520        Error E-520: toner supply sensor fault</span>

  [paraphrase] 'my pages come out crumpled'
    sparse:
      <span class="t-fail">1. 0.133  art-toner-low   Toner low warning: what it means</span>
    dense:
      <span class="success">1. 0.637  art-wrinkled    Creased or wrinkled output              [relevant]</span>
      2. 0.524  art-blank       Pages print blank
      3. 0.380  adm-humidity    Paper storage and humidity
:::

---

:::step id="exercise-step7" title="Step 7: recall_at_k() and reciprocal_rank()"
```python
def recall_at_k(ranked_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """What fraction of the relevant documents made it into the top k?"""
    # TODO: Return the fraction of relevant_ids that appear in the first k
    #       entries of ranked_ids.
    raise NotImplementedError("TODO: compute recall@k")

def reciprocal_rank(ranked_ids: list[str], relevant_ids: list[str]) -> float:
    """1 over the rank of the first relevant document (0.0 if none is found)."""
    for position, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            # TODO: Return the reciprocal of this (1-based) position.
            raise NotImplementedError("TODO: return the reciprocal rank")
    return 0.0
```
+++
**Hint:** `sum(1 for ...)` counts matches against `ranked_ids[:k]`, divided by `len(relevant_ids)`. For the second function, `position` is already 1-based thanks to `enumerate`'s `start=1`.
+++
**Answer:**

```python
return sum(1 for doc_id in relevant_ids
           if doc_id in ranked_ids[:k]) / len(relevant_ids)
```

```python
return 1.0 / position
```
:::

---

:::terminal id="exercise-output-4" title="After Step 7: The Report" cmd="uv run python module_11_applications/src/main.py" caption="Actual output. Read the columns: sparse sweeps keyword, dense sweeps paraphrase, both ace verbatim. Then read the overall row: recall@1 says sparse, recall@3 says dense, and MRR is a dead tie at 0.75."
<span class="header">4. THE REPORT</span>
    category       n    sparse (r@1   r@3   MRR)     dense (r@1   r@3   MRR)
    keyword       10         <span class="success">100%  100%  1.00</span>             <span class="t-fail">30%   60%  0.53</span>
    paraphrase    10          <span class="t-fail">10%   30%  0.25</span>             <span class="success">50%  100%  0.72</span>
    verbatim      10         100%  100%  1.00            100%  100%  1.00
    overall       30          70%   77%  <span class="t-cyan">0.75</span>             60%   87%  <span class="t-cyan">0.75</span>

  Chart saved to output/retrieval_comparison.png
  Read the category rows before believing the overall row.
:::

---

<!-- .slide: id="exercise-chart" -->

## The Picture

<div class="img-figure">
  <img src="images/retrieval_comparison.png" alt="Grouped bar chart of recall@3 per query category for the sparse and dense retrievers, with the overall score at right">
</div>

The two rightmost bars are what a leaderboard would show. The six to their left are why it would mislead you. (Actual exercise output.) <!-- .element: class="text-lg" style="margin-top: 6px;" -->

---

<!-- .slide: id="exercise-ship" -->

## Which Retriever Would You Ship?

:::columns cols="2" gap="34px"
**The case for sparse**

- Perfect on every error-code and part-number query, the bulk of support search
- Zero model dependencies, indexes in milliseconds
- Every score explainable by pointing at shared words
+++
**The case for dense**

- The only one that understands symptoms described in the user's own words
- Real users always do that
- 100% recall@3 on paraphrase against sparse's 30%
:::

**Production systems refuse the choice and run both.** Hybrid search merges the rankings (reciprocal rank fusion, the first extra credit); a reranker cleans up the shortlist. The per-category table tells you the merge is worth the complexity. <!-- .element: class="text-lg" style="margin-top: 10px;" -->

---

<!-- .slide: id="exercise-extra-credit" -->

## Extra Credit

- **Hybrid search.** Merge the two rankings with reciprocal rank fusion: score each document by the sum of 1/(60 + rank) across both rankings. Does the fusion beat both retrievers overall?
- **BM25's saturating TF.** Replace the raw count with `count * (k1 + 1) / (count + k1)`, k1 = 1.5, so the tenth repetition of a term is worth less than the first.
- **Watch IDF zero a term out.** Add "printer" to any query and confirm it changes almost nothing; then look up its IDF weight.
- **RAG prompt assembly.** Format the top article and the query into a grounded prompt with Module 6's chat template: the exact seam where the generator attaches.
- **Retrieve with the course's own model.** Embed the corpus with mean-pooled hidden states from the Module 5 checkpoint and measure how far retrieval quality drops without the contrastive objective. <!-- .element: class="text-lg" style="margin-top: 8px;" -->
