:::divider id="divider-exercise" title="Exercise" sub="Build two retrievers and find where each one wins"
:::

---

<!-- .slide: id="exercise-run" -->

## Running the Exercise

Open `module_11_applications/exercise.py`, the only file you edit, and fill in the eight `NotImplementedError` lines. The corpus, labeled queries, encoder, plotting, and runner are provided. Run after each step. <!-- .element: class="text-lg" -->

```bash
# Run every step; each is tagged CORRECT, INCORRECT, or INCOMPLETE
cd exercises
uv run python module_11_applications/src/main.py

# Run a single step (1-7)
uv run python module_11_applications/src/main.py --step 5
```

- Two worked example queries go through the sparse retriever at step 5 and the dense retriever at step 6
- Step 7 prints the per-category table and saves the bar chart to `output/retrieval_comparison.png` <!-- .element: class="text-lg" style="margin-top: 12px;" -->

---

<!-- .slide: id="exercise-results" -->

## Reading the Results

For each step the runner prints what your code produced on the real corpus, then runs the step's **tests** from `tests/`. Each test calls your function on small inputs whose correct answer is known. <!-- .element: class="text-lg" -->

- **CORRECT**: every test for the step passed
- **INCORRECT**: your code ran but a test failed; the expected and actual values are printed beneath
- **INCOMPLETE**: the function still raises `NotImplementedError`, or the step needs an earlier one you have not finished

The tag sits on the step's header line, and the individual test results follow the output. A ranking can look right and still be wrong, so the tests are how you know a step is done. <!-- .element: class="text-lg" style="margin-top: 15px;" -->

---

:::terminal id="exercise-results-example" title="What an INCORRECT Step Looks Like" cmd="uv run python module_11_applications/src/main.py --step 6" maxw="920px" caption="Here <code>mean_pool()</code> took a plain mean and ignored the mask. The rankings barely move (the right article still wins, at 0.625 instead of 0.637), so the output alone would not give it away. The failing tests show the padding row leaking into the average and say to multiply by the mask."
<span class="header">=== Step 6: mean_pool() ===</span> <span class="t-fail">INCORRECT</span>
Encoding 48 articles with the bundled MiniLM (CPU, a few seconds)...
<span class="skipped">...</span>
  [paraphrase] 'my pages come out crumpled'
  labeled answer: art-wrinkled
    dense:
      1. 0.625  art-wrinkled    Creased or wrinkled output              [relevant]
      2. 0.510  art-blank       Pages print blank
      3. 0.368  adm-humidity    Paper storage and humidity
  <span class="success">CORRECT</span>    averages the real tokens: [1, 2] and [3, 4] with mask [1, 1] give [2.0, 3.0]
  <span class="t-fail">INCORRECT</span>  ignores padding: adding a row [100, 100] with mask 0 still gives [2.0, 3.0]
             expected [2.0, 3.0], got [34.6667, 35.3333]
             (the padding row was averaged in; multiply by the mask first)
  <span class="success">CORRECT</span>    returns one vector of shape (384,) for 10 tokens of shape (10, 384)
  <span class="t-fail">INCORRECT</span>  real encoder: a query pooled alone and padded from 7 to 44 tokens gives the same vector
             the two vectors differ by up to 3.351 (padding is leaking into the average)
:::

---

<!-- .slide: id="exercise-overview" -->

## Exercise: Nothing Here Trains a Model

The corpus: 48 short support articles for fictional PX-series printers. You build **two complete retrievers** and score them on 30 labeled queries. Every step has its own tests in `tests/`.

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
    """Turn raw text into a list of normalized terms.

    Sparse retrieval matches words, so both documents and queries must pass
    through the same normalization or "Fuser" and "fuser." would never match.
    The rule here is deliberately simple: lowercase everything, replace every
    character that is not a letter or digit with a space, and split. Note what
    this does to an identifier like E-341: it becomes the two terms "e" and
    "341", and the rare term "341" is exactly what the sparse retriever will
    latch onto later.

    Args:
        text: A document body or a query, as one string.

    Returns:
        The list of terms, in order, possibly with repeats.
    """
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
    """Weight each term by how rare it is across the corpus.

    A term that appears in 3 documents out of 48 narrows the search far more
    than one that appears in 40 of them, and IDF turns that intuition into a
    number: log(N / df), where N is the corpus size and df is how many
    documents contain the term. This is Module 1's information theory wearing
    a retrieval hat: log(N / df) is, up to the base of the log, the surprisal
    of the event "a document contains this term". Rare term, high surprisal,
    big weight; a term in every document scores log(1) = 0 and stops counting
    entirely.

    Args:
        tokenized_docs: One token list per document (the output of step 1).

    Returns:
        A dict mapping every term in the corpus to its IDF weight.
    """
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

:::terminal id="exercise-output-idf" title="After Step 2: Rare Terms Weigh More" cmd="uv run python module_11_applications/src/main.py" maxw="920px" caption="Actual output. E-341 splits into <code>e</code> and <code>341</code>, and 341 appears in one article, so it gets the top weight. The paraphrase query's key word, crumpled, appears in none."
<span class="header">=== Step 1: tokenize() ===</span> <span class="success">CORRECT</span>
Corpus: 48 support articles for fictional PX-series printers
Queries: 30 labeled (10 keyword, 10 paraphrase, 10 verbatim)
  'Error E-341: fuser temperature fault'
    -> ['error', 'e', '341', 'fuser', 'temperature', 'fault']
  'my pages come out crumpled'
    -> ['my', 'pages', 'come', 'out', 'crumpled']
  All 48 articles: 2551 terms in total, 811 distinct
  <span class="success">CORRECT</span>    lowercase, split at punctuation: 'Fuser E-341: fuser' gives ['fuser', 'e', '341', 'fuser']
  <span class="success">CORRECT</span>    leaves no empty terms when spaces repeat: '  paper,,  jam ' gives ['paper', 'jam']
  <span class="success">CORRECT</span>    a string with no letters or digits gives no terms: '?! ...' gives []

<span class="header">=== Step 2: inverse_document_frequency() ===</span> <span class="success">CORRECT</span>
IDF over 48 articles, 811 distinct terms. Rarer terms weigh more:
    term       in docs     idf
    <span class="t-cyan">341              1   3.871</span>
    fuser            9   1.674
    printer         18   0.981
    the             48   0.000
  'crumpled' is in no article, so it has no weight and no query can match on it
  <span class="success">CORRECT</span>    returns one weight per distinct term: 'the', 'fuser', 'fault', 'jam'
  <span class="success">CORRECT</span>    a term in 1 of 3 documents weighs log(3/1) = 1.099 ('jam', even though it repeats)
  <span class="success">CORRECT</span>    a term in 2 of 3 documents weighs log(3/2) = 0.405 ('fuser')
  <span class="success">CORRECT</span>    a term in every document weighs log(3/3) = 0: it cannot tell documents apart ('the')

<span class="header">=== Step 3: tfidf_vector() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">TODO: fill in the term's TF-IDF entry</span>
:::

---

:::step id="exercise-step3" title="Step 3: tfidf_vector()"
```python
def tfidf_vector(tokens: list[str], idf: dict[str, float],
                 vocab_index: dict[str, int]) -> np.ndarray:
    """Build one TF-IDF vector: term counts, each weighted by the term's IDF.

    This is where a document becomes a vector. The vector has one slot per
    vocabulary term, almost all of them zero, which is why this family of
    methods is called sparse. A term's entry is its count in this document
    (term frequency) times its corpus-wide IDF weight, so a rare term
    mentioned twice dominates a common term mentioned five times. Queries go
    through this same function; a query term the corpus has never seen is
    simply skipped, since no document could match it anyway.

    Args:
        tokens: The document's (or query's) terms from step 1.
        idf: The IDF weights from step 2.
        vocab_index: term -> its slot in the vector, fixed for the corpus.

    Returns:
        A vector of shape (vocabulary size,).
    """
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

:::terminal id="exercise-output-1" title="After Step 3: The Sparse Index Exists" cmd="uv run python module_11_applications/src/main.py" maxw="920px" caption="Actual output. 48 articles became 48 vectors with one dimension per vocabulary term, almost all zero. The biggest entry for err-e341 is 341: three mentions times the top IDF weight."
<span class="header">=== Step 1: tokenize() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 2: inverse_document_frequency() ===</span> <span class="success">CORRECT</span>
<span class="skipped">  ...</span>

<span class="header">=== Step 3: tfidf_vector() ===</span> <span class="success">CORRECT</span>
Indexed 48 articles: one vector each, with one
dimension per vocabulary term (811 terms). On average only
38 of the 811 entries are nonzero, which is why these
vectors are called sparse.
  Largest entries in the vector for err-e341:
    <span class="t-cyan">341     3 x 3.871 = 11.61</span>
    cool    2 x 3.871 =  7.74
    fuser   4 x 1.674 =  6.70
  <span class="success">CORRECT</span>    count x IDF: ['jam', 'fuser', 'jam'] with idf fuser=2, jam=0.5 gives [2.0, 1.0, 0.0]
  <span class="success">CORRECT</span>    a term with IDF 0 contributes 0, even repeated 5 times: gives [0.0, 1.5]
  <span class="success">CORRECT</span>    matches count x IDF in every slot for a random 200-term text

<span class="header">=== Step 4: cosine_similarity() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">TODO: compute the cosine similarity</span>
:::

---

:::step id="exercise-step4" title="Step 4: cosine_similarity()"
```python
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Score how similar two vectors are by the angle between them.

    Cosine similarity is the dot product with both lengths divided out, so a
    long rambling document cannot outscore a short focused one just by having
    bigger numbers. It runs from 1.0 (same direction) through 0.0 (nothing in
    common). Both retrievers use this one function: TF-IDF vectors and
    embedding vectors are compared the exact same way, which is what makes the
    two retrievers swappable from here on.

    Args:
        a: One vector.
        b: Another vector of the same shape.

    Returns:
        The cosine similarity as a plain float.
    """
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

:::terminal id="exercise-output-cosine" title="After Step 4: Similarity Without Length" cmd="uv run python module_11_applications/src/main.py" maxw="920px" caption="Actual output. The closest pair is the standard and high-yield versions of one toner cartridge. Pasting an article twice doubles every count, yet its cosine with the original stays 1.000."
<span class="header">=== Step 1: tokenize() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 2: inverse_document_frequency() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 3: tfidf_vector() ===</span> <span class="success">CORRECT</span>
<span class="skipped">  ...</span>

<span class="header">=== Step 4: cosine_similarity() ===</span> <span class="success">CORRECT</span>
Most similar of the 1128 article pairs, by TF-IDF cosine:
    0.408  part-tn2210 and part-tn2211
    0.384  art-driver-mac and art-driver-win
    0.278  art-slow and adm-sleep
  err-e341 against its own text pasted twice: <span class="t-cyan">1.000</span> (length does not count)
  <span class="success">CORRECT</span>    parallel vectors score 1.0 whatever their lengths: [1, 2, 3] and [2, 4, 6]
  <span class="success">CORRECT</span>    perpendicular vectors score 0.0: [1, 0] and [0, 3]
  <span class="success">CORRECT</span>    hand-computed: [1, 2] and [3, 4] give 11 / (sqrt(5) x 5) = 0.9839
  <span class="success">CORRECT</span>    agrees with torch.nn.functional.cosine_similarity on random 384-dim vectors

<span class="header">=== Step 5: rank_documents() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">TODO: rank the documents and keep the top k</span>
:::

---

:::step id="exercise-step5" title="Step 5: rank_documents()"
```python
def rank_documents(query_vector: np.ndarray, doc_vectors: np.ndarray,
                   k: int) -> list[int]:
    """Score every document against the query and return the top k indices.

    This is the whole retrieval step: one similarity per document, then sort.
    Note that it never looks at what the vectors mean; it works identically
    for TF-IDF vectors and for embeddings, which is why steps 1-5 plus a
    different vectorizer equals a different retriever. Real systems replace
    this linear scan with an approximate index (HNSW and friends) once the
    corpus outgrows brute force; at 48 documents, brute force is instant.

    Args:
        query_vector: The query's vector.
        doc_vectors: One row per document, same width as the query vector.
        k: How many results to return.

    Returns:
        The indices of the k best-scoring documents, best first.
    """
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

:::terminal id="exercise-output-2" title="After Step 5: A Complete Retriever" cmd="uv run python module_11_applications/src/main.py" maxw="920px" caption="Actual output. The keyword query works: the rare term 341 dominates the score. The paraphrase query fails completely: not one content word is shared with the right article, so it never ranks."
<span class="header">=== Step 1: tokenize() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 2: inverse_document_frequency() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 3: tfidf_vector() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 4: cosine_similarity() ===</span> <span class="success">CORRECT</span>
<span class="skipped">  ...</span>

<span class="header">=== Step 5: rank_documents() ===</span> <span class="success">CORRECT</span>
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
  <span class="success">CORRECT</span>    returns the k best indices, best first: scores [0.1, 0.9, 0.5, 0.7], k=2 give [1, 3]
  <span class="success">CORRECT</span>    k = corpus size ranks every document: scores [0.2, 0.8, 0.5], k=3 give [1, 2, 0]
  <span class="success">CORRECT</span>    agrees with numpy's argsort on 30 random documents, k=5

<span class="header">=== Step 6: mean_pool() ===</span> <span class="skipped">INCOMPLETE</span>
  <span class="skipped">TODO: average the real token vectors</span>
:::

---

:::step id="exercise-step6" title="Step 6: mean_pool()"
```python
def mean_pool(token_vectors: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Average an encoder's per-token vectors into one vector for the text.

    The provided encoder (a 23M-parameter MiniLM, trained contrastively the
    way Module 8's CLIP was, but with text on both sides) reads a text and
    emits one 384-dimensional vector per token. Retrieval needs one vector
    per document, and the standard answer is the mean of the token vectors.
    One catch: texts are encoded in batches, so short texts are padded to the
    longest one, and the padding tokens carry vectors too. The attention mask
    marks real tokens with 1 and padding with 0, and only real tokens may
    count toward the average.

    Args:
        token_vectors: Shape (tokens, 384), including padding positions.
        attention_mask: Shape (tokens,), 1.0 for real tokens, 0.0 for padding.

    Returns:
        A vector of shape (384,): the mean over the real tokens only.
    """
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

:::terminal id="exercise-output-3" title="After Step 6: The Second Retriever Joins" cmd="uv run python module_11_applications/src/main.py" maxw="920px" caption="Actual output. The failures have swapped: dense finds the paraphrase (crumpled and wrinkled sit close in embedding space) and misses the error code."
<span class="header">=== Step 1: tokenize() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 2: inverse_document_frequency() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 3: tfidf_vector() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 4: cosine_similarity() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 5: rank_documents() ===</span> <span class="success">CORRECT</span>
<span class="skipped">  ...</span>

<span class="header">=== Step 6: mean_pool() ===</span> <span class="success">CORRECT</span>
Encoding 48 articles with the bundled MiniLM (CPU, a few seconds)...
Indexed 48 articles: one 384-dimensional embedding each. On average
382 of the 384 entries are nonzero: dense, where TF-IDF was sparse.
  [keyword] 'pages take ages to come out and now it says E-341'
  labeled answer: err-e341
    dense:
      <span class="t-fail">1. 0.447  art-slow        Long delay before the first page</span>
      <span class="t-fail">2. 0.427  art-blank       Pages print blank</span>
      <span class="t-fail">3. 0.415  err-e520        Error E-520: toner supply sensor fault</span>

  [paraphrase] 'my pages come out crumpled'
  labeled answer: art-wrinkled
    dense:
      <span class="success">1. 0.637  art-wrinkled    Creased or wrinkled output              [relevant]</span>
      2. 0.524  art-blank       Pages print blank
      3. 0.380  adm-humidity    Paper storage and humidity
  <span class="success">CORRECT</span>    averages the real tokens: [1, 2] and [3, 4] with mask [1, 1] give [2.0, 3.0]
  <span class="success">CORRECT</span>    ignores padding: adding a row [100, 100] with mask 0 still gives [2.0, 3.0]
  <span class="success">CORRECT</span>    returns one vector of shape (384,) for 10 tokens of shape (10, 384)
  <span class="success">CORRECT</span>    real encoder: a query pooled alone and padded from 7 to 44 tokens gives the same vector
:::

---

:::step id="exercise-step7" title="Step 7: recall_at_k() and reciprocal_rank()"
```python
def recall_at_k(ranked_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """What fraction of the relevant documents made it into the top k?

    Recall@k is the retrieval half of evaluating a RAG system, and it uses the
    vocabulary Module 9 borrowed from this field. It asks the only question
    the generator downstream cares about: did the right document actually make
    it into the context window? For our queries with a single relevant
    article, recall@k is simply 1.0 if that article is in the top k, else 0.0.

    Args:
        ranked_ids: The retriever's ranking, best first.
        relevant_ids: The labeled correct document ids for this query.
        k: How deep into the ranking to look.

    Returns:
        A score in [0.0, 1.0].
    """
    # TODO: Return the fraction of relevant_ids that appear in the first k
    #       entries of ranked_ids.
    raise NotImplementedError("TODO: compute recall@k")


def reciprocal_rank(ranked_ids: list[str], relevant_ids: list[str]) -> float:
    """1 over the rank of the first relevant document (0.0 if none is found).

    Recall@k treats positions 1 through k the same; reciprocal rank cares
    where in the list the hit landed: 1.0 for first place, 0.5 for second,
    0.33 for third. Averaged over all queries this is MRR, mean reciprocal
    rank, the standard single number for "how high does the right answer
    rank". Position matters downstream too: Module 11's lost-in-the-middle
    result says text buried mid-context gets used less than text at the top.

    Args:
        ranked_ids: The retriever's ranking, best first.
        relevant_ids: The labeled correct document ids for this query.

    Returns:
        A score in (0.0, 1.0], or 0.0 when no relevant document was ranked.
    """
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

:::terminal id="exercise-output-4" title="After Step 7: The Report" cmd="uv run python module_11_applications/src/main.py" maxw="920px" caption="Actual output. Read the columns: sparse sweeps keyword, dense sweeps paraphrase, both ace verbatim. Then read the overall row: recall@1 says sparse, recall@3 says dense, and MRR is a dead tie at 0.75."
<span class="header">=== Step 1: tokenize() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 2: inverse_document_frequency() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 3: tfidf_vector() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 4: cosine_similarity() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 5: rank_documents() ===</span> <span class="success">CORRECT</span>
<span class="header">=== Step 6: mean_pool() ===</span> <span class="success">CORRECT</span>
<span class="skipped">  ...</span>

<span class="header">=== Step 7: recall_at_k() and reciprocal_rank() ===</span> <span class="success">CORRECT</span>
    category       n    sparse (r@1   r@3   MRR)     dense (r@1   r@3   MRR)
    keyword       10         <span class="success">100%  100%  1.00</span>             <span class="t-fail">30%   60%  0.53</span>
    paraphrase    10          <span class="t-fail">10%   30%  0.25</span>             <span class="success">50%  100%  0.72</span>
    verbatim      10         100%  100%  1.00            100%  100%  1.00
    overall       30          70%   77%  <span class="t-cyan">0.75</span>             60%   87%  <span class="t-cyan">0.75</span>

  Chart saved to output/retrieval_comparison.png
  Read the category rows before believing the overall row.
  <span class="success">CORRECT</span>    recall@k: 'b' ranked 2nd is inside the top 2, so k=2 gives 1.0
  <span class="success">CORRECT</span>    recall@k: 'c' ranked 3rd is just past the cutoff, so k=2 gives 0.0
  <span class="success">CORRECT</span>    recall@k is a fraction: 1 of the 2 relevant ['b', 'z'] in the top 3 gives 0.5
  <span class="success">CORRECT</span>    reciprocal rank: first place scores 1/1 = 1.0
  <span class="success">CORRECT</span>    reciprocal rank: second place scores 1/2 = 0.5
  <span class="success">CORRECT</span>    reciprocal rank: fourth place scores 1/4 = 0.25
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
