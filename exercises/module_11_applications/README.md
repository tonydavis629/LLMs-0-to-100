# Module 11: Build Two Retrievers and Find Where Each One Wins

## Overview

Nothing here trains a model. The corpus is bundled (48 short support articles
for a fictional line of PX-series printers), along with 30 labeled queries and
a small pretrained sentence encoder. Your job is to build the two retrievers
the lecture describes and measure them against each other.

The **sparse retriever** is TF-IDF, built from scratch: score documents by word
overlap, weighting rare words up. The **dense retriever** swaps in embeddings
from the bundled MiniLM encoder and reuses your ranking code unchanged; you write
only the pooling that turns per-token vectors into one vector per document.

The payoff is the per-category table. Sparse retrieval is perfect on queries that
contain an exact identifier (error codes, part numbers) and nearly useless on
paraphrases; dense retrieval is the mirror image. The overall scores land close
enough that the category breakdown is the only way to tell the two apart. That
is the same lesson Module 9 taught about model benchmarks, now about retrievers.

## Setup

There is one shared environment for the whole repo. From the repo root:

```bash
uv sync
```

## Running

```bash
uv run python exercises/module_11_applications/src/main.py
```

The runner detects which steps you have implemented and skips the rest, so you
can fill in one function at a time and re-run immediately. The sparse retriever
comes alive after step 5, the dense retriever joins after step 6, and the report
appears after step 7. It prints two worked examples first (one keyword query, one
paraphrase query, top-3 results each) so you see the failure modes concretely
before any aggregate number, then the per-category table, and saves a grouped bar
chart to `output/retrieval_comparison.png`.

## What to implement

Open `exercise.py` and fill in each `raise NotImplementedError(...)` line. Each
needs only one expression or one short line.

| Step | Function | What it does |
|------|----------|--------------|
| 1 | `tokenize()` | Lowercase, strip punctuation, split into terms |
| 2 | `inverse_document_frequency()` | `log(N / df)` for every term in the corpus |
| 3 | `tfidf_vector()` | Term counts weighted by IDF, as one vector |
| 4 | `cosine_similarity()` | Dot product over the product of norms |
| 5 | `rank_documents()` | Score every document, return the top k |
| 6 | `mean_pool()` | Average the encoder's per-token vectors, ignoring padding |
| 7 | `recall_at_k()` and `reciprocal_rank()` | Score a ranking against the labels |

Steps 1&ndash;5 are a complete retriever on their own. Step 6 turns the same
ranking code into a second retriever, and step 7 makes the comparison
quantitative.

The corpus loader (`src/data.py`), the sentence encoder (`src/encoder.py`),
plotting (`src/visualization.py`), and the runner (`src/main.py`) are all
provided. You only edit `exercise.py`.

## Data

- `data/articles.jsonl`: the corpus, 48 support articles, each
  `{id, title, body}`. Ten error-code articles and six part-number articles carry
  exact identifiers (E-341, DR-4410); the rest are how-to and troubleshooting
  content written in manual vocabulary that user questions tend not to reuse.
- `data/queries.jsonl`: 30 labeled queries, each
  `{id, category, text, relevant_ids}`. Categories: `keyword` (the query contains
  an exact identifier), `paraphrase` (the query shares almost no vocabulary with
  its answer article), `overlap` (the query reuses the article's own wording).
- `data/encoder/`: all-MiniLM-L6-v2, a 23M-parameter sentence encoder
  stored in fp16 (~44MB). It runs on CPU and loads entirely from these local
  files; the exercise needs no network access. The weights are a copy of
  [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
  (Apache-2.0), committed here so nothing has to be downloaded.

## Extra credit

- **Hybrid search.** Combine the two rankings with reciprocal rank fusion: score
  each document by the sum of `1 / (60 + rank)` across both retrievers' rankings
  and re-sort. Check whether the fused ranking beats both retrievers overall;
  this is what production systems actually ship.
- **BM25's saturating term frequency.** Replace the raw count in `tfidf_vector()`
  with `count * (k1 + 1) / (count + k1)` for `k1 = 1.5`, so the tenth repetition
  of a term is worth less than the first. Compare against plain TF-IDF.
- **Watch IDF zero a term out.** Add the term "printer" to a query and confirm it
  changes almost nothing, then look up its IDF weight. A term in nearly every
  document carries almost no information about which one you want.
- **RAG prompt assembly.** Format the top retrieved article and the query into a
  grounded prompt using Module 6's chat template (system instructions, then the
  article as context, then the user question). This is the exact seam where the
  generator would attach; everything after it is Module 5's sampling loop.
- **Retrieve with the course's own model.** Embed the corpus with mean-pooled
  hidden states from the Module 5 checkpoint instead of MiniLM and measure how
  much retrieval quality drops. A model trained only on next-token prediction
  was never pushed to make similar texts nearby, and the gap is the contrastive
  objective's contribution.
