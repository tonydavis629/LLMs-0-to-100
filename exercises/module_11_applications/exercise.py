"""
Module 11 Exercise: Build two retrievers and find where each one wins

You implement the eight functions below. Everything else (the corpus of support
articles, the labeled queries, the pretrained sentence encoder, the plotting, and
the runner) is provided. Each blank is one line or one short expression.

Two retrievers, one corpus. The sparse retriever (steps 1-5) scores documents by
weighted word overlap, TF-IDF, built here from scratch. The dense retriever reuses
steps 4 and 5 unchanged and swaps in embeddings from a small pretrained encoder;
you write the pooling that turns per-token vectors into one vector per document
(step 6). Step 7 scores both retrievers on the same labeled queries, per category.
The question the report answers: which retriever is better, and better at what?

Run after each step; unfinished steps are skipped automatically:
    uv run python module_11_applications/src/main.py
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np


# ---------------------------------------------------------------------------
# Step 1: Tokenization
# ---------------------------------------------------------------------------


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
    # HINT: .split() with no argument splits on any run of whitespace and
    #       drops the empty pieces.
    raise NotImplementedError("TODO: split the cleaned text into terms")


# ---------------------------------------------------------------------------
# Step 2: Inverse document frequency
# ---------------------------------------------------------------------------


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
    # HINT: a dict comprehension over document_frequency.items(); math.log.
    raise NotImplementedError("TODO: compute the IDF weight for each term")


# ---------------------------------------------------------------------------
# Step 3: TF-IDF document vectors
# ---------------------------------------------------------------------------


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
        # HINT: vocab_index[term] is the slot; idf[term] is the weight.
        raise NotImplementedError("TODO: fill in the term's TF-IDF entry")
    return vector


# ---------------------------------------------------------------------------
# Step 4: Cosine similarity
# ---------------------------------------------------------------------------


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
    # HINT: np.dot(a, b) is the dot product; wrap the result in float().
    raise NotImplementedError("TODO: compute the cosine similarity")


# ---------------------------------------------------------------------------
# Step 5: Ranking
# ---------------------------------------------------------------------------


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
    # HINT: sorted(range(len(scores)), key=..., reverse=True) sorts document
    #       indices by their score; slice the first k.
    raise NotImplementedError("TODO: rank the documents and keep the top k")


# ---------------------------------------------------------------------------
# Step 6: Mean pooling (the dense retriever's vectorizer)
# ---------------------------------------------------------------------------


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
    # HINT: (token_vectors * mask).sum(axis=0) sums the real token vectors;
    #       mask.sum() counts the real tokens.
    raise NotImplementedError("TODO: average the real token vectors")


# ---------------------------------------------------------------------------
# Step 7: Scoring a retriever (recall@k and mean reciprocal rank)
# ---------------------------------------------------------------------------


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
    # HINT: sum(1 for ...) counts matches against ranked_ids[:k]; divide by
    #       len(relevant_ids).
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
            # HINT: position is already 1-based thanks to enumerate's start=1.
            raise NotImplementedError("TODO: return the reciprocal rank")
    return 0.0

