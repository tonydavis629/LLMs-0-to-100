"""
Module 11 Exercise runner: two retrievers over one support corpus

Run with:
    uv run python module_11_applications/src/main.py

Builds a TF-IDF index and an embedding index over the same 48 support articles,
runs 30 labeled queries through both retrievers, prints two worked examples (one
keyword query, one paraphrase query) so the failure modes are visible before any
aggregate, then prints recall@1, recall@3, and MRR per query category, and saves
a grouped bar chart.

Any step in exercise.py that still raises NotImplementedError is detected and
skipped, so you can implement one function at a time and re-run immediately.
The sparse retriever works after step 5; the dense retriever joins after step 6;
the report appears after step 7.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make the module root (parent of src/) importable so we can `from exercise import ...`,
# and src/ importable for the provided data / encoder / plotting helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exercise import (  # noqa: E402  (import after sys.path edits)
    tokenize,
    inverse_document_frequency,
    tfidf_vector,
    cosine_similarity,
    rank_documents,
    mean_pool,
    recall_at_k,
    reciprocal_rank,
)
from data import load_jsonl, article_text  # noqa: E402
from encoder import SentenceEncoder  # noqa: E402
from visualization import save_category_comparison  # noqa: E402


TOP_K = 3                                          # results shown and scored at k=3
CATEGORY_ORDER = ["keyword", "paraphrase", "verbatim"]
EXAMPLE_QUERY_IDS = ["q01", "q19"]                 # one keyword, one paraphrase

_THIS_DIR = Path(__file__).resolve().parent
_OUTPUT_DIR = _THIS_DIR.parent / "output"


def _find_data_dir() -> Path:
    """Locate the module's data/ directory (the solution runs one level deeper)."""
    for parent in _THIS_DIR.parents:
        candidate = parent / "data"
        if (candidate / "articles.jsonl").exists():
            return candidate
    raise FileNotFoundError("Could not locate the data/ directory")


_DATA_DIR = _find_data_dir()


def _is_implemented(fn, *args, **kwargs) -> bool:
    try:
        fn(*args, **kwargs)
        return True
    except NotImplementedError:
        return False
    except Exception:
        # Any other exception still means the student wrote *something*.
        return True


def _probe_steps() -> dict[str, bool]:
    """Detect which exercise.py steps are implemented, using throwaway inputs."""
    tiny_vec = np.array([1.0, 2.0])
    tiny_tokens = np.array([[1.0, 2.0], [3.0, 4.0]])
    return {
        "tokenize": _is_implemented(tokenize, "Error E-341."),
        "inverse_document_frequency": _is_implemented(
            inverse_document_frequency, [["fuser", "fault"], ["fuser"]]),
        "tfidf_vector": _is_implemented(
            tfidf_vector, ["fuser"], {"fuser": 1.0}, {"fuser": 0}),
        "cosine_similarity": _is_implemented(cosine_similarity, tiny_vec, tiny_vec),
        "rank_documents": _is_implemented(
            rank_documents, tiny_vec, np.stack([tiny_vec, tiny_vec]), 1),
        "mean_pool": _is_implemented(mean_pool, tiny_tokens, np.array([1.0, 1.0])),
        "recall_at_k": _is_implemented(recall_at_k, ["a", "b"], ["a"], 1),
        "reciprocal_rank": _is_implemented(reciprocal_rank, ["a", "b"], ["b"]),
    }


def _heading(title: str) -> None:
    print("=" * 68)
    print(title)
    print("=" * 68)


# ---------------------------------------------------------------------------
# Building the two indexes (each one is: a vector per article, plus a way to
# vectorize a query the same way). Ranking and scoring are shared.
# ---------------------------------------------------------------------------


def build_sparse_index(articles: list[dict]):
    """TF-IDF vectors for every article, plus a query vectorizer.

    Uses steps 1-3: tokenize every article, compute IDF over the corpus, fix a
    vocabulary order, and vectorize each article. The returned query_vectorizer
    closes over the same IDF table and vocabulary, because a query MUST be
    vectorized with the corpus statistics, not its own.
    """
    doc_tokens = [tokenize(article_text(article)) for article in articles]
    idf = inverse_document_frequency(doc_tokens)
    vocab_index = {term: slot for slot, term in enumerate(sorted(idf))}
    doc_vectors = np.stack([tfidf_vector(tokens, idf, vocab_index)
                            for tokens in doc_tokens])

    def query_vectorizer(query: str) -> np.ndarray:
        return tfidf_vector(tokenize(query), idf, vocab_index)

    return doc_vectors, query_vectorizer, len(vocab_index)


def build_dense_index(articles: list[dict], encoder: SentenceEncoder):
    """Mean-pooled MiniLM embeddings for every article, plus a query vectorizer.

    Uses step 6: the encoder (provided plumbing) emits per-token vectors, and
    mean_pool turns each article's tokens into one 384-dimensional vector.
    Queries go through the exact same encoder and pooling.
    """
    encoded = encoder.encode([article_text(article) for article in articles])
    doc_vectors = np.stack([mean_pool(vectors, mask) for vectors, mask in encoded])

    def query_vectorizer(query: str) -> np.ndarray:
        (vectors, mask), = encoder.encode([query])
        return mean_pool(vectors, mask)

    return doc_vectors, query_vectorizer


def retrieve(query: str, query_vectorizer, doc_vectors, articles, k):
    """Vectorize a query, rank every article against it, return the top k.

    This one function IS both retrievers: only query_vectorizer/doc_vectors
    differ between sparse and dense. Returns (article, score) pairs.
    """
    query_vector = query_vectorizer(query)
    top = rank_documents(query_vector, doc_vectors, k)
    return [(articles[i], cosine_similarity(query_vector, doc_vectors[i]))
            for i in top]


# ---------------------------------------------------------------------------
# The worked examples and the report
# ---------------------------------------------------------------------------


def print_worked_example(query: dict, retrievers: dict, articles: list[dict]) -> None:
    """One query, each retriever's top 3, with the labeled answer marked."""
    print(f"  [{query['category']}] {query['text']!r}")
    print(f"  labeled answer: {query['relevant_ids'][0]}")
    for name, (doc_vectors, query_vectorizer) in retrievers.items():
        results = retrieve(query["text"], query_vectorizer, doc_vectors,
                           articles, TOP_K)
        print(f"    {name}:")
        for rank, (article, score) in enumerate(results, start=1):
            marker = "[relevant]" if article["id"] in query["relevant_ids"] else ""
            print(f"      {rank}. {score:5.3f}  {article['id']:<16}"
                  f"{article['title'][:38]:<40}{marker}")
    print()


def score_retriever(queries, doc_vectors, query_vectorizer, articles):
    """Full ranking per query -> recall@1, recall@3, reciprocal rank per query.

    The ranking is computed over ALL articles (k = corpus size) because
    reciprocal rank needs to know where the right article landed even when it
    missed the top 3.
    """
    per_query = []
    for query in queries:
        ranked = retrieve(query["text"], query_vectorizer, doc_vectors,
                          articles, len(articles))
        ranked_ids = [article["id"] for article, _ in ranked]
        per_query.append({
            "category": query["category"],
            "recall1": recall_at_k(ranked_ids, query["relevant_ids"], 1),
            "recall3": recall_at_k(ranked_ids, query["relevant_ids"], TOP_K),
            "rr": reciprocal_rank(ranked_ids, query["relevant_ids"]),
        })
    return per_query


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _aggregate(per_query: list[dict], category: str | None) -> tuple[float, float, float]:
    rows = [r for r in per_query if category is None or r["category"] == category]
    return (_mean([r["recall1"] for r in rows]),
            _mean([r["recall3"] for r in rows]),
            _mean([r["rr"] for r in rows]))


def print_report(scores: dict, counts: dict[str, int]) -> None:
    """The per-category table. The overall row comes last, and never alone."""
    names = list(scores)
    header_left = f"    {'category':<12}{'n':>4}"
    print(header_left + "".join(f"{name + ' (r@1   r@3   MRR)':>28}" for name in names))
    for category in CATEGORY_ORDER + [None]:
        label = category if category else "overall"
        count = counts[category] if category else sum(counts.values())
        row = f"    {label:<12}{count:>4}"
        for name in names:
            r1, r3, rr = _aggregate(scores[name], category)
            row += f"{r1:>13.0%}{r3:>6.0%}{rr:>6.2f}   "
        print(row)
    print()


# ---------------------------------------------------------------------------


def main() -> None:
    articles = load_jsonl(_DATA_DIR / "articles.jsonl")
    queries = load_jsonl(_DATA_DIR / "queries.jsonl")
    counts = {cat: sum(1 for q in queries if q["category"] == cat)
              for cat in CATEGORY_ORDER}
    steps = _probe_steps()

    _heading("MODULE 11: two retrievers over one support corpus")
    print("Corpus and queries")
    print(f"  corpus            {len(articles)} support articles"
          f" (fictional PX-series printers)")
    print(f"  queries           {len(queries)} labeled queries:"
          f" {counts['keyword']} keyword, {counts['paraphrase']} paraphrase,"
          f" {counts['verbatim']} verbatim")
    print(f"  sparse retriever  TF-IDF vectors, built from scratch (steps 1-5)")
    print(f"  dense retriever   MiniLM embeddings, 384 dims, mean-pooled (step 6)")
    print(f"  scoring           recall@1, recall@{TOP_K}, MRR, per category (step 7)")
    print()

    # ------------------------------------------------------------------
    # 1. Build the sparse index (steps 1-3).
    # ------------------------------------------------------------------
    _heading("1. SPARSE INDEX (TF-IDF)")
    sparse_ready = (steps["tokenize"] and steps["inverse_document_frequency"]
                    and steps["tfidf_vector"])
    ranking_ready = steps["cosine_similarity"] and steps["rank_documents"]
    retrievers: dict[str, tuple] = {}
    if sparse_ready:
        doc_vectors, query_vectorizer, vocab_size = build_sparse_index(articles)
        nonzero = int(np.mean((doc_vectors != 0).sum(axis=1)))
        print(f"  Indexed {len(articles)} articles: one vector each, with one")
        print(f"  dimension per vocabulary term ({vocab_size} terms). On average only")
        print(f"  {nonzero} of the {vocab_size} entries are nonzero, which is why these")
        print(f"  vectors are called sparse.")
        if ranking_ready:
            retrievers["sparse"] = (doc_vectors, query_vectorizer)
        else:
            print("  [ranking needs cosine_similarity() and rank_documents(),"
                  " steps 4-5]")
    else:
        print("  [skipped: implement tokenize(), inverse_document_frequency(),")
        print("   and tfidf_vector(), steps 1-3]")
    print()

    # ------------------------------------------------------------------
    # 2. Build the dense index (step 6; the encoder itself is provided).
    # ------------------------------------------------------------------
    _heading("2. DENSE INDEX (MiniLM EMBEDDINGS)")
    if steps["mean_pool"]:
        print(f"  Encoding {len(articles)} articles with the bundled MiniLM"
              f" (CPU, a few seconds)...")
        encoder = SentenceEncoder(_DATA_DIR / "encoder")
        doc_vectors, query_vectorizer = build_dense_index(articles, encoder)
        print(f"  Indexed {len(articles)} articles: one 384-dimensional embedding")
        print(f"  each, every entry nonzero. Same corpus, entirely different geometry.")
        if ranking_ready:
            retrievers["dense"] = (doc_vectors, query_vectorizer)
        else:
            print("  [ranking needs cosine_similarity() and rank_documents(),"
                  " steps 4-5]")
    else:
        print("  [skipped: implement mean_pool(), step 6]")
    print()

    # ------------------------------------------------------------------
    # 3. Two queries in full, before any aggregate number.
    # ------------------------------------------------------------------
    _heading("3. WORKED EXAMPLES")
    if retrievers:
        for query_id in EXAMPLE_QUERY_IDS:
            query = next(q for q in queries if q["id"] == query_id)
            print_worked_example(query, retrievers, articles)
    else:
        print("  [skipped: no retriever is runnable yet]")
        print()

    # ------------------------------------------------------------------
    # 4. The report: every query, every retriever, per category.
    # ------------------------------------------------------------------
    _heading("4. THE REPORT")
    metrics_ready = steps["recall_at_k"] and steps["reciprocal_rank"]
    if retrievers and metrics_ready:
        scores = {name: score_retriever(queries, doc_vectors, query_vectorizer,
                                        articles)
                  for name, (doc_vectors, query_vectorizer) in retrievers.items()}
        print_report(scores, counts)
        if len(scores) == 2:
            sparse_by_cat = [_aggregate(scores["sparse"], c)[1] for c in CATEGORY_ORDER]
            dense_by_cat = [_aggregate(scores["dense"], c)[1] for c in CATEGORY_ORDER]
            out_img = _OUTPUT_DIR / "retrieval_comparison.png"
            save_category_comparison(
                CATEGORY_ORDER, sparse_by_cat, dense_by_cat, out_img,
                _aggregate(scores["sparse"], None)[1],
                _aggregate(scores["dense"], None)[1],
            )
            print(f"  Chart saved to {out_img}")
            print("  Read the category rows before believing the overall row.")
    elif retrievers:
        print("  [skipped: implement recall_at_k() and reciprocal_rank(), step 7]")
    else:
        print("  [skipped: no retriever is runnable yet]")
    print()

    _heading("Done")
    print("Run after each step; unfinished steps are skipped automatically.")


if __name__ == "__main__":
    main()

