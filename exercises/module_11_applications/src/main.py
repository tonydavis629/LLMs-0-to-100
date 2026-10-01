"""
Module 11 Exercise runner: two retrievers over one support corpus

Run with:
    uv run python module_11_applications/src/main.py

Each step below runs one function you write in exercise.py on the real
corpus of 48 support articles, then tests it. Read top to bottom, the steps
build two search engines and then compare them:

    1-3. sparse index: text -> terms -> IDF weights -> one TF-IDF vector per article
    4-5. ranking: cosine similarity between a query and every article, keep the top k
    6.   dense index: MiniLM token vectors -> mean pooling -> one embedding per article
    7.   scoring: recall@k and reciprocal rank on 30 labeled queries, per category

Add --step N to run one step (1-7).
Add --solution to run the finished answers from solution/exercise.py.
"""

from __future__ import annotations

import argparse
import sys
from functools import cache
from pathlib import Path

import numpy as np

# Make the module root (parent of src/) importable so we can `from exercise import ...`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# With --solution, swap in solution/exercise.py before anything imports `exercise`
from src.solution import use_solution_if_requested

use_solution_if_requested()

from exercise import (
    cosine_similarity,
    inverse_document_frequency,
    mean_pool,
    rank_documents,
    recall_at_k,
    reciprocal_rank,
    tfidf_vector,
    tokenize,
)
from src.data import DATA_DIR, article_text, load_jsonl
from src.encoder import SentenceEncoder
from src.prerequisites import needs, probe_own_blank
from src.reporting import run_step
from src.tables import CATEGORY_ORDER, aggregate, print_report, print_worked_example
from src.visualization import save_category_comparison

# One test file per step lives in tests/
from tests.test_step1_tokenize import check_tokenize
from tests.test_step2_idf import check_inverse_document_frequency
from tests.test_step3_tfidf import check_tfidf_vector
from tests.test_step4_cosine import check_cosine_similarity
from tests.test_step5_rank import check_rank_documents
from tests.test_step6_mean_pool import check_mean_pool
from tests.test_step7_metrics import check_recall_at_k, check_reciprocal_rank

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

TOP_K = 3                           # results shown and scored at k=3
EXAMPLE_QUERY_IDS = ["q01", "q19"]  # one keyword query, one paraphrase query

# The corpus (each article is {id, title, body}) and the labeled queries
# (each query is {id, category, text, relevant_ids})
ARTICLES = load_jsonl(DATA_DIR / "articles.jsonl")
QUERIES = load_jsonl(DATA_DIR / "queries.jsonl")
# How many queries fall in each category, e.g. {"keyword": 10, ...}
COUNTS = {category: sum(1 for q in QUERIES if q["category"] == category)
          for category in CATEGORY_ORDER}


# ---------------------------------------------------------------------------
# The two retrievers. Each index is a vector per article plus a function that
# turns a query into a vector the same way. Ranking and scoring are shared.
# ---------------------------------------------------------------------------


@cache  # build once, then reuse the same index in Steps 3-7
def sparse_index() -> dict:
    """TF-IDF vectors for every article, plus a query vectorizer (Steps 1-3).

    Tokenize every article, compute IDF over the corpus, fix a vocabulary
    order, and vectorize each article. The query vectorizer reuses the same
    IDF table and vocabulary, because a query MUST be vectorized with the
    corpus statistics, not its own.
    """
    needs([1, 2, 3], "build the TF-IDF index")
    doc_tokens = [tokenize(article_text(article)) for article in ARTICLES]
    idf = inverse_document_frequency(doc_tokens)
    vocab_index = {term: slot for slot, term in enumerate(sorted(idf))}  # term -> vector slot
    doc_vectors = np.stack([tfidf_vector(tokens, idf, vocab_index) for tokens in doc_tokens])

    def query_vectorizer(query: str) -> np.ndarray:
        return tfidf_vector(tokenize(query), idf, vocab_index)

    return {"doc_vectors": doc_vectors, "query_vectorizer": query_vectorizer,
            "doc_tokens": doc_tokens, "idf": idf, "vocab_index": vocab_index}


@cache  # load once (about a second on CPU), then reuse
def load_encoder() -> SentenceEncoder:
    """The bundled MiniLM encoder: text in, one 384-number vector per token out."""
    return SentenceEncoder(DATA_DIR / "encoder")


@cache  # build once, then reuse the same index in Steps 6-7
def dense_index() -> dict:
    """Mean-pooled MiniLM embeddings for every article, plus a query vectorizer (Step 6).

    The encoder emits one vector per token, and mean_pool turns each
    article's token vectors into one vector. Queries go through the exact
    same encoder and pooling.
    """
    needs([6], "pool the encoder's token vectors into article vectors")
    encoder = load_encoder()
    encoded = encoder.encode([article_text(article) for article in ARTICLES])
    doc_vectors = np.stack([mean_pool(vectors, mask) for vectors, mask in encoded])

    def query_vectorizer(query: str) -> np.ndarray:
        (vectors, mask), = encoder.encode([query])
        return mean_pool(vectors, mask)

    return {"doc_vectors": doc_vectors, "query_vectorizer": query_vectorizer}


def retrieve(query: str, index: dict, k: int) -> list[tuple[dict, float]]:
    """Vectorize a query, rank every article against it, return the top k.

    This one function IS both retrievers: only the index differs between
    sparse and dense. Returns (article, score) pairs, best first.
    """
    query_vector = index["query_vectorizer"](query)
    top = rank_documents(query_vector, index["doc_vectors"], k)
    return [(ARTICLES[i], cosine_similarity(query_vector, index["doc_vectors"][i])) for i in top]


def show_worked_examples(indexes: dict) -> None:
    """Run the two example queries through each retriever and print the top 3."""
    for number, query_id in enumerate(EXAMPLE_QUERY_IDS):
        if number > 0:
            print()
        query = next(q for q in QUERIES if q["id"] == query_id)
        print_worked_example(query, {name: retrieve(query["text"], index, TOP_K)
                                     for name, index in indexes.items()})


def score_retriever(index: dict) -> list[dict]:
    """Recall@1, recall@3 and reciprocal rank for every labeled query.

    The ranking covers ALL articles (k = corpus size) because reciprocal rank
    needs to know where the right article landed even when it missed the top 3.
    """
    per_query = []
    for query in QUERIES:
        ranked_ids = [article["id"] for article, _ in retrieve(query["text"], index, len(ARTICLES))]
        per_query.append({
            "category": query["category"],
            "recall1": recall_at_k(ranked_ids, query["relevant_ids"], 1),
            "recall3": recall_at_k(ranked_ids, query["relevant_ids"], TOP_K),
            "rr": reciprocal_rank(ranked_ids, query["relevant_ids"]),
        })
    return per_query


# ---------------------------------------------------------------------------
# Step 1: text -> terms
# ---------------------------------------------------------------------------


def step_1() -> str:
    def show():
        print(f"Corpus: {len(ARTICLES)} support articles for fictional PX-series printers")
        print(f"Queries: {len(QUERIES)} labeled ({COUNTS['keyword']} keyword, "
              f"{COUNTS['paraphrase']} paraphrase, {COUNTS['verbatim']} verbatim)")
        # One article title and one query, term by term
        for text in ("Error E-341: fuser temperature fault", "my pages come out crumpled"):
            print(f"  {text!r}")
            print(f"    -> {tokenize(text)}")
        # Then every article, counted
        all_tokens = [tokenize(article_text(article)) for article in ARTICLES]
        n_terms = sum(len(tokens) for tokens in all_tokens)
        n_distinct = len({term for tokens in all_tokens for term in tokens})
        print(f"  All {len(ARTICLES)} articles: {n_terms} terms in total, {n_distinct} distinct")

    return run_step("Step 1: tokenize()", show, lambda: check_tokenize(tokenize))


# ---------------------------------------------------------------------------
# Step 2: IDF weights, from rare terms to terms in every article
# ---------------------------------------------------------------------------


def step_2() -> str:
    def show():
        probe_own_blank(2)
        needs([1], "split the articles into terms")
        doc_tokens = [tokenize(article_text(article)) for article in ARTICLES]
        idf = inverse_document_frequency(doc_tokens)
        doc_sets = [set(tokens) for tokens in doc_tokens]  # the distinct terms of each article
        print(f"IDF over {len(doc_tokens)} articles, {len(idf)} distinct terms. Rarer terms weigh more:")
        print(f"    {'term':<10}{'in docs':>8}{'idf':>8}")
        for term in ("341", "fuser", "printer", "the"):
            in_docs = sum(term in terms for terms in doc_sets)
            weight = idf.get(term)
            shown = f"{weight:>8.3f}" if isinstance(weight, (int, float)) else f"{'missing':>8}"
            print(f"    {term:<10}{in_docs:>8}{shown}")
        print("  'crumpled' is in no article, so it has no weight and no query can match on it")

    return run_step("Step 2: inverse_document_frequency()", show,
                    lambda: check_inverse_document_frequency(inverse_document_frequency))


# ---------------------------------------------------------------------------
# Step 3: one TF-IDF vector per article
# ---------------------------------------------------------------------------


def step_3() -> str:
    def show():
        probe_own_blank(3)
        index = sparse_index()
        doc_vectors, vocab_index = index["doc_vectors"], index["vocab_index"]
        vocab_size = len(vocab_index)
        nonzero = int(np.mean((doc_vectors != 0).sum(axis=1)))
        print(f"Indexed {len(doc_vectors)} articles: one vector each, with one")
        print(f"dimension per vocabulary term ({vocab_size} terms). On average only")
        print(f"{nonzero} of the {vocab_size} entries are nonzero, which is why these")
        print("vectors are called sparse.")

        # The biggest entries of one article's vector, with the count x IDF behind each
        row = [article["id"] for article in ARTICLES].index("err-e341")
        slot_to_term = {slot: term for term, slot in vocab_index.items()}
        print("  Largest entries in the vector for err-e341:")
        for slot in np.argsort(-doc_vectors[row])[:3]:
            term = slot_to_term[int(slot)]
            count = index["doc_tokens"][row].count(term)
            print(f"    {term:<8}{count} x {index['idf'][term]:.3f} = {doc_vectors[row][slot]:5.2f}")

    return run_step("Step 3: tfidf_vector()", show, lambda: check_tfidf_vector(tfidf_vector))


# ---------------------------------------------------------------------------
# Step 4: cosine similarity between article vectors
# ---------------------------------------------------------------------------


def step_4() -> str:
    def show():
        probe_own_blank(4)
        index = sparse_index()
        doc_vectors, ids = index["doc_vectors"], [article["id"] for article in ARTICLES]

        # All 1128 pairs of the 48 articles, most similar first
        pairs = [(cosine_similarity(doc_vectors[i], doc_vectors[j]), ids[i], ids[j])
                 for i in range(len(ids)) for j in range(i + 1, len(ids))]
        pairs.sort(reverse=True)
        print(f"Most similar of the {len(pairs)} article pairs, by TF-IDF cosine:")
        for score, first, second in pairs[:3]:
            print(f"    {score:5.3f}  {first} and {second}")

        # Pasting a text twice doubles every count, so the vector gets longer
        # but keeps its direction
        row = ids.index("err-e341")
        doubled = tfidf_vector(index["doc_tokens"][row] * 2, index["idf"], index["vocab_index"])
        print(f"  err-e341 against its own text pasted twice: "
              f"{cosine_similarity(doc_vectors[row], doubled):.3f} (length does not count)")

    return run_step("Step 4: cosine_similarity()", show,
                    lambda: check_cosine_similarity(cosine_similarity))


# ---------------------------------------------------------------------------
# Step 5: ranking. The sparse retriever is complete.
# ---------------------------------------------------------------------------


def step_5() -> str:
    def show():
        probe_own_blank(5)
        needs([4], "score each document")
        show_worked_examples({"sparse": sparse_index()})

    return run_step("Step 5: rank_documents()", show, lambda: check_rank_documents(rank_documents))


# ---------------------------------------------------------------------------
# Step 6: mean pooling. The dense retriever, ranked by the same code.
# ---------------------------------------------------------------------------


def step_6() -> str:
    def show():
        probe_own_blank(6)
        print(f"Encoding {len(ARTICLES)} articles with the bundled MiniLM (CPU, a few seconds)...")
        doc_vectors = dense_index()["doc_vectors"]
        nonzero = int(np.mean((doc_vectors != 0).sum(axis=1)))
        print(f"Indexed {len(doc_vectors)} articles: one {doc_vectors.shape[1]}-dimensional embedding each. On average")
        print(f"{nonzero} of the {doc_vectors.shape[1]} entries are nonzero: dense, where TF-IDF was sparse.")

        # The ranking code from Steps 4-5, unchanged, with the new vectors
        needs([4, 5], "rank the articles for the worked examples")
        show_worked_examples({"dense": dense_index()})

    return run_step("Step 6: mean_pool()", show,
                    lambda: check_mean_pool(mean_pool, load_encoder()))


# ---------------------------------------------------------------------------
# Step 7: score both retrievers on all 30 labeled queries, per category
# ---------------------------------------------------------------------------


def step_7() -> str:
    def show():
        probe_own_blank(7)
        needs([1, 2, 3, 4, 5, 6], "have both retrievers' rankings to score")
        scores = {"sparse": score_retriever(sparse_index()),
                  "dense": score_retriever(dense_index())}
        print_report(scores, COUNTS)

        # recall@3 (the middle number from aggregate) per category, plus overall
        save_category_comparison(
            CATEGORY_ORDER,
            [aggregate(scores["sparse"], c)[1] for c in CATEGORY_ORDER],
            [aggregate(scores["dense"], c)[1] for c in CATEGORY_ORDER],
            OUTPUT_DIR / "retrieval_comparison.png",
            aggregate(scores["sparse"], None)[1],
            aggregate(scores["dense"], None)[1],
        )
        print()
        print("  Chart saved to output/retrieval_comparison.png")
        print("  Read the category rows before believing the overall row.")

    return run_step("Step 7: recall_at_k() and reciprocal_rank()", show,
                    lambda: check_recall_at_k(recall_at_k) + check_reciprocal_rank(reciprocal_rank))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

STEPS = {"1": step_1, "2": step_2, "3": step_3, "4": step_4,
         "5": step_5, "6": step_6, "7": step_7}


def main() -> None:
    parser = argparse.ArgumentParser(description="Two retrievers over one support corpus")
    parser.add_argument("--step", choices=[*STEPS, "all"], default="all",
                        help="Which step to run (default: all)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    for number, step in STEPS.items():
        if args.step in ("all", number):
            step()


if __name__ == "__main__":
    main()
