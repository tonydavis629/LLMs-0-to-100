"""
Module 11 Exercise runner: two retrievers over one support corpus

Run with:
    uv run python module_11_applications/src/main.py

Builds a TF-IDF index and an embedding index over the same 48 support articles,
runs two worked example queries through each retriever, then scores both on 30
labeled queries per category and saves a grouped bar chart.

Every step is tagged on its header line, then its output follows: what your
code produced on the real corpus and the result of each test in tests/.
The tags are:

    CORRECT     every test for the step passed
    INCORRECT   your code ran but at least one test failed (details follow)
    INCOMPLETE  the function still raises NotImplementedError

Add --step N to run one step (1-7).
Add --solution to run the finished answers from solution/exercise.py.
"""

from __future__ import annotations

import argparse
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

# Make the module root (parent of src/) importable so we can `from exercise import ...`,
# and src/ importable for the provided data / encoder / plotting helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# `--solution` swaps in the finished answers from solution/exercise.py.
# Registering it as "exercise" before the imports below means every
# `from exercise import ...` in this file picks it up with no other change.
if "--solution" in sys.argv:
    import importlib.util

    sys.argv.remove("--solution")
    _sol = Path(__file__).resolve().parent.parent / "solution" / "exercise.py"
    _spec = importlib.util.spec_from_file_location("exercise", _sol)
    _exercise = importlib.util.module_from_spec(_spec)
    sys.modules["exercise"] = _exercise
    _spec.loader.exec_module(_exercise)

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

# One test file per step lives in tests/
from tests.test_step1_tokenize import check_tokenize  # noqa: E402
from tests.test_step2_idf import check_inverse_document_frequency  # noqa: E402
from tests.test_step3_tfidf import check_tfidf_vector  # noqa: E402
from tests.test_step4_cosine import check_cosine_similarity  # noqa: E402
from tests.test_step5_rank import check_rank_documents, known_good_cosine  # noqa: E402
from tests.test_step6_mean_pool import check_mean_pool  # noqa: E402
from tests.test_step7_metrics import check_recall_at_k, check_reciprocal_rank  # noqa: E402


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


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

# The three possible outcomes for a step
CORRECT = "CORRECT"
INCORRECT = "INCORRECT"
INCOMPLETE = "INCOMPLETE"

# ANSI color codes, used only when printing to a real terminal
_COLORS = {CORRECT: "\033[32m", INCORRECT: "\033[31m", INCOMPLETE: "\033[90m"}
_RESET = "\033[0m"


def _tag(status: str) -> str:
    """Format a status label in a fixed-width column, colored on a terminal."""
    label = f"{status:<10}"
    if sys.stdout.isatty():
        return f"{_COLORS[status]}{label}{_RESET}"
    return label


def _print_checks(checks) -> None:
    """Print one line per test, with details under any that failed."""
    for check in checks:
        print(f"  {_tag(CORRECT if check.passed else INCORRECT)} {check.name}")
        if not check.passed and check.detail:
            for line in check.detail.split("\n"):
                print(f"             {line.strip()}")


def run_step(title: str, show, check) -> str:
    """Run one step and print its header, tag, output, and test results.

    `show()` prints whatever the student's code produces (training progress,
    saved plots). `check()` returns the list of Check results for the step.

    The tag goes on the header line, so the output is captured first and
    printed after the tag is known. Returns CORRECT, INCORRECT, or INCOMPLETE.
    """
    buffer = io.StringIO()
    checks = []
    note = ""
    try:
        with redirect_stdout(buffer):
            show()
        checks = check()
        status = CORRECT if all(c.passed for c in checks) else INCORRECT
    except NotImplementedError as e:
        # The student has not filled in this blank yet
        status, note = INCOMPLETE, str(e)
    except Exception as e:  # noqa: BLE001 - show students any crash, whatever its type
        status, note = INCORRECT, f"your code crashed: {type(e).__name__}: {e}"

    print(f"=== {title} === {_tag(status).rstrip()}")
    if note:
        print(f"  {note}")
    output = buffer.getvalue()
    if output and status != INCOMPLETE:
        print(output, end="" if output.endswith("\n") else "\n")
    _print_checks(checks)
    print()
    return status


# ---------------------------------------------------------------------------
# Earlier steps a demo depends on
# ---------------------------------------------------------------------------

# One tiny call per step. If it raises NotImplementedError, that step is unfinished.
_PROBES = {
    1: ("tokenize", lambda: tokenize("a b")),
    2: ("inverse_document_frequency", lambda: inverse_document_frequency([["a"]])),
    3: ("tfidf_vector", lambda: tfidf_vector(["a"], {"a": 1.0}, {"a": 0})),
    4: ("cosine_similarity", lambda: cosine_similarity(np.ones(2), np.ones(2))),
    5: ("rank_documents", lambda: rank_documents(np.ones(2), np.ones((2, 2)), 1)),
    6: ("mean_pool", lambda: mean_pool(np.ones((2, 2)), np.ones(2))),
}


def needs(steps: list[int], purpose: str) -> None:
    """Stop a demo with a pointed message if an earlier step is unfinished.

    Without this, a demo that calls an unfinished earlier function would show
    that function's TODO message under the wrong step.
    """
    for number in steps:
        name, probe = _PROBES[number]
        try:
            probe()
        except NotImplementedError:
            raise NotImplementedError(f"needs Step {number} ({name}) to {purpose}") from None
        except Exception:  # noqa: BLE001
            pass  # it runs; whether it is right is for that step's own tests to say


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

    return {"doc_vectors": doc_vectors, "query_vectorizer": query_vectorizer,
            "doc_tokens": doc_tokens, "idf": idf, "vocab_index": vocab_index}


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

    return {"doc_vectors": doc_vectors, "query_vectorizer": query_vectorizer}


def sparse_index(articles: list[dict], results: dict):
    """Build the TF-IDF index once (Steps 1-3) and share it with later steps."""
    if "sparse" not in results:
        needs([1, 2, 3], "build the TF-IDF index")
        results["sparse"] = build_sparse_index(articles)
    return results["sparse"]


def load_encoder(results: dict) -> SentenceEncoder:
    """Load the bundled MiniLM once (about a second on CPU) and share it."""
    if "encoder" not in results:
        results["encoder"] = SentenceEncoder(_DATA_DIR / "encoder")
    return results["encoder"]


def dense_index(articles: list[dict], results: dict):
    """Build the embedding index once (Step 6) and share it with later steps."""
    if "dense" not in results:
        needs([6], "pool the encoder's token vectors into article vectors")
        results["dense"] = build_dense_index(articles, load_encoder(results))
    return results["dense"]


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
            line = (f"      {rank}. {score:5.3f}  {article['id']:<16}"
                    f"{article['title'][:38]:<40}{marker}")
            print(line.rstrip())


def print_worked_examples(queries: list[dict], retrievers: dict, articles: list[dict]) -> None:
    """The two example queries (one keyword, one paraphrase), a blank line apart."""
    for number, query_id in enumerate(EXAMPLE_QUERY_IDS):
        if number > 0:
            print()
        query = next(q for q in queries if q["id"] == query_id)
        print_worked_example(query, retrievers, articles)


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
        print(row.rstrip())


# ---------------------------------------------------------------------------
# The steps
# ---------------------------------------------------------------------------


def step_1(data: dict, results: dict) -> str:
    """Tokenize an article title and a query, then the whole corpus."""

    def show():
        articles, counts = data["articles"], data["counts"]
        print(f"Corpus: {len(articles)} support articles for fictional PX-series printers")
        print(f"Queries: {len(data['queries'])} labeled ({counts['keyword']} keyword, "
              f"{counts['paraphrase']} paraphrase, {counts['verbatim']} verbatim)")
        # One article title and one query, term by term
        for text in ("Error E-341: fuser temperature fault", "my pages come out crumpled"):
            print(f"  {text!r}")
            print(f"    -> {tokenize(text)}")
        # Then every article, counted
        all_tokens = [tokenize(article_text(article)) for article in articles]
        n_terms = sum(len(tokens) for tokens in all_tokens)
        n_distinct = len({term for tokens in all_tokens for term in tokens})
        print(f"  All {len(articles)} articles: {n_terms} terms in total, {n_distinct} distinct")

    return run_step("Step 1: tokenize()", show, lambda: check_tokenize(tokenize))


def step_2(data: dict, results: dict) -> str:
    """IDF over the real corpus, for a few terms from rare to everywhere."""

    def show():
        # Call this step's own function first, so an unfinished blank reports its own TODO
        inverse_document_frequency([["probe"]])
        needs([1], "split the articles into terms")

        doc_tokens = [tokenize(article_text(article)) for article in data["articles"]]
        idf = inverse_document_frequency(doc_tokens)
        doc_sets = [set(tokens) for tokens in doc_tokens]
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


def step_3(data: dict, results: dict) -> str:
    """Vectorize every article; show how sparse the vectors are."""

    def show():
        # Call this step's own function first, so an unfinished blank reports its own TODO
        tfidf_vector(["probe"], {"probe": 1.0}, {"probe": 0})
        index = sparse_index(data["articles"], results)
        doc_vectors, vocab_index = index["doc_vectors"], index["vocab_index"]
        vocab_size = len(vocab_index)
        nonzero = int(np.mean((doc_vectors != 0).sum(axis=1)))
        print(f"Indexed {len(doc_vectors)} articles: one vector each, with one")
        print(f"dimension per vocabulary term ({vocab_size} terms). On average only")
        print(f"{nonzero} of the {vocab_size} entries are nonzero, which is why these")
        print("vectors are called sparse.")

        # The biggest entries of one article's vector, with the count x IDF behind each
        row = [article["id"] for article in data["articles"]].index("err-e341")
        slot_to_term = {slot: term for term, slot in vocab_index.items()}
        print("  Largest entries in the vector for err-e341:")
        for slot in np.argsort(-doc_vectors[row])[:3]:
            term = slot_to_term[int(slot)]
            count = index["doc_tokens"][row].count(term)
            print(f"    {term:<8}{count} x {index['idf'][term]:.3f} = {doc_vectors[row][slot]:5.2f}")

    return run_step("Step 3: tfidf_vector()", show, lambda: check_tfidf_vector(tfidf_vector))


def step_4(data: dict, results: dict) -> str:
    """Compare every pair of articles by the cosine of their TF-IDF vectors."""

    def show():
        # Call this step's own function first, so an unfinished blank reports its own TODO
        cosine_similarity(np.ones(2), np.ones(2))
        index = sparse_index(data["articles"], results)
        doc_vectors, ids = index["doc_vectors"], [article["id"] for article in data["articles"]]

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


def step_5(data: dict, results: dict) -> str:
    """The sparse retriever is complete: run the two worked examples through it."""

    def show():
        # Call this step's own function first, so an unfinished blank reports its own TODO.
        # rank_documents() calls cosine_similarity() from Step 4, so a known-good copy
        # stands in for it during this one call.
        with known_good_cosine(rank_documents):
            rank_documents(np.ones(2), np.ones((2, 2)), 1)
        needs([4], "score each document")
        index = sparse_index(data["articles"], results)
        retrievers = {"sparse": (index["doc_vectors"], index["query_vectorizer"])}
        print_worked_examples(data["queries"], retrievers, data["articles"])

    return run_step("Step 5: rank_documents()", show, lambda: check_rank_documents(rank_documents))


def step_6(data: dict, results: dict) -> str:
    """Build the dense index with mean pooling; run the same two queries through it."""

    def show():
        # Call this step's own function first, so an unfinished blank reports its own TODO
        mean_pool(np.ones((2, 2)), np.ones(2))
        articles = data["articles"]
        print(f"Encoding {len(articles)} articles with the bundled MiniLM (CPU, a few seconds)...")
        index = dense_index(articles, results)
        doc_vectors = index["doc_vectors"]
        nonzero = int(np.mean((doc_vectors != 0).sum(axis=1)))
        print(f"Indexed {len(doc_vectors)} articles: one {doc_vectors.shape[1]}-dimensional embedding each. On average")
        print(f"{nonzero} of the {doc_vectors.shape[1]} entries are nonzero: dense, where TF-IDF was sparse.")

        # The ranking code from Steps 4-5, unchanged, with the new vectors
        needs([4, 5], "rank the articles for the worked examples")
        retrievers = {"dense": (index["doc_vectors"], index["query_vectorizer"])}
        print_worked_examples(data["queries"], retrievers, articles)

    return run_step("Step 6: mean_pool()", show,
                    lambda: check_mean_pool(mean_pool, load_encoder(results)))


def step_7(data: dict, results: dict) -> str:
    """Score both retrievers on all 30 labeled queries, per category."""

    def show():
        # Call this step's own functions first, so an unfinished blank reports its own TODO
        recall_at_k(["probe"], ["probe"], 1)
        reciprocal_rank(["probe"], ["probe"])
        needs([1, 2, 3, 4, 5, 6], "have both retrievers' rankings to score")

        articles, queries = data["articles"], data["queries"]
        indexes = {"sparse": sparse_index(articles, results),
                   "dense": dense_index(articles, results)}
        scores = {name: score_retriever(queries, index["doc_vectors"],
                                        index["query_vectorizer"], articles)
                  for name, index in indexes.items()}
        print_report(scores, data["counts"])

        # recall@3 per category, plus overall, as a grouped bar chart
        save_category_comparison(
            CATEGORY_ORDER,
            [_aggregate(scores["sparse"], c)[1] for c in CATEGORY_ORDER],
            [_aggregate(scores["dense"], c)[1] for c in CATEGORY_ORDER],
            _OUTPUT_DIR / "retrieval_comparison.png",
            _aggregate(scores["sparse"], None)[1],
            _aggregate(scores["dense"], None)[1],
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
    steps = list(STEPS) if args.step == "all" else [args.step]

    _OUTPUT_DIR.mkdir(exist_ok=True)
    queries = load_jsonl(_DATA_DIR / "queries.jsonl")
    data = {
        "articles": load_jsonl(_DATA_DIR / "articles.jsonl"),
        "queries": queries,
        "counts": {category: sum(1 for q in queries if q["category"] == category)
                   for category in CATEGORY_ORDER},
    }
    results: dict = {}  # the two indexes and the encoder, built once and shared between steps

    for step in steps:
        STEPS[step](data, results)


if __name__ == "__main__":
    main()
# ---------------------------------------------------------------------------
