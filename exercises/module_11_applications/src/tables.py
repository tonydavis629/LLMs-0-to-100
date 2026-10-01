"""Printing the worked examples and the score table, provided for you.

You do NOT need to edit this file. It only formats results that
`src/main.py` has already computed with your functions.
"""

from __future__ import annotations

# The query categories, in the order the table and chart show them
CATEGORY_ORDER = ["keyword", "paraphrase", "verbatim"]


def print_worked_example(query: dict, results_by_retriever: dict) -> None:
    """One query, each retriever's top results, with the labeled answer marked.

    `results_by_retriever` maps a retriever's name ("sparse" or "dense") to
    its ranked list of (article, score) pairs.
    """
    print(f"  [{query['category']}] {query['text']!r}")
    print(f"  labeled answer: {query['relevant_ids'][0]}")
    for name, results in results_by_retriever.items():
        print(f"    {name}:")
        for rank, (article, score) in enumerate(results, start=1):
            marker = "[relevant]" if article["id"] in query["relevant_ids"] else ""
            line = (f"      {rank}. {score:5.3f}  {article['id']:<16}"
                    f"{article['title'][:38]:<40}{marker}")
            print(line.rstrip())


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def aggregate(per_query: list[dict], category: str | None) -> tuple[float, float, float]:
    """Average recall@1, recall@3 and reciprocal rank over one category (None = all)."""
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
            r1, r3, rr = aggregate(scores[name], category)
            row += f"{r1:>13.0%}{r3:>6.0%}{rr:>6.2f}   "
        print(row.rstrip())
