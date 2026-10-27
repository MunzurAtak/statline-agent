"""Measure catalog search quality for vector, BM25 and hybrid search.

Usage: uv run python evals/run_retrieval_eval.py
Writes a summary to evals/results/retrieval.json (committed, used in the README).
"""

import json
from pathlib import Path

from statline_agent.catalog import CatalogIndex, FastEmbedEmbeddings, HybridCatalogRetriever
from statline_agent.config import get_settings
from statline_agent.evaluation import RetrievalReport, evaluate, load_cases

EVALS_DIR = Path(__file__).parent
TOP_K = 10


def print_report(report: RetrievalReport) -> None:
    s = report.summary()
    print(f"{s['method']:10} hit@1={s['hit@1']:.2f}  hit@5={s['hit@5']:.2f}  mrr={s['mrr']:.2f}")


def main() -> None:
    settings = get_settings()
    embeddings = FastEmbedEmbeddings(settings.embedding_model)
    index = CatalogIndex.load(settings.catalog_index_dir, embeddings)
    retriever = HybridCatalogRetriever(index=index, k=TOP_K)
    cases = load_cases(EVALS_DIR / "retrieval_cases.jsonl")

    reports = [
        evaluate("vector", cases, lambda q: retriever.vector_ranking(q)[:TOP_K]),
        evaluate("bm25", cases, lambda q: retriever.keyword_ranking(q)[:TOP_K]),
        evaluate("hybrid", cases, lambda q: [str(d.id) for d in retriever.invoke(q)]),
    ]

    print(f"Retrieval eval on {len(cases)} questions ({len(index)} tables in index)\n")
    for report in reports:
        print_report(report)

    print("\nMisses for hybrid (no acceptable table in top 5):")
    for result in reports[-1].results:
        if not result.hit_at(5):
            print(f"  - {result.case.question}  (first hit at: {result.first_hit_rank})")

    out = EVALS_DIR / "results" / "retrieval.json"
    out.parent.mkdir(exist_ok=True)
    payload = {
        "cases": len(cases),
        "index_size": len(index),
        "reports": [report.summary() for report in reports],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved summary to {out}")


if __name__ == "__main__":
    main()
