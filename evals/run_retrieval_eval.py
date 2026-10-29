"""Measure catalog search quality for vector, BM25, hybrid and hybrid + LLM query rewriting.

Usage: uv run python evals/run_retrieval_eval.py
Writes a summary to evals/results/retrieval.json (committed, used in the README).
The rewrite variant needs an LLM (GROQ_API_KEY in .env) and is skipped otherwise.
"""

import json
from pathlib import Path

from statline_agent.catalog import (
    CatalogIndex,
    FastEmbedEmbeddings,
    HybridCatalogRetriever,
    QueryRewriter,
)
from statline_agent.config import get_settings
from statline_agent.evaluation import RetrievalReport, evaluate, load_cases
from statline_agent.llm import MissingAPIKeyError, get_chat_model

EVALS_DIR = Path(__file__).parent
TOP_K = 10


def print_report(report: RetrievalReport) -> None:
    s = report.summary()
    print(f"{s['method']:15} hit@1={s['hit@1']:.2f}  hit@5={s['hit@5']:.2f}  mrr={s['mrr']:.2f}")


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

    rewrites: dict[str, list[str]] = {}
    try:
        rewriter = QueryRewriter.from_llm(get_chat_model(settings))

        def rewrite_and_remember(question: str) -> list[str]:
            rewrites[question] = rewriter(question)
            return rewrites[question]

        with_rewrite = retriever.model_copy(update={"rewriter": rewrite_and_remember})
        reports.append(
            evaluate("hybrid+rewrite", cases, lambda q: [str(d.id) for d in with_rewrite.invoke(q)])
        )
    except MissingAPIKeyError:
        print("No LLM configured: skipping hybrid+rewrite\n")

    print(f"Retrieval eval on {len(cases)} questions ({len(index)} tables in index)\n")
    for report in reports:
        print_report(report)

    best = reports[-1]
    print(f"\nMisses for {best.method} (no acceptable table in top 5):")
    for result in best.results:
        if not result.hit_at(5):
            question = result.case.question
            print(f"  - {question}  (first hit at: {result.first_hit_rank})")
            if question in rewrites:
                print(f"      rewrites: {rewrites[question]}")

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
