"""Compare vector, keyword and hybrid search for a question.

Usage: uv run python scripts/search_catalog.py "hoeveel mensen werken in de zorg"
"""

import sys

from statline_agent.catalog import CatalogIndex, FastEmbedEmbeddings, HybridCatalogRetriever
from statline_agent.config import get_settings


def main() -> None:
    query = " ".join(sys.argv[1:]) or "hoeveel mensen werken in de zorg"
    settings = get_settings()
    index = CatalogIndex.load(
        settings.catalog_index_dir, FastEmbedEmbeddings(settings.embedding_model)
    )
    retriever = HybridCatalogRetriever(index=index, k=5)
    titles = {str(doc.id): doc.metadata["title"] for doc in index.documents}

    print(f"Query: {query}\n")
    for name, ranking in [
        ("Vector", retriever.vector_ranking(query)[:5]),
        ("Keyword (BM25)", retriever.keyword_ranking(query)[:5]),
    ]:
        print(f"== {name}")
        for table_id in ranking:
            print(f"  {table_id:10} {titles[table_id]}")
    print("== Hybrid (RRF)")
    for doc in retriever.invoke(query):
        m = doc.metadata
        ranks = f"vector #{m['vector_rank']}, bm25 #{m['keyword_rank']}"
        print(f"  {m['table_id']:10} {m['title']}  ({ranks})")


if __name__ == "__main__":
    main()
