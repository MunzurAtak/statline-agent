import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from statline_agent.catalog import (
    CatalogIndex,
    HybridCatalogRetriever,
    reciprocal_rank_fusion,
    tokenize,
)


def doc(table_id: str, text: str) -> Document:
    return Document(id=table_id, page_content=text, metadata={"table_id": table_id})


@pytest.fixture
def retriever() -> HybridCatalogRetriever:
    docs = [
        doc("A", "Bevolking; geslacht, leeftijd en burgerlijke staat"),
        doc("B", "Zorg en welzijn; werkzame personen en banen"),
        doc("C", "Woningvoorraad; eigendom, type verhuurder"),
        doc("D", "Beloning van in zorg werkzame artsen"),
    ]
    index = CatalogIndex.build(docs, DeterministicFakeEmbedding(size=32))
    return HybridCatalogRetriever(index=index, k=3, candidates=4)


def test_tokenize_removes_stopwords_and_punctuation() -> None:
    assert tokenize("Hoeveel mensen werken in de zorg?") == ["mensen", "werken", "zorg"]


def test_rrf_rewards_items_ranked_high_by_both_methods() -> None:
    fused = reciprocal_rank_fusion([["x", "y", "z"], ["y", "z", "x"]], k=60)
    assert [item_id for item_id, _ in fused] == ["y", "x", "z"]


def test_rrf_includes_items_from_only_one_list() -> None:
    fused = dict(reciprocal_rank_fusion([["a"], ["b"]]))
    assert set(fused) == {"a", "b"}


def test_keyword_ranking_prefers_exact_terms(retriever: HybridCatalogRetriever) -> None:
    assert retriever.keyword_ranking("werkzame personen zorg")[0] == "B"


def test_keyword_ranking_empty_for_stopword_query(retriever: HybridCatalogRetriever) -> None:
    assert retriever.keyword_ranking("hoeveel is het") == []


def test_invoke_returns_documents_with_ranks(retriever: HybridCatalogRetriever) -> None:
    results = retriever.invoke("werkzame personen zorg")
    assert len(results) == 3
    top = results[0].metadata
    assert top["keyword_rank"] is not None
    assert {"score", "vector_rank", "keyword_rank", "table_id"} <= set(top)
