from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from statline_agent.agent import AgentContext, Candidate, build_graph
from statline_agent.agent.nodes.search import TableChoice, format_candidates
from statline_agent.catalog import CatalogIndex, HybridCatalogRetriever
from tests.fakes import FakeStructuredChatModel, prompt_text


def table(table_id: str, title: str, measures: list[str]) -> Document:
    return Document(
        id=table_id,
        page_content=f"{title}\nBeschrijving van {title}",
        metadata={
            "table_id": table_id,
            "title": title,
            "measures": measures,
            "dimensions": ["Regio's", "Perioden"],
            "modified": "2026-07-31",
        },
    )


@pytest.fixture
def retriever() -> HybridCatalogRetriever:
    docs = [
        table("03759ned", "Bevolking op 1 januari; regio", ["Bevolking op 1 januari"]),
        table("85036NED", "Gemiddelde WOZ-waarde van woningen", ["Gemiddelde WOZ-waarde"]),
        table("82242NED", "Faillissementen; kerncijfers", ["Uitgesproken faillissementen"]),
    ]
    index = CatalogIndex.build(docs, DeterministicFakeEmbedding(size=32))
    return HybridCatalogRetriever(index=index, k=3, candidates=3)


def run(
    retriever: HybridCatalogRetriever, llm: FakeStructuredChatModel, question: str
) -> dict[str, Any]:
    graph = build_graph()
    return dict(graph.invoke({"question": question}, context=AgentContext(llm, retriever)))


def test_graph_selects_table_chosen_by_llm(retriever: HybridCatalogRetriever) -> None:
    llm = FakeStructuredChatModel(responses=[TableChoice(table_id="85036ned", reason="WOZ")])
    state = run(retriever, llm, "gemiddelde WOZ-waarde woningen")

    assert len(state["candidates"]) == 3
    assert state["table_id"] == "85036NED"  # matched case-insensitively
    assert state["table_title"] == "Gemiddelde WOZ-waarde van woningen"
    assert state["selection_reason"] == "WOZ"
    # The LLM saw the question and the candidates, including their measures
    sent = prompt_text(llm.prompts[0])
    assert "gemiddelde WOZ-waarde woningen" in sent
    assert "Onderwerpen: Uitgesproken faillissementen" in sent


def test_none_means_no_suitable_table(retriever: HybridCatalogRetriever) -> None:
    llm = FakeStructuredChatModel(responses=[TableChoice(table_id="NONE", reason="Geen match")])
    state = run(retriever, llm, "wat is de hoofdstad van Frankrijk")
    assert state["table_id"] is None
    assert state["selection_reason"] == "Geen match"


def test_unknown_id_falls_back_to_best_search_result(retriever: HybridCatalogRetriever) -> None:
    llm = FakeStructuredChatModel(responses=[TableChoice(table_id="99999XYZ", reason="?")])
    state = run(retriever, llm, "faillissementen")
    assert state["table_id"] == state["candidates"][0].table_id
    assert "99999XYZ" in state["selection_reason"]


def test_format_candidates_is_compact() -> None:
    candidate = Candidate(
        table_id="T1", title="Titel", measures=[f"m{i}" for i in range(30)], dimensions=["Regio's"]
    )
    text = format_candidates([candidate])
    assert text.startswith("- T1: Titel")
    assert "m11" in text and "m12" not in text  # at most 12 measures shown
