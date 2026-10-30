from dataclasses import replace
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from statline_agent.agent import AgentContext, build_graph
from statline_agent.agent.memory import memory_checkpointer
from statline_agent.agent.nodes.search import TableChoice
from statline_agent.catalog import CatalogIndex, HybridCatalogRetriever
from statline_agent.cbs import CBSClient, DataStore
from tests.fakes import FakeStructuredChatModel


def table(table_id: str, title: str) -> Document:
    return Document(
        id=table_id, page_content=title, metadata={"table_id": table_id, "title": title}
    )


@pytest.fixture
def context(cbs_client: CBSClient) -> AgentContext:
    docs = [
        table("03759ned", "Bevolking op 1 januari"),
        table("70072ned", "Regionale kerncijfers"),
        table("37259ned", "Bevolkingsontwikkeling"),
    ]
    embeddings = DeterministicFakeEmbedding(size=8)
    retriever = HybridCatalogRetriever(index=CatalogIndex.build(docs, embeddings), k=3)
    llm = FakeStructuredChatModel(responses=[TableChoice(table_id="03759ned", reason="past")])
    store = DataStore(cbs_client, ":memory:")
    return AgentContext(llm, retriever, cbs_client, embeddings, store, confirm_tables=True)


class Session:
    """Stops before plan_query, so only the table selection is exercised."""

    def __init__(self, context: AgentContext) -> None:
        self.context = context
        self.graph = build_graph(checkpointer=memory_checkpointer())
        self.config: RunnableConfig = {"configurable": {"thread_id": "t"}}

    def run(self, payload: Any) -> dict[str, Any]:
        state = self.graph.invoke(
            payload, self.config, context=self.context, interrupt_before=["plan_query"]
        )
        return dict(state)


def test_graph_pauses_with_proposal_and_alternatives(context: AgentContext) -> None:
    state = Session(context).run({"question": "Hoeveel inwoners?"})
    request = state["__interrupt__"][0].value
    assert request["proposed"]["table_id"] == "03759ned"
    assert request["proposed"]["reason"] == "past"
    assert {a["table_id"] for a in request["alternatives"]} == {"70072ned", "37259ned"}


@pytest.mark.parametrize(
    ("answer", "expected_table"),
    [("", "03759ned"), ("70072NED", "70072ned"), ("onbekend", "03759ned")],
)
def test_resume_accepts_or_changes_the_table(
    context: AgentContext, answer: str, expected_table: str
) -> None:
    session = Session(context)
    session.run({"question": "Hoeveel inwoners?"})
    state = session.run(Command(resume=answer))
    assert state["table_id"] == expected_table


def test_rejecting_all_tables_ends_with_explanation(context: AgentContext) -> None:
    session = Session(context)
    session.run({"question": "Hoeveel inwoners?"})
    state = session.run(Command(resume="NONE"))
    assert state["table_id"] is None
    assert state["answer"].startswith("Ik heb geen CBS-tabel gevonden")


def test_auto_mode_does_not_pause(context: AgentContext) -> None:
    state = Session(replace(context, confirm_tables=False)).run({"question": "Hoeveel inwoners?"})
    assert "__interrupt__" not in state
    assert state["table_id"] == "03759ned"
