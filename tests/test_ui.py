from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from streamlit.testing.v1 import AppTest

from statline_agent.agent import AgentContext, build_graph
from statline_agent.agent.memory import memory_checkpointer
from statline_agent.agent.nodes.answer import Answer
from statline_agent.agent.nodes.data import SqlQuery
from statline_agent.agent.nodes.plan import QueryPlan
from statline_agent.agent.nodes.search import TableChoice
from statline_agent.catalog import CatalogIndex, HybridCatalogRetriever
from statline_agent.cbs import CBSClient, DataStore
from statline_agent.charts import ChartSpec
from statline_agent.ui import app
from statline_agent.ui.turns import TurnResult, describe_step
from tests.fakes import FakeStructuredChatModel

APP_FILE = str(Path(__file__).parents[1] / "streamlit_app.py")


@pytest.fixture
def fake_agent(cbs_client: CBSClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the real agent (Groq + index) with the graph on recorded data and a fake LLM."""
    doc = Document(
        id="70072ned",
        page_content="Regionale kerncijfers Nederland\nKerncijfers per regio",
        metadata={"table_id": "70072ned", "title": "Regionale kerncijfers Nederland"},
    )
    embeddings = DeterministicFakeEmbedding(size=8)
    retriever = HybridCatalogRetriever(index=CatalogIndex.build([doc], embeddings), k=1)
    llm = FakeStructuredChatModel(
        responses=[
            TableChoice(table_id="70072ned", reason="regio"),
            QueryPlan(
                measures=["M000352_3"],
                filters=[],
                frequency="jaar",
                year_from=0,
                year_to=0,
                latest_only=True,
            ),
            SqlQuery(sql='SELECT Perioden_title AS jaar, "Value" AS n FROM data', explanation=""),
            Answer(
                text="Er wonen 598.239 mensen.",
                chart=ChartSpec(type="bar", x="jaar", y="n", color="", title="Bevolking"),
            ),
        ]
    )
    context = AgentContext(
        llm, retriever, cbs_client, embeddings, DataStore(cbs_client, ":memory:")
    )
    graph = build_graph(checkpointer=memory_checkpointer())
    monkeypatch.setattr(app, "load_agent", lambda: (graph, context))


def test_start_page_shows_examples_without_loading_the_agent() -> None:
    page = AppTest.from_file(APP_FILE).run()
    assert not page.exception
    assert page.title[0].value == "Vraag het CBS"
    assert [b.label for b in page.button if b.label in app.EXAMPLES] == app.EXAMPLES


def test_example_question_gives_answer_chart_and_explanation(fake_agent: None) -> None:
    page = AppTest.from_file(APP_FILE).run()
    next(b for b in page.button if b.label == app.EXAMPLES[0]).click().run()

    assert not page.exception
    assert any("598.239" in m.value for m in page.markdown)
    assert page.expander[0].label == "Hoe kwam ik hieraan?"
    assert any("70072ned" in c.value for c in page.caption)


def test_describe_step_summarizes_updates() -> None:
    assert describe_step("search_tables", {"candidates": []}) == (
        "Tabellen zoeken in de CBS-catalogus: 0 kandidaten"
    )
    failed = describe_step("run_sql", {"attempts": 1, "sql_error": "x"})
    assert failed == "Analyse uitvoeren: poging 1 mislukt, opnieuw proberen"
    assert describe_step("unknown", {}) == "unknown"


def test_turn_result_from_state() -> None:
    turn = TurnResult.from_state("vraag", {"answer": "a", "sql": "SELECT 1", "attempts": 2})
    assert (turn.answer, turn.sql, turn.attempts, turn.chart) == ("a", "SELECT 1", 2, None)
