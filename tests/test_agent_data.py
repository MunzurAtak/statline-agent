from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from pydantic import BaseModel

from statline_agent.agent import AgentContext, build_graph
from statline_agent.agent.nodes.data import MAX_SQL_ATTEMPTS, SqlQuery
from statline_agent.agent.nodes.plan import QueryPlan
from statline_agent.agent.nodes.search import TableChoice
from statline_agent.catalog import CatalogIndex, HybridCatalogRetriever
from statline_agent.cbs import CBSClient, DataStore
from tests.fakes import FakeStructuredChatModel, prompt_text

PLAN = QueryPlan(
    measures=["M000352_3"],
    filters=[],
    frequency="jaar",
    year_from=0,
    year_to=0,
    latest_only=True,
)
GOOD_SQL = SqlQuery(
    sql=(
        "SELECT Perioden_title AS jaar, \"Value\" AS inwoners FROM data WHERE Measure = 'M000352_3'"
    ),
    explanation="Totale bevolking",
)
BAD_SQL = SqlQuery(sql="SELECT inwoners FROM data", explanation="kolom bestaat niet")


@pytest.fixture
def context(cbs_client: CBSClient) -> AgentContext:
    doc = Document(
        id="70072ned",
        page_content="Regionale kerncijfers Nederland\nKerncijfers per regio",
        metadata={"table_id": "70072ned", "title": "Regionale kerncijfers Nederland"},
    )
    embeddings = DeterministicFakeEmbedding(size=8)
    retriever = HybridCatalogRetriever(index=CatalogIndex.build([doc], embeddings), k=1)
    store = DataStore(cbs_client, ":memory:")
    return AgentContext(FakeStructuredChatModel(), retriever, cbs_client, embeddings, store)


def run(context: AgentContext, *sql_responses: SqlQuery) -> dict[str, Any]:
    assert isinstance(context.llm, FakeStructuredChatModel)
    responses: list[BaseModel] = [TableChoice(table_id="70072ned", reason="regio"), PLAN]
    context.llm.responses = [*responses, *sql_responses]
    state = build_graph().invoke(
        {"question": "Hoeveel inwoners?"}, context=context, interrupt_before=["compose_answer"]
    )
    return dict(state)


def test_happy_path_returns_query_result(context: AgentContext) -> None:
    state = run(context, GOOD_SQL)
    assert state["attempts"] == 1
    assert state["sql_error"] is None
    assert state["result_columns"] == ["jaar", "inwoners"]
    assert state["result_rows"] == [{"jaar": "1995", "inwoners": 598239.0}]
    assert "Tabel data" in state["data_overview"]


def test_failed_sql_is_corrected_with_the_error_message(context: AgentContext) -> None:
    state = run(context, BAD_SQL, GOOD_SQL)
    assert state["attempts"] == 2
    assert state["result_rows"]
    # The second SQL prompt contained the failed query and DuckDB's error message
    assert isinstance(context.llm, FakeStructuredChatModel)
    retry_prompt = prompt_text(context.llm.prompts[-1])
    assert "SELECT inwoners FROM data" in retry_prompt
    assert "inwoners" in retry_prompt and "Schrijf een verbeterde query" in retry_prompt


def test_gives_up_after_max_attempts(context: AgentContext) -> None:
    state = run(context, *[BAD_SQL] * MAX_SQL_ATTEMPTS)
    assert state["attempts"] == MAX_SQL_ATTEMPTS
    assert state["sql_error"]
    assert not state["result_rows"]


def test_empty_result_counts_as_error(context: AgentContext) -> None:
    empty = SqlQuery(sql="SELECT * FROM data WHERE 1 = 0", explanation="niets")
    state = run(context, empty, GOOD_SQL)
    assert state["attempts"] == 2
    assert state["result_rows"]
