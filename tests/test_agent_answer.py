from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from pydantic import BaseModel

from statline_agent.agent import AgentContext, build_graph
from statline_agent.agent.nodes.answer import Answer, failure_message, format_result
from statline_agent.agent.nodes.data import MAX_SQL_ATTEMPTS, SqlQuery
from statline_agent.agent.nodes.plan import QueryPlan
from statline_agent.agent.nodes.search import TableChoice
from statline_agent.catalog import CatalogIndex, HybridCatalogRetriever
from statline_agent.cbs import CBSClient, DataStore
from statline_agent.charts import ChartSpec
from tests.fakes import FakeStructuredChatModel, prompt_text

PLAN = QueryPlan(
    measures=["M000352_3"], filters=[], frequency="jaar", year_from=0, year_to=0, latest_only=True
)
SQL = SqlQuery(
    sql=(
        "SELECT Perioden_title AS jaar, \"Value\" AS inwoners FROM data WHERE Measure = 'M000352_3'"
    ),
    explanation="Totale bevolking per jaar",
)
LINE_CHART = ChartSpec(type="line", x="jaar", y="inwoners", color="", title="Bevolking")


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


def run(context: AgentContext, *responses: BaseModel) -> dict[str, Any]:
    assert isinstance(context.llm, FakeStructuredChatModel)
    context.llm.responses = list(responses)
    return dict(build_graph().invoke({"question": "Hoeveel inwoners?"}, context=context))


def test_full_run_produces_answer_chart_and_source(context: AgentContext) -> None:
    answer = Answer(text="In 1995 waren er 598.239 inwoners.", chart=LINE_CHART)
    state = run(context, TableChoice(table_id="70072ned", reason="r"), PLAN, SQL, answer)

    assert state["answer"] == "In 1995 waren er 598.239 inwoners."
    assert state["chart"] == LINE_CHART
    assert state["source"].table_id == "70072ned"
    assert state["source"].url == "https://dataportal.cbs.nl/detail/CBS/70072ned"
    # The answer prompt contained the actual query result
    assert isinstance(context.llm, FakeStructuredChatModel)
    sent = prompt_text(context.llm.prompts[-1])
    assert "jaar | inwoners" in sent and "1995 | 598239.00" in sent


def test_chart_with_unknown_columns_is_dropped(context: AgentContext) -> None:
    bad_chart = ChartSpec(type="bar", x="regio", y="inwoners", color="", title="?")
    answer = Answer(text="...", chart=bad_chart)
    state = run(context, TableChoice(table_id="70072ned", reason="r"), PLAN, SQL, answer)
    assert state["chart"] is None


def test_no_table_gives_fixed_answer_without_llm_call(context: AgentContext) -> None:
    state = run(context, TableChoice(table_id="NONE", reason="Gaat niet over CBS-cijfers."))
    assert state["answer"].startswith("Ik heb geen CBS-tabel gevonden")
    assert "Gaat niet over CBS-cijfers." in state["answer"]
    assert state["source"] is None


def test_failed_sql_gives_fixed_answer(context: AgentContext) -> None:
    bad = SqlQuery(sql="SELECT kapot FROM data", explanation="")
    responses = [TableChoice(table_id="70072ned", reason="r"), PLAN, *[bad] * MAX_SQL_ATTEMPTS]
    state = run(context, *responses)
    assert state["answer"].startswith("Het is me niet gelukt")
    assert state["chart"] is None


def test_failure_message_is_none_when_there_is_a_result() -> None:
    assert failure_message({"table_id": "T", "result_rows": [{"a": 1}]}) is None


def test_format_result_rounds_floats_and_truncates() -> None:
    rows = [{"jaar": str(1900 + i), "waarde": i / 3} for i in range(60)]
    text = format_result(["jaar", "waarde"], rows)
    lines = text.splitlines()
    assert lines[0] == "jaar | waarde"
    assert lines[2] == "1901 | 0.33"
    assert lines[-1] == "... (10 rijen weggelaten)"
