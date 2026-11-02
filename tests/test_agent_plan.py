from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from statline_agent.agent import AgentContext, build_graph
from statline_agent.agent.nodes.plan import (
    DimensionChoice,
    QueryPlan,
    measure_labels,
    shortlist_measures,
)
from statline_agent.agent.nodes.search import TableChoice
from statline_agent.catalog import CatalogIndex, HybridCatalogRetriever
from statline_agent.cbs import CBSClient, DataStore, Measure, MeasureGroup
from tests import conftest
from tests.fakes import FakeStructuredChatModel, prompt_text

# Uses the recorded 70072ned fixtures: 15 measures, RegioS codes NL01..PV29, Perioden 1995-2009.


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


def run(context: AgentContext, plan: QueryPlan) -> dict[str, Any]:
    assert isinstance(context.llm, FakeStructuredChatModel)
    context.llm.responses = [TableChoice(table_id="70072ned", reason="regio"), plan]
    state = build_graph().invoke(
        {"question": "Bevolking Zuid-Holland"}, context=context, interrupt_before=["fetch_data"]
    )
    return dict(state)


def make_plan(**overrides: Any) -> QueryPlan:
    values: dict[str, Any] = {
        "measures": ["M000352_3"],
        "filters": [],
        "frequency": "jaar",
        "year_from": 0,
        "year_to": 0,
        "latest_only": True,
    }
    return QueryPlan(**{**values, **overrides})


def test_plan_resolves_names_and_period_range(context: AgentContext) -> None:
    plan = make_plan(
        filters=[DimensionChoice(dimension="Regio's", values=["Zuid-Holland"])],
        year_from=2005,
        year_to=2009,
        latest_only=False,
    )
    state = run(context, plan)
    assert state["filters"] == {
        "Measure": ["M000352_3"],
        "RegioS": ["PV28"],
        "Perioden": ["2005JJ00", "2006JJ00", "2007JJ00", "2008JJ00", "2009JJ00"],
    }
    assert state["measure_titles"] == ["Bevolking > Totale bevolking"]
    assert state["measure_units"] == ["aantal"]
    assert state["requested_filters"] == {"Regio's": ["Zuid-Holland"]}
    assert state["plan_notes"] == []


def test_unmentioned_dimensions_get_total_and_latest_period(context: AgentContext) -> None:
    state = run(context, make_plan())
    assert state["filters"]["RegioS"] == ["NL01"]
    assert state["filters"]["Perioden"] == ["2009JJ00"]


def test_problems_are_reported_as_notes(context: AgentContext) -> None:
    plan = make_plan(
        measures=["BESTAAT_NIET"],
        filters=[DimensionChoice(dimension="Regio's", values=["Atlantis"])],
        year_from=2030,
        latest_only=False,
    )
    state = run(context, plan)
    assert state["filters"]["Measure"] == ["M000352_3"]  # falls back to the first measure
    assert state["filters"]["RegioS"] == ["NL01"]
    assert state["filters"]["Perioden"] == ["2009JJ00"]
    assert len(state["plan_notes"]) == 3


def test_prompt_lists_measure_codes_and_small_dimensions(context: AgentContext) -> None:
    run(context, make_plan())
    assert isinstance(context.llm, FakeStructuredChatModel)
    sent = prompt_text(context.llm.prompts[-1])
    assert "Bevolking > Totale bevolking (M000352_3) [aantal]" in sent
    # The group makes clear that 'Nederland' is an origin, not the country
    assert "Bevolking > Herkomst > Nederland (X007245) [aantal]" in sent
    assert "Perioden: 1995 t/m 2009" in sent


def test_plan_works_without_measure_groups(
    context: AgentContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    routes = {k: v for k, v in conftest.ROUTES.items() if not k.endswith("MeasureGroups")}
    monkeypatch.setattr(conftest, "ROUTES", routes)  # CBS answers 404
    state = run(context, make_plan())
    assert state["measure_titles"] == ["Totale bevolking"]


def test_measure_labels_follow_nested_groups() -> None:
    groups = [
        MeasureGroup(id="G1", title="Bevolking naar herkomst"),
        MeasureGroup(id="G2", title="Land van herkomst", parent_id="G1"),
        MeasureGroup(id="G3", title="Totale groei"),  # same title as its measure: skipped
    ]
    measures = [
        Measure(identifier="M1", title="Nederland", measure_group_id="G2"),
        Measure(identifier="M2", title="Totale groei", measure_group_id="G3"),
        Measure(identifier="M3", title="Zonder groep"),
        Measure(identifier="M4", title="Onbekende groep", measure_group_id="X"),
    ]
    assert measure_labels(measures, groups) == {
        "M1": "Bevolking naar herkomst > Land van herkomst > Nederland",
        "M2": "Totale groei",
        "M3": "Zonder groep",
        "M4": "Onbekende groep",
    }


def test_measure_labels_survive_a_cycle_in_groups() -> None:
    groups = [
        MeasureGroup(id="A", title="A", parent_id="B"),
        MeasureGroup(id="B", title="B", parent_id="A"),
    ]
    labels = measure_labels([Measure(identifier="M", title="m", measure_group_id="A")], groups)
    assert labels["M"].endswith("A > m")
    # Group titles equal to the measure title are skipped; the loop must still end
    same = [MeasureGroup(id="A", title="m", parent_id="A")]
    assert measure_labels([Measure(identifier="M", title="m", measure_group_id="A")], same) == {
        "M": "m"
    }


def test_shortlist_keeps_small_lists_and_trims_large_ones() -> None:
    measures = [Measure(identifier=f"M{i}", title=f"Onderwerp {i}") for i in range(40)]
    embeddings = DeterministicFakeEmbedding(size=8)
    assert len(shortlist_measures("vraag", measures[:10], embeddings)) == 10
    short = shortlist_measures("vraag", measures, embeddings, k=5)
    assert len(short) == 5
    assert [m.identifier for m in short] == sorted(
        (m.identifier for m in short), key=lambda x: int(x[1:])
    )
