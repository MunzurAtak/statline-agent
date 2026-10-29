from statline_agent.agent import AgentState, Candidate
from statline_agent.cli import format_update


def test_format_search_update_lists_candidates() -> None:
    update: AgentState = {
        "search_queries": ["bevolking"],
        "candidates": [Candidate(table_id="03759ned", title="Bevolking op 1 januari")],
    }
    lines = format_update("search_tables", update)
    assert lines[0] == "[search_tables]"
    assert any("03759ned" in line and "Bevolking" in line for line in lines)


def test_format_select_update_without_table() -> None:
    lines = format_update("select_table", {"table_id": None, "selection_reason": "geen match"})
    assert "geen geschikte tabel" in lines[1]
    assert lines[2].endswith("geen match")


def test_unknown_node_falls_back_to_key_values() -> None:
    assert format_update("other", {"question": "q"}) == ["[other]", "  question: q"]


def test_format_plan_update_shows_filters_and_notes() -> None:
    update: AgentState = {
        "measure_titles": ["Totale bevolking"],
        "filters": {"RegioS": ["GM0599"]},
        "plan_notes": ["Niet gevonden: X."],
    }
    lines = format_update("plan_query", update)
    assert "  filter RegioS: ['GM0599']" in lines
    assert lines[-1] == "  let op: Niet gevonden: X."
