from pathlib import Path

from statline_agent.agent import AgentState, Candidate
from statline_agent.agent.state import Source
from statline_agent.charts import ChartSpec
from statline_agent.cli import format_update, save_chart


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


def test_format_run_sql_shows_error_or_rows() -> None:
    failed = format_update("run_sql", {"attempts": 1, "sql_error": "kolom bestaat niet"})
    assert failed[1] == "  fout bij poging 1: kolom bestaat niet"
    ok = format_update("run_sql", {"attempts": 2, "result_rows": [{"jaar": "2024"}]})
    assert ok[1:] == ["  1 rijen (poging 2)", "  {'jaar': '2024'}"]


def test_format_answer_shows_text_and_source() -> None:
    source = Source(table_id="T1", title="Titel", url="https://example.org/T1")
    lines = format_update("compose_answer", {"answer": "Het antwoord.", "source": source})
    assert "Het antwoord." in lines
    assert lines[-1] == "Bron: CBS, Titel (T1) - https://example.org/T1"


def test_save_chart_writes_html(tmp_path: Path) -> None:
    state: AgentState = {
        "chart": ChartSpec(type="bar", x="regio", y="n", color="", title="Test"),
        "result_rows": [{"regio": "A", "n": 1.0}, {"regio": "B", "n": 2.0}],
    }
    path = save_chart(state, tmp_path / "chart.html")
    assert path is not None
    html = path.read_text(encoding="utf-8")
    assert "<html>" in html and "plotly" in html
    assert save_chart({"chart": None}, tmp_path / "none.html") is None
