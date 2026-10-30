"""UI-independent helpers: progress labels and the data shown for one answered question.

Kept out of the Streamlit script so they can be unit tested without a browser.
"""

from dataclasses import dataclass, field
from typing import Any

from statline_agent.agent import AgentState
from statline_agent.agent.state import Source
from statline_agent.charts import ChartSpec

STEP_LABELS = {
    "understand_question": "Vraag begrijpen",
    "search_tables": "Tabellen zoeken in de CBS-catalogus",
    "select_table": "Beste tabel kiezen",
    "confirm_table": "Tabel bevestigen",
    "plan_query": "Onderwerpen, regio's en perioden bepalen",
    "fetch_data": "Gegevens ophalen bij het CBS",
    "write_sql": "Analyse schrijven",
    "run_sql": "Analyse uitvoeren",
    "compose_answer": "Antwoord formuleren",
}


def describe_step(node: str, update: AgentState) -> str:
    """One short line per finished step, shown in the progress box."""
    label = STEP_LABELS.get(node, node)
    if node == "search_tables":
        return f"{label}: {len(update.get('candidates', []))} kandidaten"
    if node == "select_table" and update.get("table_id"):
        return f"{label}: {update['table_id']} {update.get('table_title', '')}"
    if node == "fetch_data" and update.get("data_overview"):
        return f"{label}: {update['data_overview'].splitlines()[0]}"
    if node == "confirm_table":
        if update.get("table_id"):
            return f"{label}: gekozen door gebruiker"
        return f"{label}: voorstel bevestigd"
    if node == "run_sql":
        if update.get("sql_error"):
            return f"{label}: poging {update.get('attempts')} mislukt, opnieuw proberen"
        return f"{label}: {len(update.get('result_rows', []))} rijen"
    return label


@dataclass
class TurnResult:
    """Everything the chat shows for one answered question."""

    question: str
    answer: str = ""
    chart: ChartSpec | None = None
    rows: list[dict[str, Any]] = field(default_factory=list)
    source: Source | None = None
    filters: dict[str, list[str]] = field(default_factory=dict)
    requested: dict[str, list[str]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    sql: str = ""
    attempts: int = 0
    steps: list[str] = field(default_factory=list)
    error: str = ""

    @classmethod
    def from_state(
        cls, question: str, state: AgentState, steps: list[str] | None = None
    ) -> "TurnResult":
        return cls(
            question=question,
            steps=steps or [],
            answer=state.get("answer", ""),
            chart=state.get("chart"),
            rows=state.get("result_rows", []),
            source=state.get("source"),
            filters=state.get("filters", {}),
            requested=state.get("requested_filters", {}),
            notes=state.get("plan_notes", []),
            sql=state.get("sql", ""),
            attempts=state.get("attempts", 0),
        )


def friendly_error(exc: Exception) -> str:
    """A message for the user instead of a stack trace."""
    text = str(exc)
    if "rate_limit" in text or "429" in text:
        return (
            "De gratis limiet van het taalmodel is even bereikt. "
            "Wacht een minuut en stel je vraag opnieuw."
        )
    return f"Er ging iets mis ({type(exc).__name__}). Probeer het opnieuw of stel de vraag anders."
