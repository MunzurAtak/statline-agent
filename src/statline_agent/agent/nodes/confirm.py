"""Human-in-the-loop: let the user confirm (or change) the table before any data is fetched.

This is its own node on purpose. On resume, LangGraph re-runs the interrupted node from the
start; keeping the LLM call in select_table means it is not repeated (and not paid for twice).
"""

from typing import Any

from langgraph.runtime import Runtime
from langgraph.types import interrupt

from statline_agent.agent.context import AgentContext
from statline_agent.agent.state import AgentState

MAX_ALTERNATIVES = 3
NO_TABLE = "NONE"


def confirm_table(state: AgentState, runtime: Runtime[AgentContext]) -> AgentState:
    if not runtime.context.confirm_tables:
        return {}  # auto mode (tests, evaluations, CLI with --auto)

    proposed = state["table_id"]
    candidates = state.get("candidates", [])
    alternatives = [c for c in candidates if c.table_id != proposed][:MAX_ALTERNATIVES]
    request: dict[str, Any] = {
        "proposed": {
            "table_id": proposed,
            "title": state.get("table_title", ""),
            "reason": state.get("selection_reason", ""),
        },
        "alternatives": [{"table_id": c.table_id, "title": c.title} for c in alternatives],
    }
    # Pauses the graph; the checkpointer saves the state. The value passed with
    # Command(resume=...) comes back here: a table id, "" to accept, or "NONE".
    choice = str(interrupt(request) or "").strip()

    if not choice or choice == proposed:
        return {}
    if choice.upper() == NO_TABLE:
        reason = "Volgens de gebruiker past geen van de voorgestelde tabellen."
        return {"table_id": None, "selection_reason": reason}
    chosen = next((c for c in candidates if c.table_id.lower() == choice.lower()), None)
    if chosen is None:
        return {}  # unknown id: keep the proposal
    return {
        "table_id": chosen.table_id,
        "table_title": chosen.title,
        "selection_reason": "Gekozen door de gebruiker.",
    }
