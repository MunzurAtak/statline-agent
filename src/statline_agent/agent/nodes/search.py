"""Nodes that find the right CBS table: search the catalog, then let the LLM choose."""

from langchain_core.prompts import ChatPromptTemplate
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from statline_agent.agent.context import AgentContext
from statline_agent.agent.state import AgentState, Candidate
from statline_agent.llm import with_structured_output

MAX_MEASURES_SHOWN = 12
NO_TABLE = "NONE"


def search_tables(state: AgentState, runtime: Runtime[AgentContext]) -> AgentState:
    docs = runtime.context.retriever.invoke(state["question"])
    queries = docs[0].metadata.get("search_queries", []) if docs else []
    return {
        "candidates": [Candidate.from_document(doc) for doc in docs],
        "search_queries": queries or [state["question"]],
    }


SELECT_SYSTEM_PROMPT = """\
Je bent een data-analist die werkt met CBS StatLine. Kies uit de kandidaat-tabellen de \
tabel waarmee de vraag het best beantwoord kan worden.

Let op:
- De tabel moet de gevraagde grootheid bevatten (zie de onderwerpen).
- De tabel moet de juiste indeling hebben, bijvoorbeeld regio's als de vraag over een \
gemeente gaat.
- Geef bij twijfel de voorkeur aan een recent bijgewerkte tabel.
- Antwoord met NONE als geen enkele tabel de vraag kan beantwoorden."""

SELECT_PROMPT = ChatPromptTemplate.from_messages(
    [("system", SELECT_SYSTEM_PROMPT), ("human", "Vraag: {question}\n\nKandidaten:\n{candidates}")]
)


class TableChoice(BaseModel):
    table_id: str = Field(description="ID van de best passende tabel, of NONE")
    reason: str = Field(description="Korte uitleg in één zin")


def format_candidates(candidates: list[Candidate]) -> str:
    """Compact text representation; keeps the prompt small for low token-per-minute limits."""
    blocks = []
    for c in candidates:
        lines = [f"- {c.table_id}: {c.title}"]
        if c.description:
            lines.append(f"  {c.description}")
        if c.measures:
            lines.append("  Onderwerpen: " + "; ".join(c.measures[:MAX_MEASURES_SHOWN]))
        if c.dimensions:
            lines.append("  Indeling: " + ", ".join(c.dimensions))
        if c.modified:
            lines.append(f"  Bijgewerkt: {c.modified}")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def select_table(state: AgentState, runtime: Runtime[AgentContext]) -> AgentState:
    candidates = state.get("candidates", [])
    if not candidates:
        return {"table_id": None, "selection_reason": "Geen kandidaat-tabellen gevonden."}

    chain = SELECT_PROMPT | with_structured_output(runtime.context.llm, TableChoice)
    choice = chain.invoke(
        {"question": state["question"], "candidates": format_candidates(candidates)}
    )

    if choice.table_id.strip().upper() == NO_TABLE:
        return {"table_id": None, "selection_reason": choice.reason}

    by_id = {c.table_id.lower(): c for c in candidates}
    chosen = by_id.get(choice.table_id.strip().lower())
    reason = choice.reason
    if chosen is None:
        # The model returned an id that is not in the list: fall back to the best search hit.
        chosen = candidates[0]
        reason = f"Onbekende keuze '{choice.table_id}'; beste zoekresultaat gebruikt."
    return {"table_id": chosen.table_id, "table_title": chosen.title, "selection_reason": reason}
