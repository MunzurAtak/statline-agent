"""Write the final answer: plain-language text, a chart spec and the source."""

from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from statline_agent.agent.context import AgentContext
from statline_agent.agent.state import AgentState, Source
from statline_agent.charts import ChartSpec
from statline_agent.llm import with_structured_output

MAX_ROWS_IN_PROMPT = 50
OFF_TOPIC_ANSWER = (
    "Ik beantwoord vragen met cijfers van het CBS, bijvoorbeeld over bevolking, wonen, werk, "
    "inkomen of zorg in Nederland. Waar wil je iets over weten?"
)
STATLINE_URL = "https://dataportal.cbs.nl/detail/CBS/{table_id}"

ANSWER_SYSTEM_PROMPT = """\
Je bent een data-analist die vragen beantwoordt met cijfers van het CBS.

- Antwoord in de taal van de vraag, in 2 tot 5 zinnen.
- Gebruik uitsluitend getallen uit het queryresultaat; verzin niets. Rond getallen logisch af \
en noem de periode waarop ze betrekking hebben.
- Noem beperkingen als die er zijn (zie 'Let op').
- Kies een grafiek die past bij het resultaat, met kolomnamen die exact in het resultaat \
voorkomen. Kies type none als het resultaat één getal is."""

ANSWER_HUMAN_PROMPT = """\
Vraag: {question}

Bron: {table}
Onderwerpen: {measures}
Let op: {notes}
Query: {explanation}

Resultaat:
{result}"""

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [("system", ANSWER_SYSTEM_PROMPT), ("human", ANSWER_HUMAN_PROMPT)]
)


class Answer(BaseModel):
    text: str = Field(description="Het antwoord op de vraag")
    chart: ChartSpec


def format_result(columns: list[str], rows: list[dict[str, Any]]) -> str:
    """Compact, CSV-like text; floats rounded to keep the prompt small."""

    def cell(value: Any) -> str:
        return f"{value:.2f}" if isinstance(value, float) else str(value)

    lines = [" | ".join(columns)]
    lines += [" | ".join(cell(row[c]) for c in columns) for row in rows[:MAX_ROWS_IN_PROMPT]]
    if len(rows) > MAX_ROWS_IN_PROMPT:
        lines.append(f"... ({len(rows) - MAX_ROWS_IN_PROMPT} rijen weggelaten)")
    return "\n".join(lines)


def failure_message(state: AgentState) -> str | None:
    """Fixed texts for the cases where there is no result to talk about (no LLM call needed)."""
    if state.get("intent") == "off_topic":
        return OFF_TOPIC_ANSWER
    if not state.get("table_id"):
        reason = state.get("selection_reason", "")
        return f"Ik heb geen CBS-tabel gevonden die deze vraag kan beantwoorden. {reason}".strip()
    if state.get("error"):
        return f"Het ophalen van de gegevens is mislukt: {state['error']}"
    if not state.get("result_rows"):
        return (
            "Het is me niet gelukt een werkende analyse te maken op de gevonden gegevens. "
            f"Laatste fout: {state.get('sql_error', 'onbekend')}"
        )
    return None


def compose_answer(state: AgentState, runtime: Runtime[AgentContext]) -> AgentState:
    table_id = state.get("table_id")
    source = (
        Source(
            table_id=table_id,
            title=state.get("table_title", ""),
            url=STATLINE_URL.format(table_id=table_id),
        )
        if table_id
        else None
    )

    failure = failure_message(state)
    if failure:
        return {
            "answer": failure,
            "chart": None,
            "source": source,
            "messages": [AIMessage(failure)],
        }

    columns, rows = state["result_columns"], state["result_rows"]
    chain = ANSWER_PROMPT | with_structured_output(runtime.context.llm, Answer)
    answer = chain.invoke(
        {
            "question": state["question"],
            "table": f"{table_id} {state.get('table_title', '')}",
            "measures": ", ".join(state.get("measure_titles", [])),
            "notes": " ".join(state.get("plan_notes", [])) or "geen",
            "explanation": state.get("sql_explanation", ""),
            "result": format_result(columns, rows),
        }
    )
    # Only keep the chart if it refers to columns that really exist in the result.
    chart = answer.chart if answer.chart.is_valid_for(columns) else None
    return {
        "answer": answer.text,
        "chart": chart,
        "source": source,
        "messages": [AIMessage(answer.text)],
    }
