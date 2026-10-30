"""Fetch the data, then let the LLM write SQL and fix it until it runs (max 3 attempts)."""

from langchain_core.prompts import ChatPromptTemplate
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from statline_agent.agent.context import AgentContext
from statline_agent.agent.state import AgentState
from statline_agent.cbs import QUERY_VIEW, CBSError, QueryError
from statline_agent.llm import with_structured_output

MAX_SQL_ATTEMPTS = 3
MAX_RESULT_ROWS = 100
EMPTY_RESULT_ERROR = (
    "De query gaf geen rijen terug. Controleer de WHERE-voorwaarden tegen de voorbeeldwaarden; "
    "de data is al gefilterd, dus vaak is een WHERE op regio, onderwerp of periode niet nodig."
)


def fetch_data(state: AgentState, runtime: Runtime[AgentContext]) -> AgentState:
    table_id = state["table_id"]
    assert table_id is not None  # guaranteed by the graph's routing
    store = runtime.context.store
    try:
        data_table = store.load(table_id, state["filters"])
    except CBSError as exc:
        return {"error": f"Data ophalen bij CBS mislukt: {exc}"}
    # Reset the SQL loop: this is a fresh dataset.
    return {
        "data_table": data_table,
        "data_overview": store.describe(data_table),
        "attempts": 0,
        "sql_error": None,
    }


SQL_SYSTEM_PROMPT = f"""\
Je bent een data-analist. Schrijf één DuckDB SQL SELECT-query op de tabel `{QUERY_VIEW}` die \
de vraag beantwoordt.

- De tabel bevat al precies de gevraagde selectie (onderwerpen, regio's en perioden). Filter \
daar niet opnieuw op, tenzij de vraag om een deel van de gegevens vraagt.
- Gebruik de *_title-kolommen voor namen (onderwerp, regio, periode) en Value voor de cijfers.
- Neem altijd de naamkolommen op die het resultaat beschrijven (zoals regio en periode), ook \
als er maar één rij is: het antwoord moet kunnen zeggen waar de cijfers over gaan.
- Houd het resultaat klein (hooguit 50 rijen) en geschikt voor een grafiek, bijvoorbeeld de \
kolommen periode, regio en waarde.
- Sorteer tijdreeksen op de periodecode (de kolom zonder _title), niet op de titel.
- Bereken afgeleide cijfers zoals verschillen of procentuele groei in SQL als de vraag daarom \
vraagt.
- Zet kolomnamen met hoofdletters tussen dubbele aanhalingstekens, bijvoorbeeld "Value".
- Een apostrof in tekst schrijf je dubbel, bijvoorbeeld '''s-Gravenhage' voor 's-Gravenhage."""

SQL_HUMAN_PROMPT = "Vraag: {question}\n\n{overview}{previous}"
SQL_PROMPT = ChatPromptTemplate.from_messages(
    [("system", SQL_SYSTEM_PROMPT), ("human", SQL_HUMAN_PROMPT)]
)


class SqlQuery(BaseModel):
    sql: str = Field(description="Eén DuckDB SELECT-query")
    explanation: str = Field(description="Wat de query berekent, in één zin")


def write_sql(state: AgentState, runtime: Runtime[AgentContext]) -> AgentState:
    previous = ""
    if state.get("sql_error"):
        # Self-correction: show the model its own failed query and the error message.
        previous = (
            f"\n\nJe vorige query:\n{state.get('sql', '')}\n"
            f"gaf deze fout:\n{state['sql_error']}\nSchrijf een verbeterde query."
        )
    chain = SQL_PROMPT | with_structured_output(runtime.context.llm, SqlQuery)
    query = chain.invoke(
        {
            "question": state["question"],
            "overview": state.get("data_overview", ""),
            "previous": previous,
        }
    )
    return {"sql": query.sql, "sql_explanation": query.explanation}


def run_sql(state: AgentState, runtime: Runtime[AgentContext]) -> AgentState:
    attempts = state.get("attempts", 0) + 1
    try:
        columns, rows = runtime.context.store.run_select(
            state["sql"], state["data_table"], max_rows=MAX_RESULT_ROWS
        )
    except QueryError as exc:
        return {"attempts": attempts, "sql_error": str(exc)}
    if not rows:
        return {"attempts": attempts, "sql_error": EMPTY_RESULT_ERROR}
    return {
        "attempts": attempts,
        "sql_error": None,
        "result_columns": columns,
        "result_rows": rows,
    }
