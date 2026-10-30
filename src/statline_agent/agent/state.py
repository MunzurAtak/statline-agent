"""The state that flows through the LangGraph agent."""

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel

from statline_agent.charts import ChartSpec

MAX_DESCRIPTION_CHARS = 200
Intent = Literal["new_question", "follow_up", "off_topic"]


class Candidate(BaseModel):
    """A CBS table that might answer the question (a search result)."""

    table_id: str
    title: str
    description: str = ""
    measures: list[str] = []
    dimensions: list[str] = []
    modified: str | None = None

    @classmethod
    def from_document(cls, doc: Document) -> "Candidate":
        meta = doc.metadata
        # page_content = title + "\n" + summary/description; the title is stored separately
        description = doc.page_content.partition("\n")[2]
        return cls(
            table_id=meta["table_id"],
            title=meta["title"],
            description=description[:MAX_DESCRIPTION_CHARS],
            measures=meta.get("measures", []),
            dimensions=meta.get("dimensions", []),
            modified=meta.get("modified"),
        )


class Source(BaseModel):
    table_id: str
    title: str
    url: str


class AgentState(TypedDict, total=False):
    """total=False: every node returns only the keys it changes."""

    # add_messages appends instead of overwriting: this is the conversation history.
    messages: Annotated[list[AnyMessage], add_messages]
    intent: Intent
    question: str  # the input; replaced by a standalone version for follow-up questions
    search_queries: list[str]
    candidates: list[Candidate]
    table_id: str | None
    table_title: str
    selection_reason: str
    filters: dict[str, list[str]]  # OData filters: {"Measure": [...], "RegioS": [...], ...}
    requested_filters: dict[str, list[str]]  # the LLM's request: {"Regio's": ["Utrecht"]}
    measure_titles: list[str]
    plan_notes: list[str]  # things the agent could not do exactly as asked
    data_table: str  # DuckDB table with the fetched observations
    data_overview: str  # compact description of that table for the LLM
    sql: str
    sql_explanation: str
    sql_error: str | None
    attempts: int  # SQL attempts for the current dataset
    result_columns: list[str]
    result_rows: list[dict[str, Any]]
    error: str  # unrecoverable problem (e.g. CBS unavailable)
    answer: str
    chart: ChartSpec | None
    source: Source | None
