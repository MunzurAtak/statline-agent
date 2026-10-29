"""The state that flows through the LangGraph agent."""

from typing import TypedDict

from langchain_core.documents import Document
from pydantic import BaseModel

MAX_DESCRIPTION_CHARS = 200


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


class AgentState(TypedDict, total=False):
    """total=False: every node returns only the keys it changes."""

    question: str
    search_queries: list[str]
    candidates: list[Candidate]
    table_id: str | None
    table_title: str
    selection_reason: str
