"""Turn CBS catalog entries into LangChain Documents for retrieval."""

import re
from collections.abc import Iterable

from langchain_core.documents import Document

from statline_agent.catalog.enrichment import TableMetadata
from statline_agent.cbs import DatasetInfo

MAX_DESCRIPTION_CHARS = 300
_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Strip HTML tags and collapse whitespace."""
    return _WHITESPACE.sub(" ", _HTML_TAG.sub(" ", text)).strip()


def select_datasets(datasets: Iterable[DatasetInfo]) -> list[DatasetInfo]:
    """Keep only tables the agent can actually query: active and available via OData."""
    return [d for d in datasets if d.is_active and d.has_odata]


def dataset_to_document(dataset: DatasetInfo, metadata: TableMetadata | None = None) -> Document:
    """Build the searchable text for one table.

    Order matters: the embedding model only reads the first ~128 tokens, so the most
    informative parts (title, summary, measures) come first. BM25 reads everything.
    """
    parts = [dataset.title]
    if metadata:
        if metadata.summary:
            parts.append(clean_text(metadata.summary))
        if metadata.measures:
            parts.append("Onderwerpen: " + "; ".join(metadata.measures))
        if metadata.dimensions:
            parts.append("Indeling: " + ", ".join(metadata.dimensions))
    parts.append(clean_text(dataset.description)[:MAX_DESCRIPTION_CHARS])

    return Document(
        id=dataset.identifier,
        page_content="\n".join(p for p in parts if p),
        metadata={
            "table_id": dataset.identifier,
            "title": dataset.title,
            "status": dataset.status,
            "modified": dataset.modified.date().isoformat() if dataset.modified else None,
            "observation_count": dataset.observation_count,
        },
    )
