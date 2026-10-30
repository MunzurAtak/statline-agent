"""Turn CBS catalog entries into LangChain Documents for retrieval."""

import re
from collections.abc import Iterable

from langchain_core.documents import Document

from statline_agent.catalog.enrichment import TableMetadata
from statline_agent.cbs import DatasetInfo

MAX_DESCRIPTION_CHARS = 300
# The EU's High Value Datasets: CBS's core tables (population, labour, prices, ...).
HIGH_VALUE_MARKER = "High Value Dataset"
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

    The text is kept short and descriptive: the embedding model only reads ~128 tokens,
    and an ablation showed that adding the (often generic) measure titles to the text made
    both BM25 and vector search worse. Measures and dimensions are kept as metadata, so the
    agent can show them to the LLM when it chooses between candidate tables.
    """
    parts = [dataset.title]
    if metadata and metadata.summary:
        parts.append(clean_text(metadata.summary))
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
            "measures": metadata.measures if metadata else [],
            "dimensions": metadata.dimensions if metadata else [],
            "high_value": bool(metadata and HIGH_VALUE_MARKER in metadata.summary),
        },
    )
