"""Vector index over the CBS catalog, persisted to disk."""

import json
from pathlib import Path
from typing import Self

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

VECTORS_FILE = "vectors.json"
DOCUMENTS_FILE = "documents.json"


class CatalogIndex:
    """Holds the catalog documents and their embeddings.

    The documents are stored separately as well, so keyword search (BM25)
    can be built on the same data without touching the vector store internals.
    """

    def __init__(self, documents: list[Document], vector_store: InMemoryVectorStore) -> None:
        self.documents = documents
        self.vector_store = vector_store

    @classmethod
    def build(cls, documents: list[Document], embeddings: Embeddings) -> Self:
        store = InMemoryVectorStore(embedding=embeddings)
        store.add_documents(documents, ids=[str(doc.id) for doc in documents])
        return cls(documents, store)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.vector_store.dump(str(directory / VECTORS_FILE))
        payload = [
            doc.model_dump(include={"id", "page_content", "metadata"}) for doc in self.documents
        ]
        (directory / DOCUMENTS_FILE).write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path, embeddings: Embeddings) -> Self:
        if not (directory / VECTORS_FILE).exists():
            raise FileNotFoundError(
                f"No catalog index in {directory}. "
                "Build it first: uv run python scripts/build_catalog_index.py"
            )
        store = InMemoryVectorStore.load(str(directory / VECTORS_FILE), embedding=embeddings)
        raw = json.loads((directory / DOCUMENTS_FILE).read_text(encoding="utf-8"))
        documents = [Document(**item) for item in raw]
        return cls(documents, store)

    def __len__(self) -> int:
        return len(self.documents)
