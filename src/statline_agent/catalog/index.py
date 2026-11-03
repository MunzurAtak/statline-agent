"""Vector index over the CBS catalog, persisted to disk.

The index is small enough to ship with the repository (data/catalog_index), so the app starts
without first downloading and embedding ~1,300 tables:
- documents.json: id, text and metadata per table, one table per line (readable diffs when
  the catalog is refreshed)
- vectors.npz: the embeddings as float16 (~1 MB instead of ~16 MB as JSON)
"""

import json
from pathlib import Path
from typing import Self

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

DOCUMENTS_FILE = "documents.json"
VECTORS_FILE = "vectors.npz"
LEGACY_VECTORS_FILE = "vectors.json"  # format before v0.1.0 (InMemoryVectorStore.dump)


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
        documents = sorted(self.documents, key=lambda doc: str(doc.id))
        ids = [str(doc.id) for doc in documents]
        vectors = np.array([self.vector_store.store[i]["vector"] for i in ids], dtype=np.float16)
        np.savez_compressed(directory / VECTORS_FILE, ids=np.array(ids), vectors=vectors)

        lines = [
            json.dumps(
                doc.model_dump(include={"id", "page_content", "metadata"}), ensure_ascii=False
            )
            for doc in documents
        ]
        text = "[\n" + ",\n".join(lines) + "\n]\n"
        (directory / DOCUMENTS_FILE).write_text(text, encoding="utf-8", newline="\n")
        (directory / LEGACY_VECTORS_FILE).unlink(missing_ok=True)

    @classmethod
    def load(cls, directory: Path, embeddings: Embeddings) -> Self:
        documents_path = directory / DOCUMENTS_FILE
        if not documents_path.exists():
            raise FileNotFoundError(
                f"No catalog index in {directory}. "
                "Build it first: uv run python scripts/build_catalog_index.py"
            )
        raw = json.loads(documents_path.read_text(encoding="utf-8"))
        documents = [Document(**item) for item in raw]

        if not (directory / VECTORS_FILE).exists() and (directory / LEGACY_VECTORS_FILE).exists():
            store = InMemoryVectorStore.load(
                str(directory / LEGACY_VECTORS_FILE), embedding=embeddings
            )
            return cls(documents, store)

        with np.load(directory / VECTORS_FILE) as data:
            ids = [str(i) for i in data["ids"]]
            vectors = data["vectors"].astype(np.float32)
        by_id = {str(doc.id): doc for doc in documents}
        store = InMemoryVectorStore(embedding=embeddings)
        # Fill the store with the saved vectors instead of embedding everything again. This is
        # the same structure InMemoryVectorStore.load() restores from its own JSON dump.
        store.store = {
            i: {
                "id": i,
                "vector": vector.tolist(),
                "text": by_id[i].page_content,
                "metadata": by_id[i].metadata,
            }
            for i, vector in zip(ids, vectors, strict=True)
        }
        return cls(documents, store)

    def __len__(self) -> int:
        return len(self.documents)
