from pathlib import Path
from typing import Any

import numpy as np
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from statline_agent.catalog import (
    CatalogIndex,
    FastEmbedEmbeddings,
    clean_text,
    dataset_to_document,
    select_datasets,
)
from statline_agent.catalog import embeddings as embeddings_module
from statline_agent.cbs import CBSClient, DatasetInfo


def make_dataset(identifier: str, title: str, description: str = "") -> DatasetInfo:
    return DatasetInfo(
        identifier=identifier,
        title=title,
        description=description,
        status="Regulier",
        distributions=[{"Format": "odata", "DownloadUrl": f"https://x/{identifier}"}],
    )


def test_clean_text_strips_html_and_whitespace() -> None:
    assert clean_text('Zie <a href="x">tabel</a>\n\n  hier') == "Zie tabel hier"


def test_select_datasets_keeps_active_odata_tables(cbs_client: CBSClient) -> None:
    datasets = cbs_client.list_datasets()
    # Fixture: 00370 = CSV only, 00371edu = CSV only, 00372 = discontinued
    assert select_datasets(datasets) == []
    assert select_datasets([make_dataset("1", "Actief")])[0].identifier == "1"


def test_dataset_to_document() -> None:
    doc = dataset_to_document(make_dataset("70072ned", "Regionale kerncijfers", "x" * 2000))
    assert doc.id == "70072ned"
    assert doc.page_content.startswith("Regionale kerncijfers\n")
    assert len(doc.page_content) < 600
    assert doc.metadata["table_id"] == "70072ned"


def test_index_build_save_load_roundtrip(tmp_path: Path) -> None:
    embeddings = DeterministicFakeEmbedding(size=16)
    docs = [
        dataset_to_document(make_dataset("A1", "Bevolking per gemeente")),
        dataset_to_document(make_dataset("B2", "Woningvoorraad naar eigendom")),
    ]
    CatalogIndex.build(docs, embeddings).save(tmp_path)

    loaded = CatalogIndex.load(tmp_path, embeddings)
    assert len(loaded) == 2
    # The fake embedding is deterministic: identical text gives an identical vector.
    hits = loaded.vector_store.similarity_search(docs[1].page_content, k=1)
    assert hits[0].metadata["table_id"] == "B2"


def test_load_without_index_gives_helpful_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="build_catalog_index"):
        CatalogIndex.load(tmp_path, DeterministicFakeEmbedding(size=16))


def test_fastembed_wrapper_converts_vectors(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTextEmbedding:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def embed(self, texts: list[str], **_: Any) -> list[np.ndarray[Any, Any]]:
            return [np.array([float(len(t)), 1.0]) for t in texts]

    monkeypatch.setattr(embeddings_module, "TextEmbedding", FakeTextEmbedding)
    emb = FastEmbedEmbeddings("some/model")
    assert emb.embed_documents(["ab", "abcd"]) == [[2.0, 1.0], [4.0, 1.0]]
    assert emb.embed_query("abc") == [3.0, 1.0]
