"""Download the CBS catalog and build the local search index.

Usage: uv run python scripts/build_catalog_index.py
"""

import time

from statline_agent.catalog import (
    CatalogIndex,
    FastEmbedEmbeddings,
    dataset_to_document,
    select_datasets,
)
from statline_agent.cbs import CBSClient
from statline_agent.config import get_settings


def main() -> None:
    settings = get_settings()

    started = time.perf_counter()
    with CBSClient() as client:
        datasets = client.list_datasets()
    selected = select_datasets(datasets)
    print(f"Catalog: {len(datasets)} tables, {len(selected)} active and available via OData")

    documents = [dataset_to_document(d) for d in selected]
    embeddings = FastEmbedEmbeddings(settings.embedding_model)
    print(f"Embedding {len(documents)} documents with {settings.embedding_model} ...")
    index = CatalogIndex.build(documents, embeddings)
    index.save(settings.catalog_index_dir)

    elapsed = time.perf_counter() - started
    print(f"Saved index to {settings.catalog_index_dir} in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
