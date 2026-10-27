"""Download the CBS catalog, enrich it with per-table metadata and build the search index.

Usage: uv run python scripts/build_catalog_index.py
The first run fetches metadata for ~1300 tables (a few minutes); later runs use the cache.
"""

import time
from pathlib import Path

from statline_agent.catalog import (
    CatalogIndex,
    FastEmbedEmbeddings,
    dataset_to_document,
    fetch_all_metadata,
    select_datasets,
)
from statline_agent.cbs import CBSClient
from statline_agent.config import get_settings

METADATA_CACHE = Path(".cache/catalog_metadata.json")


def print_progress(done: int, total: int) -> None:
    if done % 100 == 0 or done == total:
        print(f"  metadata {done}/{total}")


def main() -> None:
    settings = get_settings()
    started = time.perf_counter()

    with CBSClient() as client:
        datasets = client.list_datasets()
        selected = select_datasets(datasets)
        print(f"Catalog: {len(datasets)} tables, {len(selected)} active and available via OData")

        metadata = fetch_all_metadata(
            client, [d.identifier for d in selected], METADATA_CACHE, on_progress=print_progress
        )
    print(f"Metadata available for {len(metadata)}/{len(selected)} tables")

    documents = [dataset_to_document(d, metadata.get(d.identifier)) for d in selected]
    embeddings = FastEmbedEmbeddings(settings.embedding_model)
    print(f"Embedding {len(documents)} documents with {settings.embedding_model} ...")
    index = CatalogIndex.build(documents, embeddings)
    index.save(settings.catalog_index_dir)

    elapsed = time.perf_counter() - started
    print(f"Saved index to {settings.catalog_index_dir} in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
