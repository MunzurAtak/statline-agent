"""Searchable index over the CBS StatLine catalog."""

from statline_agent.catalog.documents import clean_text, dataset_to_document, select_datasets
from statline_agent.catalog.embeddings import FastEmbedEmbeddings
from statline_agent.catalog.enrichment import TableMetadata, fetch_all_metadata
from statline_agent.catalog.index import CatalogIndex
from statline_agent.catalog.search import (
    HybridCatalogRetriever,
    reciprocal_rank_fusion,
    tokenize,
)

__all__ = [
    "CatalogIndex",
    "FastEmbedEmbeddings",
    "HybridCatalogRetriever",
    "TableMetadata",
    "clean_text",
    "dataset_to_document",
    "fetch_all_metadata",
    "reciprocal_rank_fusion",
    "select_datasets",
    "tokenize",
]
