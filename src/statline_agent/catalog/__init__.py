"""Searchable index over the CBS StatLine catalog."""

from statline_agent.catalog.documents import clean_text, dataset_to_document, select_datasets
from statline_agent.catalog.embeddings import FastEmbedEmbeddings
from statline_agent.catalog.index import CatalogIndex

__all__ = [
    "CatalogIndex",
    "FastEmbedEmbeddings",
    "clean_text",
    "dataset_to_document",
    "select_datasets",
]
