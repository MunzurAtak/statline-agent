"""Dependencies of the agent, injected via LangGraph's runtime context.

Nodes receive these through `runtime.context` instead of global variables, which makes
it easy to swap in fakes in tests and different models in production.
"""

from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever

from statline_agent.catalog import (
    CatalogIndex,
    FastEmbedEmbeddings,
    HybridCatalogRetriever,
    QueryRewriter,
)
from statline_agent.config import Settings, get_settings
from statline_agent.llm import get_chat_model

SEARCH_CANDIDATES = 8


@dataclass
class AgentContext:
    llm: BaseChatModel
    retriever: BaseRetriever


def create_context(settings: Settings | None = None) -> AgentContext:
    """Wire up the real dependencies (LLM, catalog index, retriever with query rewriting)."""
    settings = settings or get_settings()
    llm = get_chat_model(settings)
    index = CatalogIndex.load(
        settings.catalog_index_dir, FastEmbedEmbeddings(settings.embedding_model)
    )
    retriever = HybridCatalogRetriever(
        index=index, k=SEARCH_CANDIDATES, rewriter=QueryRewriter.from_llm(llm)
    )
    return AgentContext(llm=llm, retriever=retriever)
