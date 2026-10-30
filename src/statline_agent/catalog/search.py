"""Hybrid search over the CBS catalog: vector search + BM25, merged with Reciprocal Rank Fusion."""

import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import PrivateAttr
from rank_bm25 import BM25Okapi
from snowballstemmer import stemmer

from statline_agent.catalog.index import CatalogIndex

# Small Dutch stopword list: question words and function words carry no topic information.
_STOPWORDS_TEXT = """
de het een en of van in op te voor met naar per door bij aan om uit over tot als
is zijn was waren wordt worden er die dat deze dit wat hoe hoeveel welke wie waar wanneer
ik je jij we wij ze zij hun mijn ons onze ook nog niet geen meer dan sinds
heeft hebben had gaat gaan ging gingen komt komen kwam kwamen krijgt krijgen ligt liggen zich
"""
STOPWORDS = frozenset(_STOPWORDS_TEXT.split())
_TOKEN = re.compile(r"\w+", re.UNICODE)
_STEMMER = stemmer("dutch_porter")


def tokenize(text: str) -> list[str]:
    """Lowercase, drop stopwords, then stem: "gemeenten" and "gemeente" both become "gemeent"."""
    words = [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]
    stems: list[str] = _STEMMER.stemWords(words)
    return stems


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int = 60
) -> list[tuple[str, float]]:
    """Merge ranked lists of ids: score(id) = sum over lists of 1 / (k + rank).

    RRF only uses ranks, not raw scores, so it can combine methods whose scores
    are on different scales (cosine similarity vs. BM25).
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


class HybridCatalogRetriever(BaseRetriever):
    """LangChain retriever that finds the most relevant CBS tables for a question."""

    index: CatalogIndex
    rewriter: Callable[[str], list[str]] | None = None  # optional LLM query rewriting
    k: int = 5
    candidates: int = 25  # how many results each method contributes before fusion
    rrf_k: int = 60
    boost_high_value: bool = True  # prefer CBS's core tables when they are relevant

    _bm25: BM25Okapi = PrivateAttr()
    _by_id: dict[str, Document] = PrivateAttr()

    def model_post_init(self, context: Any, /) -> None:
        self._bm25 = BM25Okapi([tokenize(doc.page_content) for doc in self.index.documents])
        self._by_id = {str(doc.id): doc for doc in self.index.documents}

    def vector_ranking(self, query: str) -> list[str]:
        hits = self.index.vector_store.similarity_search(query, k=self.candidates)
        return [str(doc.id) for doc in hits]

    def keyword_ranking(self, query: str) -> list[str]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top = [i for i in ranked[: self.candidates] if scores[i] > 0]
        return [str(self.index.documents[i].id) for i in top]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        queries = [query, *(self.rewriter(query) if self.rewriter else [])]

        # Each query contributes a vector and a keyword ranking; RRF fuses all of them.
        rankings: list[list[str]] = []
        for q in queries:
            rankings += [self.vector_ranking(q), self.keyword_ranking(q)]
        if self.boost_high_value:
            # An extra ranking with only the High Value Datasets among the results: core tables
            # (e.g. "Bevolking op 1 januari") get a small bonus over niche tables that happen
            # to share a word with the question. Tables that were not found get no bonus.
            found = reciprocal_rank_fusion(rankings, k=self.rrf_k)
            rankings.append([i for i, _ in found if self._by_id[i].metadata.get("high_value")])
        fused = reciprocal_rank_fusion(rankings, k=self.rrf_k)[: self.k]

        vector, keyword = rankings[0], rankings[1]  # ranks for the original question
        results = []
        for table_id, score in fused:
            doc = self._by_id[table_id]
            metadata = {
                **doc.metadata,
                "score": round(score, 5),
                "vector_rank": vector.index(table_id) + 1 if table_id in vector else None,
                "keyword_rank": keyword.index(table_id) + 1 if table_id in keyword else None,
                "search_queries": queries,
            }
            results.append(Document(id=doc.id, page_content=doc.page_content, metadata=metadata))
        return results
