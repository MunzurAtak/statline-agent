"""LLM query rewriting: translate a user's question into CBS terminology.

Users ask "hoeveel inwoners", CBS tables say "bevolking". Neither BM25 nor a small
embedding model bridges that gap well; an LLM does. We generate a few search queries
and fuse the results of all of them (also known as multi-query retrieval / RAG-Fusion).
"""

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from statline_agent.llm import with_structured_output

logger = logging.getLogger(__name__)

# Note: the examples deliberately do NOT overlap with evals/retrieval_cases.jsonl,
# otherwise the evaluation would be measuring the prompt instead of the method.
SYSTEM_PROMPT = """\
Je helpt bij het zoeken in de catalogus van CBS StatLine (open data van het Centraal \
Bureau voor de Statistiek). Herschrijf de vraag naar 2 tot 3 korte zoekopdrachten in de \
terminologie die CBS in tabeltitels gebruikt.

Richtlijnen:
- Gebruik formele, statistische termen, bijvoorbeeld "levendgeborenen" in plaats van \
"baby's", "echtscheidingen" in plaats van "uit elkaar gaan", "werkzame beroepsbevolking" \
in plaats van "mensen met een baan".
- Laat plaatsnamen, jaartallen en vraagwoorden weg; die staan niet in tabeltitels.
- Elke zoekopdracht beschrijft zelfstandig het onderwerp; dus niet alleen "gemeente" of \
"regio".
- Elke zoekopdracht is maximaal 6 woorden.
- Beantwoord de vraag nooit zelf, ook niet als hij niet over statistiek gaat; geef alleen \
zoekopdrachten."""

PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{question}")])


class SearchQueries(BaseModel):
    """Search queries for the CBS catalog."""

    # No min/max length constraints here: strict JSON-schema mode does not support them.
    # The rewriter caps the number of queries in code instead.
    queries: list[str] = Field(description="2 tot 3 korte zoekopdrachten in CBS-terminologie")


class QueryRewriter:
    """Turns a question into extra search queries. Never raises: on failure it returns []."""

    def __init__(self, chain: Runnable[dict[str, str], SearchQueries], max_queries: int = 3):
        self._chain = chain
        self.max_queries = max_queries

    @classmethod
    def from_llm(cls, llm: BaseChatModel) -> "QueryRewriter":
        return cls(PROMPT | with_structured_output(llm, SearchQueries))

    def __call__(self, question: str) -> list[str]:
        try:
            result = self._chain.invoke({"question": question})
        except Exception as exc:  # the search must keep working if the LLM is unavailable
            logger.warning("Query rewriting failed, using the original question only: %s", exc)
            return []
        queries = [q.strip() for q in result.queries if q.strip()]
        return queries[: self.max_queries]
