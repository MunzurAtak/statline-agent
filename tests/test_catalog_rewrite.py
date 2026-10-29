from langchain_core.runnables import RunnableLambda

from statline_agent.catalog import QueryRewriter, SearchQueries


def test_rewriter_returns_cleaned_and_capped_queries() -> None:
    def fake_llm(_: dict[str, str]) -> SearchQueries:
        return SearchQueries(queries=[" bevolking ", "", "a", "b", "c"])

    rewriter = QueryRewriter(RunnableLambda(fake_llm), max_queries=3)
    assert rewriter("Hoeveel inwoners?") == ["bevolking", "a", "b"]


def test_rewriter_passes_question_to_chain() -> None:
    seen: list[dict[str, str]] = []

    def fake_llm(inputs: dict[str, str]) -> SearchQueries:
        seen.append(inputs)
        return SearchQueries(queries=["x"])

    QueryRewriter(RunnableLambda(fake_llm))("Wat is de inflatie?")
    assert seen == [{"question": "Wat is de inflatie?"}]


def test_rewriter_falls_back_to_empty_list_on_error() -> None:
    def broken(_: dict[str, str]) -> SearchQueries:
        raise RuntimeError("rate limited")

    assert QueryRewriter(RunnableLambda(broken))("vraag") == []
