from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from statline_agent.agent import AgentContext, build_graph
from statline_agent.agent.memory import memory_checkpointer
from statline_agent.agent.nodes.answer import OFF_TOPIC_ANSWER, Answer
from statline_agent.agent.nodes.data import SqlQuery
from statline_agent.agent.nodes.plan import DimensionChoice, QueryPlan
from statline_agent.agent.nodes.search import TableChoice
from statline_agent.agent.nodes.understand import QuestionIntent, format_history
from statline_agent.catalog import CatalogIndex, HybridCatalogRetriever
from statline_agent.cbs import CBSClient, DataStore
from statline_agent.charts import ChartSpec
from tests.fakes import FakeStructuredChatModel, prompt_text

SQL = SqlQuery(sql='SELECT "Value" AS inwoners FROM data', explanation="bevolking")
NO_CHART = ChartSpec(type="none", x="", y="", color="", title="")


def plan(region: str) -> QueryPlan:
    return QueryPlan(
        measures=["M000352_3"],
        filters=[DimensionChoice(dimension="Regio's", values=[region])],
        frequency="jaar",
        year_from=0,
        year_to=0,
        latest_only=True,
    )


FIRST_TURN: list[BaseModel] = [
    TableChoice(table_id="70072ned", reason="regio"),
    plan("Nederland"),
    SQL,
    Answer(text="Antwoord over Nederland.", chart=NO_CHART),
]


@pytest.fixture
def context(cbs_client: CBSClient) -> AgentContext:
    doc = Document(
        id="70072ned",
        page_content="Regionale kerncijfers Nederland\nKerncijfers per regio",
        metadata={"table_id": "70072ned", "title": "Regionale kerncijfers Nederland"},
    )
    embeddings = DeterministicFakeEmbedding(size=8)
    retriever = HybridCatalogRetriever(index=CatalogIndex.build([doc], embeddings), k=1)
    store = DataStore(cbs_client, ":memory:")
    return AgentContext(FakeStructuredChatModel(), retriever, cbs_client, embeddings, store)


class Conversation:
    """Runs several turns on one thread, like a chat session."""

    def __init__(self, context: AgentContext) -> None:
        self.context = context
        self.graph = build_graph(checkpointer=memory_checkpointer())
        self.config: RunnableConfig = {"configurable": {"thread_id": "test-thread"}}

    def ask(self, question: str, *responses: BaseModel) -> dict[str, Any]:
        assert isinstance(self.context.llm, FakeStructuredChatModel)
        self.context.llm.responses = list(responses)
        state = self.graph.invoke({"question": question}, self.config, context=self.context)
        assert self.context.llm.responses == [], "not all fake LLM responses were used"
        return dict(state)


def test_follow_up_reuses_table_and_skips_search(context: AgentContext) -> None:
    chat = Conversation(context)
    chat.ask("Hoeveel inwoners heeft Nederland?", *FIRST_TURN)

    intent = QuestionIntent(
        intent="follow_up", standalone_question="Hoeveel inwoners heeft Zuid-Holland?"
    )
    answer = Answer(text="Antwoord over Zuid-Holland.", chart=NO_CHART)
    # No TableChoice in this list: if the agent searched again, the fake LLM would fail.
    state = chat.ask("En Zuid-Holland?", intent, plan("Zuid-Holland"), SQL, answer)

    assert state["question"] == "Hoeveel inwoners heeft Zuid-Holland?"
    assert state["table_id"] == "70072ned"
    assert state["filters"]["RegioS"] == ["PV28"]
    assert [type(m) for m in state["messages"]] == [HumanMessage, AIMessage] * 2
    assert state["messages"][-1].content == "Antwoord over Zuid-Holland."


def test_intent_prompt_contains_the_conversation(context: AgentContext) -> None:
    chat = Conversation(context)
    chat.ask("Hoeveel inwoners heeft Nederland?", *FIRST_TURN)
    off_topic = QuestionIntent(intent="off_topic", standalone_question="Wie wint het WK?")
    chat.ask("Wie wint het WK?", off_topic)

    assert isinstance(context.llm, FakeStructuredChatModel)
    sent = prompt_text(context.llm.prompts[-1])
    assert "Gebruiker: Hoeveel inwoners heeft Nederland?" in sent
    assert "Assistent: Antwoord over Nederland." in sent
    assert "Regionale kerncijfers Nederland" in sent  # the current table


def test_off_topic_gets_direct_answer(context: AgentContext) -> None:
    chat = Conversation(context)
    chat.ask("Hoeveel inwoners heeft Nederland?", *FIRST_TURN)
    off_topic = QuestionIntent(intent="off_topic", standalone_question="Wie wint het WK?")
    state = chat.ask("Wie wint het WK?", off_topic)

    assert state["answer"] == OFF_TOPIC_ANSWER
    assert state["table_id"] is None
    assert state["result_rows"] == []  # the previous answer's data was reset


def test_new_question_searches_again(context: AgentContext) -> None:
    chat = Conversation(context)
    chat.ask("Hoeveel inwoners heeft Nederland?", *FIRST_TURN)
    new = QuestionIntent(intent="new_question", standalone_question="Hoeveel woningen zijn er?")
    state = chat.ask("Hoeveel woningen zijn er?", new, *FIRST_TURN)
    assert state["question"] == "Hoeveel woningen zijn er?"
    assert len(state["messages"]) == 4


def test_first_question_needs_no_intent_call(context: AgentContext) -> None:
    state = Conversation(context).ask("Hoeveel inwoners heeft Nederland?", *FIRST_TURN)
    assert state["intent"] == "new_question"


def test_format_history_keeps_the_last_messages() -> None:
    messages = [HumanMessage(f"vraag {i}") for i in range(10)]
    text = format_history(list(messages))
    assert text.splitlines()[0] == "Gebruiker: vraag 4"
    assert len(text.splitlines()) == 6
