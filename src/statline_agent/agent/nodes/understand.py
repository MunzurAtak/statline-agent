"""First node of every turn: reset per-question state and work out what the user means."""

from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from statline_agent.agent.context import AgentContext
from statline_agent.agent.state import AgentState, Intent
from statline_agent.llm import with_structured_output

MAX_HISTORY_MESSAGES = 6
MAX_MESSAGE_CHARS = 300

INTENT_SYSTEM_PROMPT = """\
Je hoort bij een assistent die vragen beantwoordt met cijfers van het CBS (Centraal Bureau \
voor de Statistiek). Bepaal wat voor bericht de gebruiker stuurt, gegeven het gesprek:

- follow_up: een vervolgvraag die met dezelfde tabel te beantwoorden is, bijvoorbeeld voor \
een andere regio of periode, of een vergelijking. Huidige tabel: {table}
- new_question: een vraag over een ander onderwerp met Nederlandse cijfers.
- off_topic: geen vraag over Nederlandse statistieken of cijfers.

Herschrijf het bericht daarnaast als zelfstandige vraag die zonder het gesprek te begrijpen \
is (vul bijvoorbeeld het onderwerp aan uit eerdere vragen)."""

INTENT_PROMPT = ChatPromptTemplate.from_messages(
    [("system", INTENT_SYSTEM_PROMPT), ("human", "Gesprek:\n{history}\n\nNieuw bericht: {message}")]
)


class QuestionIntent(BaseModel):
    intent: Intent
    standalone_question: str = Field(description="Het bericht als zelfstandige vraag")


def format_history(messages: list[AnyMessage]) -> str:
    lines = []
    for message in messages[-MAX_HISTORY_MESSAGES:]:
        role = "Gebruiker" if message.type == "human" else "Assistent"
        lines.append(f"{role}: {str(message.content)[:MAX_MESSAGE_CHARS]}")
    return "\n".join(lines)


def understand_question(state: AgentState, runtime: Runtime[AgentContext]) -> AgentState:
    message = state["question"]
    history = state.get("messages", [])  # earlier turns; this message is added below

    # Fields that belong to the previous question. The chosen table is kept on purpose:
    # a follow-up question reuses it.
    update: AgentState = {
        "messages": [HumanMessage(message)],
        "intent": "new_question",
        "question": message,
        "search_queries": [],
        "plan_notes": [],
        "sql": "",
        "sql_explanation": "",
        "sql_error": None,
        "attempts": 0,
        "result_columns": [],
        "result_rows": [],
        "error": "",
        "answer": "",
        "chart": None,
    }
    if not history:
        return update  # first message: nothing to relate it to, so no LLM call needed

    chain = INTENT_PROMPT | with_structured_output(runtime.context.llm, QuestionIntent)
    result = chain.invoke(
        {
            "table": state.get("table_title") or "geen",
            "history": format_history(history),
            "message": message,
        }
    )
    intent = result.intent
    if intent == "follow_up" and not state.get("table_id"):
        intent = "new_question"  # nothing to follow up on
    update["intent"] = intent
    update["question"] = result.standalone_question.strip() or message
    if intent == "off_topic":
        update["table_id"] = None
    return update
