"""Assembles the nodes into a LangGraph StateGraph.

understand_question -> search_tables -> select_table -> plan_query -> fetch_data
        |  (follow-up: same table) ------------------------^              |
        |  (off-topic) ------------------------------------> compose_answer <- write_sql <-> run_sql
"""

from typing import Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from statline_agent.agent.context import AgentContext
from statline_agent.agent.nodes import (
    compose_answer,
    fetch_data,
    plan_query,
    run_sql,
    search_tables,
    select_table,
    understand_question,
    write_sql,
)
from statline_agent.agent.nodes.data import MAX_SQL_ATTEMPTS
from statline_agent.agent.state import AgentState

# The Literal return types tell LangGraph which edges exist (used when drawing the graph).


def route_by_intent(
    state: AgentState,
) -> Literal["search_tables", "plan_query", "compose_answer"]:
    """Follow-ups reuse the current table (no search needed); off-topic gets a direct answer."""
    intent = state.get("intent", "new_question")
    if intent == "off_topic":
        return "compose_answer"
    return "plan_query" if intent == "follow_up" else "search_tables"


def route_after_selection(state: AgentState) -> Literal["plan_query", "compose_answer"]:
    """No suitable table: go straight to the answer, which explains that."""
    return "plan_query" if state.get("table_id") else "compose_answer"


def route_after_fetch(state: AgentState) -> Literal["write_sql", "compose_answer"]:
    return "compose_answer" if state.get("error") else "write_sql"


def route_after_sql(state: AgentState) -> Literal["write_sql", "compose_answer"]:
    """The self-correction loop: retry with the error message, at most MAX_SQL_ATTEMPTS times."""
    if state.get("sql_error") and state.get("attempts", 0) < MAX_SQL_ATTEMPTS:
        return "write_sql"
    return "compose_answer"


def build_graph(
    checkpointer: BaseCheckpointSaver[str] | None = None,
) -> CompiledStateGraph[AgentState, AgentContext, AgentState, AgentState]:
    """Without a checkpointer every call is independent; with one, state is kept per thread_id."""
    graph = StateGraph(AgentState, context_schema=AgentContext)
    graph.add_node("understand_question", understand_question)
    graph.add_node("search_tables", search_tables)
    graph.add_node("select_table", select_table)
    graph.add_node("plan_query", plan_query)
    graph.add_node("fetch_data", fetch_data)
    graph.add_node("write_sql", write_sql)
    graph.add_node("run_sql", run_sql)
    graph.add_node("compose_answer", compose_answer)

    graph.add_edge(START, "understand_question")
    graph.add_conditional_edges("understand_question", route_by_intent)
    graph.add_edge("search_tables", "select_table")
    graph.add_conditional_edges("select_table", route_after_selection)
    graph.add_edge("plan_query", "fetch_data")
    graph.add_conditional_edges("fetch_data", route_after_fetch)
    graph.add_edge("write_sql", "run_sql")
    graph.add_conditional_edges("run_sql", route_after_sql)
    graph.add_edge("compose_answer", END)
    return graph.compile(checkpointer=checkpointer)
