"""Assembles the nodes into a LangGraph StateGraph.

search_tables -> select_table -> plan_query -> fetch_data -> write_sql <-> run_sql
                      |                            |                          |
                      +-------- (no table) ------- + ------ (error) --------- +--> compose_answer
"""

from typing import Literal

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
    write_sql,
)
from statline_agent.agent.nodes.data import MAX_SQL_ATTEMPTS
from statline_agent.agent.state import AgentState

# The Literal return types tell LangGraph which edges exist (used when drawing the graph).


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


def build_graph() -> CompiledStateGraph[AgentState, AgentContext, AgentState, AgentState]:
    graph = StateGraph(AgentState, context_schema=AgentContext)
    graph.add_node("search_tables", search_tables)
    graph.add_node("select_table", select_table)
    graph.add_node("plan_query", plan_query)
    graph.add_node("fetch_data", fetch_data)
    graph.add_node("write_sql", write_sql)
    graph.add_node("run_sql", run_sql)
    graph.add_node("compose_answer", compose_answer)

    graph.add_edge(START, "search_tables")
    graph.add_edge("search_tables", "select_table")
    graph.add_conditional_edges("select_table", route_after_selection)
    graph.add_edge("plan_query", "fetch_data")
    graph.add_conditional_edges("fetch_data", route_after_fetch)
    graph.add_edge("write_sql", "run_sql")
    graph.add_conditional_edges("run_sql", route_after_sql)
    graph.add_edge("compose_answer", END)
    return graph.compile()
