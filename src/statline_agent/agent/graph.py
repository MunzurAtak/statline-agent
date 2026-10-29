"""Assembles the nodes into a LangGraph StateGraph."""

from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from statline_agent.agent.context import AgentContext
from statline_agent.agent.nodes import plan_query, search_tables, select_table
from statline_agent.agent.state import AgentState


def route_after_selection(state: AgentState) -> Literal["plan_query", "__end__"]:
    """Stop when no suitable table was found; otherwise continue with planning.

    The Literal return type tells LangGraph which edges exist (used when drawing the graph).
    """
    return "plan_query" if state.get("table_id") else "__end__"


def build_graph() -> CompiledStateGraph[AgentState, AgentContext, AgentState, AgentState]:
    graph = StateGraph(AgentState, context_schema=AgentContext)
    graph.add_node("search_tables", search_tables)
    graph.add_node("select_table", select_table)
    graph.add_node("plan_query", plan_query)

    graph.add_edge(START, "search_tables")
    graph.add_edge("search_tables", "select_table")
    graph.add_conditional_edges("select_table", route_after_selection)
    graph.add_edge("plan_query", END)
    return graph.compile()
