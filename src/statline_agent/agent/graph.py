"""Assembles the nodes into a LangGraph StateGraph."""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from statline_agent.agent.context import AgentContext
from statline_agent.agent.nodes import search_tables, select_table
from statline_agent.agent.state import AgentState


def build_graph() -> CompiledStateGraph[AgentState, AgentContext, AgentState, AgentState]:
    graph = StateGraph(AgentState, context_schema=AgentContext)
    graph.add_node("search_tables", search_tables)
    graph.add_node("select_table", select_table)

    graph.add_edge(START, "search_tables")
    graph.add_edge("search_tables", "select_table")
    graph.add_edge("select_table", END)
    return graph.compile()
