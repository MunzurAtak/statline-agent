"""Command line interface: ask a question and watch the agent work step by step.

Usage: uv run statline-agent "Hoeveel inwoners heeft Rotterdam?"
"""

import argparse
from collections.abc import Callable
from typing import Any

from statline_agent.agent import AgentState, build_graph, create_context


def _search(update: AgentState) -> list[str]:
    lines = [f"  zoekopdrachten: {update.get('search_queries', [])}"]
    lines += [f"  - {c.table_id:10} {c.title}" for c in update.get("candidates", [])]
    return lines


def _select(update: AgentState) -> list[str]:
    table = update.get("table_id") or "geen geschikte tabel"
    title = update.get("table_title", "")
    return [f"  gekozen: {table} {title}".rstrip(), f"  reden: {update.get('selection_reason')}"]


FORMATTERS: dict[str, Callable[[AgentState], list[str]]] = {
    "search_tables": _search,
    "select_table": _select,
}


def format_update(node: str, update: AgentState) -> list[str]:
    """Human-readable lines for one streamed node update."""
    formatter = FORMATTERS.get(node)
    body = (
        formatter(update) if formatter else [f"  {key}: {value}" for key, value in update.items()]
    )
    return [f"[{node}]", *body]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ask a question about CBS open data.")
    parser.add_argument("question", help="Your question, e.g. 'Hoeveel inwoners heeft Rotterdam?'")
    args = parser.parse_args(argv)

    graph = build_graph()
    context = create_context()
    # stream_mode="updates" yields {node_name: partial_state} after every node
    chunk: dict[str, Any]
    for chunk in graph.stream({"question": args.question}, context=context, stream_mode="updates"):
        for node, update in chunk.items():
            print("\n".join(format_update(node, update)))


if __name__ == "__main__":
    main()
