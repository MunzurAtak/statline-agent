"""Command line interface: ask a question and watch the agent work step by step.

Usage:
    uv run statline-agent "Hoeveel inwoners heeft Rotterdam?"   # one question
    uv run statline-agent                                       # chat with follow-ups
"""

import argparse
import textwrap
import uuid
import webbrowser
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from statline_agent.agent import AgentContext, AgentState, build_graph, create_context
from statline_agent.agent.memory import memory_checkpointer
from statline_agent.charts import build_figure

MAX_ROWS_SHOWN = 10
DEFAULT_CHART_PATH = Path(".cache/charts/answer.html")


def _understand(update: AgentState) -> list[str]:
    return [f"  type: {update.get('intent')}", f"  vraag: {update.get('question', '')}"]


def _search(update: AgentState) -> list[str]:
    lines = [f"  zoekopdrachten: {update.get('search_queries', [])}"]
    lines += [f"  - {c.table_id:10} {c.title}" for c in update.get("candidates", [])]
    return lines


def _select(update: AgentState) -> list[str]:
    table = update.get("table_id") or "geen geschikte tabel"
    title = update.get("table_title", "")
    return [f"  gekozen: {table} {title}".rstrip(), f"  reden: {update.get('selection_reason')}"]


def _confirm(update: AgentState) -> list[str]:
    if not update:
        return ["  voorstel bevestigd"]
    table = update.get("table_id") or "geen tabel"
    return [f"  gekozen door gebruiker: {table} {update.get('table_title', '')}".rstrip()]


def _plan(update: AgentState) -> list[str]:
    lines = [f"  onderwerpen: {update.get('measure_titles', [])}"]
    requested = update.get("requested_filters", {}).items()
    lines += [f"  gevraagd {key}: {names}" for key, names in requested]
    lines += [f"  filter {key}: {codes}" for key, codes in update.get("filters", {}).items()]
    lines += [f"  let op: {note}" for note in update.get("plan_notes", [])]
    return lines


def _fetch(update: AgentState) -> list[str]:
    if update.get("error"):
        return [f"  fout: {update['error']}"]
    return [f"  {update.get('data_overview', '').splitlines()[0]}"]


def _write_sql(update: AgentState) -> list[str]:
    return [f"  {update.get('sql_explanation', '')}", f"  sql: {update.get('sql', '')}"]


def _run_sql(update: AgentState) -> list[str]:
    attempt = update.get("attempts")
    if update.get("sql_error"):
        return [f"  fout bij poging {attempt}: {update['sql_error']}"]
    rows = update.get("result_rows", [])
    return [
        f"  {len(rows)} rijen (poging {attempt})",
        *(f"  {row}" for row in rows[:MAX_ROWS_SHOWN]),
    ]


def _answer(update: AgentState) -> list[str]:
    lines = ["", *textwrap.wrap(update.get("answer", ""), width=100), ""]
    source = update.get("source")
    if source:
        lines.append(f"Bron: CBS, {source.title} ({source.table_id}) - {source.url}")
    return lines


FORMATTERS: dict[str, Callable[[AgentState], list[str]]] = {
    "understand_question": _understand,
    "search_tables": _search,
    "select_table": _select,
    "confirm_table": _confirm,
    "plan_query": _plan,
    "fetch_data": _fetch,
    "write_sql": _write_sql,
    "run_sql": _run_sql,
    "compose_answer": _answer,
}


def format_update(node: str, update: AgentState) -> list[str]:
    """Human-readable lines for one streamed node update."""
    formatter = FORMATTERS.get(node)
    body = (
        formatter(update) if formatter else [f"  {key}: {value}" for key, value in update.items()]
    )
    return [f"[{node}]", *body]


def save_chart(state: AgentState, path: Path) -> Path | None:
    """Render the chart spec from the final state to an HTML file (if there is a chart)."""
    chart = state.get("chart")
    if chart is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    build_figure(chart, state.get("result_rows", [])).write_html(path, include_plotlyjs="cdn")
    return path


Graph = CompiledStateGraph[AgentState, AgentContext, AgentState, AgentState]


def choose_table(request: dict[str, Any], read: Callable[[str], str] = input) -> str:
    """Show the proposed table and alternatives; return the chosen id ("" = accept proposal)."""
    proposed = request["proposed"]
    options = [proposed, *request["alternatives"]]
    print("\nVoorgestelde tabel en alternatieven:")
    for number, option in enumerate(options, start=1):
        marker = "  <- voorstel" if number == 1 else ""
        print(f"  {number}. {option['table_id']:10} {option['title']}{marker}")
    print("  0. Geen van deze")
    answer = read("Kies een nummer (Enter = voorstel): ").strip()
    if answer == "0":
        return "NONE"
    if answer.isdigit() and 1 <= int(answer) <= len(options):
        return str(options[int(answer) - 1]["table_id"])
    return ""


def ask(graph: Graph, context: AgentContext, config: RunnableConfig, question: str) -> AgentState:
    """Run one turn, printing every node update; returns the merged updates of this turn.

    When the graph pauses for confirmation (an interrupt), ask the user and resume it.
    """
    state: AgentState = {}
    payload: Any = {"question": question}
    while payload is not None:
        request = None
        # stream_mode="updates" yields {node_name: partial_state} after every node
        stream = graph.stream(payload, config, context=context, stream_mode="updates")
        for chunk in stream:
            for node, raw_update in chunk.items():
                if node == "__interrupt__":
                    request = raw_update[0].value
                    continue
                update = cast(AgentState, raw_update or {})  # nodes without changes stream None
                print("\n".join(format_update(node, update)))
                state.update(update)
        payload = Command(resume=choose_table(request)) if request else None
    return state


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ask questions about CBS open data.")
    parser.add_argument("question", nargs="?", help="Leave out to start a chat with follow-ups")
    parser.add_argument("--chart", type=Path, default=DEFAULT_CHART_PATH, help="HTML output path")
    parser.add_argument("--open", action="store_true", help="Open the chart in your browser")
    parser.add_argument("--auto", action="store_true", help="Do not ask to confirm the table")
    args = parser.parse_args(argv)

    graph = build_graph(checkpointer=memory_checkpointer())
    context = replace(create_context(), confirm_tables=not args.auto)
    # One thread = one conversation: the checkpointer keeps the state between turns.
    thread_id = str(uuid.uuid4())
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        # Shown in LangSmith (if tracing is on): filter traces by interface or conversation.
        "run_name": "statline-agent",
        "tags": ["cli"],
        "metadata": {"thread_id": thread_id},
    }

    def turn(question: str) -> None:
        state = ask(graph, context, config, question)
        chart_path = save_chart(state, args.chart)
        if chart_path:
            print(f"Grafiek opgeslagen in {chart_path}")
            if args.open:
                webbrowser.open(chart_path.resolve().as_uri())

    if args.question:
        turn(args.question)
        return
    print("Chat met de agent; stel gerust vervolgvragen. Laat de vraag leeg om te stoppen.")
    while question := input("\nVraag: ").strip():
        turn(question)


if __name__ == "__main__":
    main()
