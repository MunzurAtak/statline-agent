"""Command line interface: ask a question and watch the agent work step by step.

Usage: uv run statline-agent "Hoeveel inwoners heeft Rotterdam?"
"""

import argparse
import textwrap
import webbrowser
from collections.abc import Callable
from pathlib import Path

from statline_agent.agent import AgentState, build_graph, create_context
from statline_agent.charts import build_figure

MAX_ROWS_SHOWN = 10
DEFAULT_CHART_PATH = Path(".cache/charts/answer.html")


def _search(update: AgentState) -> list[str]:
    lines = [f"  zoekopdrachten: {update.get('search_queries', [])}"]
    lines += [f"  - {c.table_id:10} {c.title}" for c in update.get("candidates", [])]
    return lines


def _select(update: AgentState) -> list[str]:
    table = update.get("table_id") or "geen geschikte tabel"
    title = update.get("table_title", "")
    return [f"  gekozen: {table} {title}".rstrip(), f"  reden: {update.get('selection_reason')}"]


def _plan(update: AgentState) -> list[str]:
    lines = [f"  onderwerpen: {update.get('measure_titles', [])}"]
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
    "search_tables": _search,
    "select_table": _select,
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ask a question about CBS open data.")
    parser.add_argument("question", help="Your question, e.g. 'Hoeveel inwoners heeft Rotterdam?'")
    parser.add_argument("--chart", type=Path, default=DEFAULT_CHART_PATH, help="HTML output path")
    parser.add_argument("--open", action="store_true", help="Open the chart in your browser")
    args = parser.parse_args(argv)

    graph = build_graph()
    context = create_context()
    state: AgentState = {}
    # stream_mode="updates" yields {node_name: partial_state} after every node
    for chunk in graph.stream({"question": args.question}, context=context, stream_mode="updates"):
        for node, update in chunk.items():
            print("\n".join(format_update(node, update)))
            state.update(update)

    chart_path = save_chart(state, args.chart)
    if chart_path:
        print(f"Grafiek opgeslagen in {chart_path}")
        if args.open:
            webbrowser.open(chart_path.resolve().as_uri())


if __name__ == "__main__":
    main()
