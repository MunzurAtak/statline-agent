"""Regenerate the agent diagram in README.md from the actual LangGraph definition.

Usage: uv run python scripts/draw_graph.py
The diagram lives between the GRAPH:START and GRAPH:END markers; a test checks it is current.
"""

from pathlib import Path

from statline_agent.agent import build_graph

README = Path(__file__).parents[1] / "README.md"
START, END = "<!-- GRAPH:START -->", "<!-- GRAPH:END -->"


def render_block() -> str:
    mermaid = build_graph().get_graph().draw_mermaid().strip()
    return f"{START}\n```mermaid\n{mermaid}\n```\n{END}"


def update_readme(text: str) -> str:
    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    return before + render_block() + after


def main() -> None:
    README.write_text(update_readme(README.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"Updated diagram in {README}")


if __name__ == "__main__":
    main()
