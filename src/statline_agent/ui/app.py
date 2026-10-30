"""Streamlit chat app. Run with: uv run streamlit run streamlit_app.py"""

import uuid
from dataclasses import dataclass, replace
from typing import Any, cast

import streamlit as st
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from statline_agent.agent import AgentContext, AgentState, build_graph, create_context
from statline_agent.agent.memory import sqlite_checkpointer
from statline_agent.charts import build_figure
from statline_agent.config import get_settings
from statline_agent.ui.turns import TurnResult, describe_step, friendly_error

Graph = CompiledStateGraph[AgentState, AgentContext, AgentState, AgentState]

EXAMPLES = [
    "Hoeveel inwoners heeft Rotterdam?",
    "Hoe is de gemiddelde WOZ-waarde in Utrecht sinds 2015 veranderd?",
    "Hoeveel faillissementen waren er in 2024?",
]


@dataclass
class Pending:
    """A turn that is paused, waiting for the user to confirm the table."""

    question: str
    state: AgentState
    steps: list[str]
    request: dict[str, Any]


@st.cache_resource(show_spinner="Agent laden (zoekindex en model)...")
def load_agent() -> tuple[Graph, AgentContext]:
    """Loaded once per server process and shared by all sessions."""
    checkpointer = sqlite_checkpointer(get_settings().catalog_index_dir.parent / "chats.sqlite")
    return build_graph(checkpointer=checkpointer), create_context()


def init_session() -> None:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.turns = []
        st.session_state.pending = None


# -- rendering -----------------------------------------------------------------------


def render_turn(turn: TurnResult, number: int) -> None:
    with st.chat_message("user"):
        st.markdown(turn.question)
    with st.chat_message("assistant"):
        render_answer(turn, number)


def render_answer(turn: TurnResult, number: int) -> None:
    """`number` makes widget keys unique: identical charts in one chat would otherwise clash."""
    if turn.error:
        st.error(turn.error)
        return
    st.markdown(turn.answer)
    for note in turn.notes:
        st.info(note)
    if turn.chart is not None:
        st.plotly_chart(build_figure(turn.chart, turn.rows), key=f"chart-{number}")
    if turn.source is None:
        return
    st.caption(f"Bron: CBS, [{turn.source.title} ({turn.source.table_id})]({turn.source.url})")
    with st.expander("Hoe kwam ik hieraan?"):
        st.markdown("**Stappen**\n" + "\n".join(f"1. {step}" for step in turn.steps))
        if turn.requested:
            st.markdown(f"**Gevraagd:** {turn.requested}")
        st.markdown(f"**Filters (CBS-codes):** {turn.filters}")
        if turn.sql:
            st.markdown(f"**SQL** (gelukt bij poging {turn.attempts}):")
            st.code(turn.sql, language="sql")
        if turn.rows:
            st.dataframe(turn.rows, key=f"rows-{number}")


# -- running the graph ---------------------------------------------------------------


def stream(payload: Any, state: AgentState, steps: list[str]) -> dict[str, Any] | None:
    """Run the graph until it finishes or pauses. Updates `state` and `steps` in place;
    returns the confirmation request if the graph paused (interrupt), else None.
    """
    graph, context = load_agent()
    context = replace(context, confirm_tables=st.session_state.get("confirm", False))
    thread_id = st.session_state.thread_id
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "run_name": "statline-agent",
        "tags": ["streamlit"],
        "metadata": {"thread_id": thread_id},
    }
    request = None
    with st.status("Bezig...", expanded=False) as status:
        for chunk in graph.stream(payload, config, context=context, stream_mode="updates"):
            for node, raw_update in chunk.items():
                if node == "__interrupt__":
                    request = raw_update[0].value
                    continue
                update = cast(AgentState, raw_update or {})  # nodes without changes stream None
                steps.append(describe_step(node, update))
                status.update(label=steps[-1])
                state.update(update)
        status.update(label="Kies een tabel" if request else "Klaar", state="complete")
    return request


def continue_turn(question: str, payload: Any, state: AgentState, steps: list[str]) -> None:
    """Stream, then either store the paused turn or the finished one, and redraw the page."""
    try:
        request = stream(payload, state, steps)
    except Exception as exc:  # e.g. the free LLM tier is rate limited; keep the app usable
        st.session_state.turns.append(TurnResult(question=question, error=friendly_error(exc)))
        st.session_state.pending = None
        st.rerun()
    if request is not None:
        st.session_state.pending = Pending(question, state, steps, request)
    else:
        st.session_state.turns.append(TurnResult.from_state(question, state, steps))
        st.session_state.pending = None
    st.rerun()


def render_pending(pending: Pending) -> None:
    """Human-in-the-loop: show the proposed table and alternatives as buttons."""
    proposed = pending.request["proposed"]
    with st.chat_message("user"):
        st.markdown(pending.question)
    with st.chat_message("assistant"):
        st.markdown(
            f"Ik stel deze tabel voor: **{proposed['title']}** ({proposed['table_id']}).  \n"
            f"_{proposed['reason']}_"
        )
        choice = None
        if st.button("Ja, gebruik deze tabel", type="primary", key="choose-proposed"):
            choice = ""
        alternatives = pending.request["alternatives"]
        if alternatives:
            st.markdown("Of kies een andere:")
        for option in alternatives:
            label = f"{option['title']} ({option['table_id']})"
            if st.button(label, key=f"choose-{option['table_id']}"):
                choice = option["table_id"]
        if st.button("Geen van deze", key="choose-none"):
            choice = "NONE"
    if choice is not None:
        continue_turn(pending.question, Command(resume=choice), pending.state, pending.steps)


# -- page ------------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="StatLine Agent", page_icon="📊")
    init_session()

    with st.sidebar:
        st.header("StatLine Agent")
        st.write(
            "Stel vragen over Nederlandse cijfers. De agent zoekt de juiste CBS-tabel, "
            "haalt de data op, schrijft zelf een analyse en laat zien hoe hij eraan kwam."
        )
        st.toggle("Laat mij de tabel bevestigen", key="confirm")
        if st.button("Nieuw gesprek"):
            del st.session_state.thread_id
            st.rerun()

    st.title("Vraag het CBS")
    for number, turn in enumerate(st.session_state.turns):
        render_turn(turn, number)

    pending: Pending | None = st.session_state.pending
    question = st.chat_input("Stel een vraag over CBS-cijfers", disabled=pending is not None)
    if pending is not None:
        render_pending(pending)
        return

    if not st.session_state.turns:
        st.write("Probeer bijvoorbeeld:")
        for example in EXAMPLES:
            if st.button(example):
                question = example

    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            continue_turn(question, {"question": question}, {}, [])
