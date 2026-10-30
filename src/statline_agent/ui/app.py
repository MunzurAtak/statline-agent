"""Streamlit chat app. Run with: uv run streamlit run streamlit_app.py"""

import uuid
from typing import Any, cast

import streamlit as st
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from statline_agent.agent import AgentContext, AgentState, build_graph, create_context
from statline_agent.agent.memory import sqlite_checkpointer
from statline_agent.charts import build_figure
from statline_agent.config import get_settings
from statline_agent.ui.turns import TurnResult, describe_step

Graph = CompiledStateGraph[AgentState, AgentContext, AgentState, AgentState]

EXAMPLES = [
    "Hoeveel inwoners heeft Rotterdam?",
    "Hoe is de gemiddelde WOZ-waarde in Utrecht sinds 2015 veranderd?",
    "Hoeveel faillissementen waren er in 2024?",
]


@st.cache_resource(show_spinner="Agent laden (zoekindex en model)...")
def load_agent() -> tuple[Graph, AgentContext]:
    """Loaded once per server process and shared by all sessions."""
    checkpointer = sqlite_checkpointer(get_settings().catalog_index_dir.parent / "chats.sqlite")
    return build_graph(checkpointer=checkpointer), create_context()


def init_session() -> None:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.turns = []


def render_turn(turn: TurnResult, number: int) -> None:
    with st.chat_message("user"):
        st.markdown(turn.question)
    with st.chat_message("assistant"):
        render_answer(turn, number)


def render_answer(turn: TurnResult, number: int) -> None:
    """`number` makes widget keys unique: identical charts in one chat would otherwise clash."""
    st.markdown(turn.answer)
    if turn.chart is not None:
        st.plotly_chart(build_figure(turn.chart, turn.rows), key=f"chart-{number}")
    if turn.source is not None:
        st.caption(f"Bron: CBS, [{turn.source.title} ({turn.source.table_id})]({turn.source.url})")
        with st.expander("Hoe kwam ik hieraan?"):
            st.markdown(f"**Tabel:** {turn.source.table_id} {turn.source.title}")
            if turn.requested:
                st.markdown(f"**Gevraagd:** {turn.requested}")
            st.markdown(f"**Filters (CBS-codes):** {turn.filters}")
            for note in turn.notes:
                st.warning(note)
            if turn.sql:
                st.markdown(f"**SQL** (gelukt bij poging {turn.attempts}):")
                st.code(turn.sql, language="sql")
            if turn.rows:
                st.dataframe(turn.rows, key=f"rows-{number}")


def run_turn(question: str) -> TurnResult:
    graph, context = load_agent()
    config: RunnableConfig = {"configurable": {"thread_id": st.session_state.thread_id}}
    state: AgentState = {}
    payload: Any = {"question": question}
    with st.status("Bezig...", expanded=False) as status:
        for chunk in graph.stream(payload, config, context=context, stream_mode="updates"):
            for node, raw_update in chunk.items():
                update = cast(AgentState, raw_update or {})  # nodes without changes stream None
                status.update(label=describe_step(node, update))
                status.write(describe_step(node, update))
                state.update(update)
        status.update(label="Klaar", state="complete")
    return TurnResult.from_state(question, state)


def main() -> None:
    st.set_page_config(page_title="StatLine Agent", page_icon="📊")
    init_session()

    with st.sidebar:
        st.header("StatLine Agent")
        st.write(
            "Stel vragen over Nederlandse cijfers. De agent zoekt de juiste CBS-tabel, "
            "haalt de data op, schrijft zelf een analyse en laat zien hoe hij eraan kwam."
        )
        if st.button("Nieuw gesprek"):
            del st.session_state.thread_id
            st.rerun()

    st.title("Vraag het CBS")
    for number, turn in enumerate(st.session_state.turns):
        render_turn(turn, number)

    question = st.chat_input("Stel een vraag over CBS-cijfers")
    if not st.session_state.turns:
        st.write("Probeer bijvoorbeeld:")
        for example in EXAMPLES:
            if st.button(example):
                question = example

    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            turn = run_turn(question)
            render_answer(turn, len(st.session_state.turns))
        st.session_state.turns.append(turn)
