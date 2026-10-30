"""Checkpointers: they store the graph state per conversation (thread_id).

This is what gives the agent memory between turns ("and now for Utrecht?") and what makes
human-in-the-loop possible: a paused graph is saved and resumed later.
"""

import sqlite3
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from statline_agent.agent.state import Candidate, Source
from statline_agent.charts import ChartSpec

# Our own Pydantic models in the state. LangGraph only deserializes types that are explicitly
# allowed, so a tampered checkpoint cannot instantiate arbitrary classes.
STATE_MODELS = (Candidate, ChartSpec, Source)


def make_serializer() -> JsonPlusSerializer:
    allowed = [(model.__module__, model.__name__) for model in STATE_MODELS]
    return JsonPlusSerializer(allowed_msgpack_modules=allowed)


def memory_checkpointer() -> InMemorySaver:
    """For tests and the CLI: memory lasts as long as the process."""
    return InMemorySaver(serde=make_serializer())


def sqlite_checkpointer(path: Path) -> SqliteSaver:
    """For the app: conversations survive a restart."""
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False)
    return SqliteSaver(connection, serde=make_serializer())
