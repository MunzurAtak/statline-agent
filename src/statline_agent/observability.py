"""Optional LangSmith tracing.

LangChain/LangGraph send traces when LANGSMITH_* environment variables are set. Our settings
come from .env via pydantic-settings, which does not export them, so we do it here: one
place, explicit, and only when tracing is switched on.
"""

import os

from statline_agent.config import Settings


def configure_tracing(settings: Settings) -> bool:
    """Enable LangSmith tracing if configured. Returns whether tracing is on."""
    if not settings.langsmith_tracing or settings.langsmith_api_key is None:
        os.environ["LANGSMITH_TRACING"] = "false"
        return False
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key.get_secret_value()
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    if settings.langsmith_endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    return True
