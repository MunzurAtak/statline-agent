import os

import pytest
from pydantic import SecretStr

from statline_agent.config import Settings
from statline_agent.llm import get_chat_model
from statline_agent.observability import configure_tracing


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "LANGSMITH_TRACING",
        "LANGSMITH_API_KEY",
        "LANGSMITH_PROJECT",
        "LANGSMITH_ENDPOINT",
    ):
        monkeypatch.delenv(name, raising=False)


def test_tracing_is_off_by_default() -> None:
    assert configure_tracing(Settings(_env_file=None)) is False
    assert os.environ["LANGSMITH_TRACING"] == "false"


def test_tracing_needs_an_api_key() -> None:
    assert configure_tracing(Settings(langsmith_tracing=True, _env_file=None)) is False


def test_tracing_exports_langsmith_variables() -> None:
    settings = Settings(
        langsmith_tracing=True,
        langsmith_api_key=SecretStr("ls-key"),
        langsmith_project="demo",
        langsmith_endpoint="https://eu.api.smith.langchain.com",
        _env_file=None,
    )
    assert configure_tracing(settings) is True
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_API_KEY"] == "ls-key"
    assert os.environ["LANGSMITH_PROJECT"] == "demo"
    assert os.environ["LANGSMITH_ENDPOINT"] == "https://eu.api.smith.langchain.com"


def test_groq_model_retries_and_model_override() -> None:
    settings = Settings(
        llm_provider="groq", groq_api_key=SecretStr("k"), llm_max_retries=7, _env_file=None
    )
    llm = get_chat_model(settings, model="openai/gpt-oss-20b")
    assert llm.max_retries == 7  # type: ignore[attr-defined]
    assert llm.model_name == "openai/gpt-oss-20b"  # type: ignore[attr-defined]
