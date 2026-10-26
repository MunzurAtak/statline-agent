import pytest
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from pydantic import SecretStr

from statline_agent.config import Settings
from statline_agent.llm import MissingAPIKeyError, get_chat_model


def test_groq_requires_api_key() -> None:
    settings = Settings(llm_provider="groq", groq_api_key=None, _env_file=None)
    with pytest.raises(MissingAPIKeyError):
        get_chat_model(settings)


def test_groq_model_is_created() -> None:
    settings = Settings(
        llm_provider="groq",
        llm_model="openai/gpt-oss-120b",
        groq_api_key=SecretStr("test-key"),
        _env_file=None,
    )
    model = get_chat_model(settings)
    assert isinstance(model, ChatGroq)
    assert model.model_name == "openai/gpt-oss-120b"


def test_ollama_model_is_created() -> None:
    settings = Settings(llm_provider="ollama", llm_model="qwen3:8b", _env_file=None)
    model = get_chat_model(settings)
    assert isinstance(model, ChatOllama)
    assert model.model == "qwen3:8b"
