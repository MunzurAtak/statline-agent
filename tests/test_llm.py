from typing import Any

import pytest
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from pydantic import BaseModel, SecretStr

from statline_agent.config import Settings
from statline_agent.llm import MissingAPIKeyError, get_chat_model, with_structured_output


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


class Answer(BaseModel):
    text: str


def test_structured_output_uses_strict_json_schema_for_groq() -> None:
    settings = Settings(llm_provider="groq", groq_api_key=SecretStr("test-key"), _env_file=None)
    runnable: Any = with_structured_output(get_chat_model(settings), Answer)
    response_format = runnable.first.kwargs["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
