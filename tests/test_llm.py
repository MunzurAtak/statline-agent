from typing import Any

import httpx
import pytest
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from pydantic import BaseModel, SecretStr

from statline_agent.config import Settings
from statline_agent.llm import (
    LLMUnavailableError,
    MissingAPIKeyError,
    check_ollama,
    get_chat_model,
    with_structured_output,
)


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


def test_ollama_model_is_created_with_local_settings() -> None:
    settings = Settings(llm_provider="ollama", llm_model="qwen3:8b", _env_file=None)
    model = get_chat_model(settings)
    assert isinstance(model, ChatOllama)
    assert model.model == "qwen3:8b"
    assert model.num_ctx == 8192  # Ollama's default is too small for the prompts
    assert model.reasoning is False  # no slow "thinking" on a CPU
    assert model.keep_alive == "30m"


def test_models_default_per_provider() -> None:
    groq = Settings(llm_provider="groq", _env_file=None)
    assert (groq.llm_model, groq.judge_model) == ("openai/gpt-oss-120b", "openai/gpt-oss-20b")
    local = Settings(llm_provider="ollama", _env_file=None)
    assert (local.llm_model, local.judge_model) == ("qwen3.5:4b", "qwen3.5:4b")


def test_groq_model_name_with_ollama_is_a_clear_error() -> None:
    with pytest.raises(ValueError, match="Groq model name"):
        Settings(llm_provider="ollama", llm_model="openai/gpt-oss-120b", _env_file=None)


def ollama_http(models: list[str] | None) -> httpx.Client:
    """A fake Ollama server; None means it is not running."""

    def handler(request: httpx.Request) -> httpx.Response:
        if models is None:
            raise httpx.ConnectError("connection refused", request=request)
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": m} for m in models]})

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_check_ollama_accepts_a_downloaded_model() -> None:
    settings = Settings(llm_provider="ollama", _env_file=None)
    check_ollama(settings, "qwen3.5:4b", http=ollama_http(["qwen3.5:4b", "llama3.2:latest"]))
    check_ollama(settings, "llama3.2", http=ollama_http(["llama3.2:latest"]))  # implicit tag


def test_check_ollama_explains_what_to_do() -> None:
    settings = Settings(llm_provider="ollama", _env_file=None)
    with pytest.raises(LLMUnavailableError, match=r"ollama pull qwen3\.5:4b"):
        check_ollama(settings, "qwen3.5:4b", http=ollama_http(["llama3.2:latest"]))
    with pytest.raises(LLMUnavailableError, match="Start the Ollama app"):
        check_ollama(settings, "qwen3.5:4b", http=ollama_http(None))


class Answer(BaseModel):
    text: str


def test_structured_output_uses_strict_json_schema_for_groq() -> None:
    settings = Settings(llm_provider="groq", groq_api_key=SecretStr("test-key"), _env_file=None)
    runnable: Any = with_structured_output(get_chat_model(settings), Answer)
    response_format = runnable.first.kwargs["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True


def test_judge_uses_groq_when_a_key_is_set() -> None:
    local = Settings(llm_provider="ollama", groq_api_key=SecretStr("k"), _env_file=None)
    assert (local.judge_provider, local.judge_model) == ("groq", "openai/gpt-oss-20b")
    judge = get_chat_model(local, model=local.judge_model, provider=local.judge_provider)
    assert isinstance(judge, ChatGroq)


def test_only_the_default_groq_model_is_the_published_setup() -> None:
    assert Settings(llm_provider="groq", _env_file=None).is_published_setup
    dev = Settings(llm_provider="groq", llm_model="openai/gpt-oss-20b", _env_file=None)
    assert not dev.is_published_setup
    assert not Settings(llm_provider="ollama", _env_file=None).is_published_setup
