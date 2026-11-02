"""Chat model factory, so the rest of the code never depends on a specific provider."""

from typing import cast

import httpx
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from statline_agent.config import Provider, Settings, get_settings


class MissingAPIKeyError(RuntimeError):
    """Raised when the selected provider needs an API key that is not configured."""


class LLMUnavailableError(RuntimeError):
    """Raised when the local model cannot be used (Ollama not running, model not pulled)."""


def _with_tag(model: str) -> str:
    return model if ":" in model else f"{model}:latest"


def check_ollama(settings: Settings, model: str, http: httpx.Client | None = None) -> None:
    """Fail fast with a clear message, instead of an obscure error at the first question."""
    client = http or httpx.Client(timeout=5)
    try:
        response = client.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags")
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise LLMUnavailableError(
            f"Ollama is not reachable at {settings.ollama_base_url}. Start the Ollama app "
            "(or run `ollama serve`) and try again."
        ) from exc
    finally:
        if http is None:
            client.close()
    available = {_with_tag(m.get("name", "")) for m in response.json().get("models", [])}
    if _with_tag(model) not in available:
        raise LLMUnavailableError(
            f"Ollama does not have the model {model!r}. Download it with: ollama pull {model}"
        )


def get_chat_model(
    settings: Settings | None = None,
    model: str | None = None,
    check: bool = False,
    provider: Provider | None = None,
) -> BaseChatModel:
    """Build the chat model for the configured provider (Groq or Ollama).

    `model` and `provider` override the configuration, e.g. for the evaluation judge. With
    `check`, a local model is verified first (Ollama running, model downloaded).
    """
    settings = settings or get_settings()
    model_name = model or settings.llm_model

    if (provider or settings.llm_provider) == "groq":
        if settings.groq_api_key is None:
            raise MissingAPIKeyError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        return init_chat_model(
            model_name,
            model_provider="groq",
            temperature=settings.llm_temperature,
            api_key=settings.groq_api_key.get_secret_value(),
            max_retries=settings.llm_max_retries,
        )

    if check:
        check_ollama(settings, model_name)
    return ChatOllama(
        model=model_name,
        temperature=settings.llm_temperature,
        base_url=settings.ollama_base_url,
        num_ctx=settings.ollama_num_ctx,
        reasoning=settings.ollama_reasoning,
        keep_alive=settings.ollama_keep_alive,
    )


def with_structured_output[T: BaseModel](
    llm: BaseChatModel, schema: type[T]
) -> Runnable[LanguageModelInput, T]:
    """Make the model return an instance of `schema`, using JSON-schema mode.

    The default (function calling) is unreliable with Groq's gpt-oss models: they often
    answer in plain text instead of calling the tool ("tool_use_failed"). In JSON-schema
    mode with strict=True, Groq constrains the output server-side to the schema.
    """
    if isinstance(llm, ChatGroq):
        runnable = llm.with_structured_output(schema, method="json_schema", strict=True)
    else:
        runnable = llm.with_structured_output(schema, method="json_schema")
    return cast(Runnable[LanguageModelInput, T], runnable)
