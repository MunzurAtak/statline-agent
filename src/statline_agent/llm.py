"""Chat model factory, so the rest of the code never depends on a specific provider."""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from statline_agent.config import Settings, get_settings


class MissingAPIKeyError(RuntimeError):
    """Raised when the selected provider needs an API key that is not configured."""


def get_chat_model(settings: Settings | None = None) -> BaseChatModel:
    """Build the chat model for the configured provider (Groq or Ollama)."""
    settings = settings or get_settings()

    if settings.llm_provider == "groq":
        if settings.groq_api_key is None:
            raise MissingAPIKeyError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        return init_chat_model(
            settings.llm_model,
            model_provider="groq",
            temperature=settings.llm_temperature,
            api_key=settings.groq_api_key.get_secret_value(),
        )

    return init_chat_model(
        settings.llm_model,
        model_provider="ollama",
        temperature=settings.llm_temperature,
        base_url=settings.ollama_base_url,
    )
