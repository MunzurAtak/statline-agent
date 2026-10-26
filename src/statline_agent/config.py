"""Application settings, loaded from environment variables and an optional .env file."""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration lives here, so nothing else reads os.environ directly."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: Literal["groq", "ollama"] = "groq"
    llm_model: str = "openai/gpt-oss-120b"
    llm_temperature: float = 0.0

    groq_api_key: SecretStr | None = None
    ollama_base_url: str = "http://localhost:11434"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (read once per process)."""
    return Settings()
