"""Application settings, loaded from environment variables and an optional .env file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration lives here, so nothing else reads os.environ directly."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: Literal["groq", "ollama"] = "groq"
    llm_model: str = "openai/gpt-oss-120b"
    llm_temperature: float = 0.0
    # Groq's free tier answers 429 when the per-minute limit is reached; the client then
    # waits (honouring the retry-after header) and tries again.
    llm_max_retries: int = 5
    # A different model judges the answers in the evaluation (see evals/run_agent_eval.py).
    judge_model: str = "openai/gpt-oss-20b"

    groq_api_key: SecretStr | None = None
    ollama_base_url: str = "http://localhost:11434"

    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    catalog_index_dir: Path = Path(".cache/catalog_index")

    # Optional tracing with LangSmith (free Developer plan). Off unless enabled in .env.
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "statline-agent"
    langsmith_endpoint: str | None = None  # e.g. https://eu.api.smith.langchain.com


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (read once per process)."""
    return Settings()
