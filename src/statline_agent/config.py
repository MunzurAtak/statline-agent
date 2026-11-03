"""Application settings, loaded from environment variables and an optional .env file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Used when LLM_MODEL / JUDGE_MODEL are not set. Ollama runs locally: a small model that fits
# in 16 GB RAM next to the rest of the app, and it judges its own answers in local evals.
Provider = Literal["groq", "ollama"]
DEFAULT_MODELS: dict[str, str] = {"groq": "openai/gpt-oss-120b", "ollama": "qwen3.5:4b"}
DEFAULT_JUDGE_MODELS: dict[str, str] = {"groq": "openai/gpt-oss-20b"}


class Settings(BaseSettings):
    """All runtime configuration lives here, so nothing else reads os.environ directly."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: Provider = "groq"
    llm_model: str = ""  # empty: the provider's default (see DEFAULT_MODELS)
    llm_temperature: float = 0.0
    # Groq's free tier answers 429 when the per-minute limit is reached; the client then
    # waits (honouring the retry-after header) and tries again.
    llm_max_retries: int = 5
    # A different model judges the answers in the evaluation (see evals/run_agent_eval.py).
    # Empty: Groq's gpt-oss-20b whenever a Groq key is set, also for local runs, so every
    # model in the comparison is judged by the same judge. Without a key: the agent's model.
    judge_provider: Provider | None = None
    judge_model: str = ""

    groq_api_key: SecretStr | None = None

    ollama_base_url: str = "http://localhost:11434"
    # Ollama's default context window is too small for the planning and SQL prompts.
    ollama_num_ctx: int = 8192
    # "Thinking" makes small models much slower on a CPU and the agent does not need it.
    ollama_reasoning: bool = False
    ollama_keep_alive: str = "30m"  # keep the model in memory between questions

    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # Shipped with the repository (refreshed by .github/workflows/refresh-catalog.yml).
    catalog_index_dir: Path = Path("data/catalog_index")
    chat_db: Path = Path(".cache/chats.sqlite")  # conversations of the chat app
    # Public demo: questions per visitor, to protect the free daily token limit (0 = no limit).
    demo_max_questions: int = 0

    # Optional tracing with LangSmith (free Developer plan). Off unless enabled in .env.
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "statline-agent"
    langsmith_endpoint: str | None = None  # e.g. https://eu.api.smith.langchain.com

    @model_validator(mode="after")
    def _fill_in_models(self) -> Self:
        if self.llm_provider == "ollama" and self.llm_model.startswith("openai/"):
            raise ValueError(
                f"LLM_MODEL={self.llm_model} is a Groq model name. For Ollama use an Ollama "
                f"model such as {DEFAULT_MODELS['ollama']}, or leave LLM_MODEL empty."
            )
        self.llm_model = self.llm_model or DEFAULT_MODELS[self.llm_provider]
        if self.judge_provider is None:
            self.judge_provider = "groq" if self.groq_api_key else self.llm_provider
        if not self.judge_model:
            same = self.judge_provider == self.llm_provider
            fallback = self.llm_model if same else DEFAULT_MODELS[self.judge_provider]
            self.judge_model = DEFAULT_JUDGE_MODELS.get(self.judge_provider, fallback)
        return self

    @property
    def is_published_setup(self) -> bool:
        """The setup whose evaluation results are published in the README."""
        return self.llm_provider == "groq" and self.llm_model == DEFAULT_MODELS["groq"]


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (read once per process)."""
    return Settings()
