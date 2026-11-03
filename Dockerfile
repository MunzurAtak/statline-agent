# syntax=docker/dockerfile:1

# --- build: install the locked dependencies and the package into a virtualenv -------------
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    FASTEMBED_CACHE_PATH=/opt/fastembed
WORKDIR /app

# Dependencies first: this layer is reused as long as uv.lock does not change.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY README.md LICENSE ./
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

# Bake the embedding model (~220 MB) into the image, so the container starts without
# downloading it first.
RUN .venv/bin/python -c "from statline_agent.catalog import FastEmbedEmbeddings; \
from statline_agent.config import Settings; \
FastEmbedEmbeddings(Settings(_env_file=None).embedding_model).embed_query('test')"

# --- runtime: only the virtualenv, the model, the search index and the app ---------------
FROM python:3.12-slim

ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/opt/fastembed \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY --from=build /app/.venv /app/.venv
COPY --from=build /opt/fastembed /opt/fastembed
COPY data ./data
COPY streamlit_app.py ./
RUN mkdir .cache && chown app:app .cache

USER app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

# Configuration comes from environment variables (GROQ_API_KEY, LLM_MODEL, ...), see .env.example.
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
