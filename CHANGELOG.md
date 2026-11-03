# Changelog

All notable changes to this project. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versions follow
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-11-03

First public release.

### Agent

- LangGraph agent that answers questions about Dutch open data from CBS StatLine: understand
  the question, search ~1,300 tables, choose a table, plan the query, fetch the data, write
  and self-correct SQL (up to 3 attempts), and answer with a chart and a source link.
- Conversation memory with follow-up questions ("En Amsterdam?") via a checkpointer, and an
  optional human-in-the-loop step to confirm the chosen table (`interrupt`).
- Deterministic resolution of names to CBS codes, including the municipality/province
  ambiguity ("Utrecht" vs "provincie Utrecht").
- Measures are shown to the model with their CBS group ("Herkomst > Nederland") and the best
  matching totals are always included; CBS units ("x 1 000 euro") are applied in code.

### Retrieval

- Hybrid catalog search: local multilingual embeddings (fastembed) + BM25 with Dutch
  stemming, LLM query rewriting and Reciprocal Rank Fusion, with a retrieval evaluation.
- The index ships with the repository (float16 vectors, ~1 MB) and is refreshed monthly.

### Evaluation and observability

- End-to-end evaluation on 12 questions with LLM-free reference values, an LLM judge for
  groundedness, and a model comparison table in the README.
- Optional LangSmith tracing.

### Apps and delivery

- Streamlit chat app with live progress, charts, explanations and a per-visitor question
  limit for the public demo; streaming command line interface.
- LLM via Groq's free tier (`gpt-oss-120b`, `gpt-oss-20b` for development) or, experimentally,
  a local model with Ollama.
- Docker image on GitHub Container Registry; GitHub Actions for CI, Docker, the monthly index
  refresh and on-demand evaluations.

[0.1.0]: https://github.com/MunzurAtak/statline-agent/releases/tag/v0.1.0
