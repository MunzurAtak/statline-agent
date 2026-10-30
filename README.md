# statline-agent

[![CI](https://github.com/MunzurAtak/statline-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/MunzurAtak/statline-agent/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-1.x-purple)
![License](https://img.shields.io/badge/license-MIT-green)

**An AI data analyst for Dutch open data.** Ask a question in plain language; the agent finds
the right table among ~1,300 datasets of [CBS StatLine](https://opendata.cbs.nl) (Statistics
Netherlands), fetches exactly the data it needs, writes and self-corrects SQL, and answers with
numbers, a chart and a source link. It runs as a chat app with follow-up questions ("En
Amsterdam?"), can ask you to confirm the table it chose, and shows how it got to every answer.

Built with **LangGraph** and **LangChain**, using only free tools and APIs.

```text
$ statline-agent "Hoe is de bevolking van Rotterdam en Amsterdam sinds 2015 veranderd?"
(output shortened)

[select_table]  03759ned  Bevolking op 1 januari en gemiddeld; geslacht, leeftijd en regio
[plan_query]    RegioS: GM0599, GM0363 · Perioden: 2015JJ00 … 2026JJ00
[run_sql]       24 rijen (poging 1)

In 2015 telde Amsterdam 821.752 inwoners en Rotterdam 623.652. Tegen 2026 steeg de bevolking
van Amsterdam tot 941.927 (+120.175, ≈14,6 %) en die van Rotterdam tot 673.804 (+50.152, ≈8,0 %).

Bron: CBS, Bevolking op 1 januari en gemiddeld (03759ned) - https://dataportal.cbs.nl/detail/CBS/03759ned
```

## How it works

The agent is a LangGraph `StateGraph`. Dashed edges are conditional: the agent can stop early
(no suitable table, data unavailable) and **retries failed SQL with the error message** (up to
3 attempts). This diagram is generated from the code (`scripts/draw_graph.py`).

<!-- GRAPH:START -->
```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	understand_question(understand_question)
	search_tables(search_tables)
	select_table(select_table)
	confirm_table(confirm_table)
	plan_query(plan_query)
	fetch_data(fetch_data)
	write_sql(write_sql)
	run_sql(run_sql)
	compose_answer(compose_answer)
	__end__([<p>__end__</p>]):::last
	__start__ --> understand_question;
	confirm_table -.-> compose_answer;
	confirm_table -.-> plan_query;
	fetch_data -.-> compose_answer;
	fetch_data -.-> write_sql;
	plan_query --> fetch_data;
	run_sql -.-> compose_answer;
	run_sql -.-> write_sql;
	search_tables --> select_table;
	select_table -.-> compose_answer;
	select_table -.-> confirm_table;
	understand_question -.-> compose_answer;
	understand_question -.-> plan_query;
	understand_question -.-> search_tables;
	write_sql --> run_sql;
	compose_answer --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```
<!-- GRAPH:END -->

| Node | What it does |
|---|---|
| `understand_question` | Reads the conversation: a follow-up ("En Utrecht?") is rewritten as a standalone question and reuses the current table; off-topic questions get a direct answer |
| `search_tables` | Hybrid search over the CBS catalog: vector search + BM25, with LLM query rewriting, fused with Reciprocal Rank Fusion |
| `select_table` | The LLM picks the best table from 8 candidates, seeing each table's measures (an LLM re-ranker) |
| `confirm_table` | Optional human-in-the-loop: the graph pauses (`interrupt`) so the user can confirm or change the table |
| `plan_query` | The LLM chooses measures and names regions/periods in plain language; deterministic code resolves them to CBS codes (`Den Haag` → `GM0518`) |
| `fetch_data` | Downloads only the filtered observations via the OData v4 API and caches them in DuckDB |
| `write_sql` ⇄ `run_sql` | The LLM writes DuckDB SQL; errors (or empty results) go back to the LLM with a hint |
| `compose_answer` | Plain-language answer from the query result only, a declarative chart spec (rendered with Plotly) and the source |

## Design decisions

- **The LLM decides, code executes.** The model never sees full code lists (1,000+ regions) and
  never writes plotting code. It names things; deterministic, tested code maps them to exact
  codes and renders charts. This keeps prompts small (Groq's free tier allows ~8k tokens/minute)
  and behaviour predictable.
- **Generated SQL is untrusted input.** Only a single `SELECT` passes DuckDB's own parser,
  external file/network access is disabled, and every query runs on its own cursor against a
  temporary view, so concurrent users cannot interfere.
- **Structured output everywhere.** Every LLM call returns a Pydantic model. Groq's gpt-oss
  models were unreliable with function calling (`tool_use_failed`), so the agent uses strict
  JSON-schema mode, where the server enforces the schema.
- **Dependency injection via LangGraph's runtime context.** Nodes get the LLM, retriever and
  data store from `runtime.context`, which makes the whole graph testable with a fake LLM.
- **Memory and human-in-the-loop via a checkpointer.** The state of each conversation is saved
  per `thread_id` (SQLite in the app). The confirmation step is a separate node, because
  LangGraph re-runs an interrupted node on resume: the LLM call that chose the table is not
  repeated (or paid for twice).
- **Ambiguity is handled in code, not left to the model.** "Utrecht" is both a municipality
  and a province. Plain names resolve to the municipality; "provincie Utrecht" to the province,
  and if the model drops the word "provincie", code restores it from the question.

## Retrieval evaluation

Finding the right table is the hardest step. I measured it on 20 realistic questions with known
correct tables ([`evals/retrieval_cases.jsonl`](evals/retrieval_cases.jsonl)):

| Approach | hit@1 | hit@5 | MRR |
|---|---|---|---|
| BM25 only | 0.35 | 0.50 | 0.43 |
| Vector only (multilingual MiniLM) | 0.40 | 0.60 | 0.50 |
| Hybrid (RRF) | 0.45 | 0.55 | 0.51 |
| Hybrid + LLM query rewriting | 0.50 | 0.70 | 0.58 |
| **+ topic-only rewrites and a High Value Dataset boost** | **0.50** | **0.75** | **0.60** |

What I learned along the way:

- **Adding more metadata made search worse.** Indexing each table's measure titles lowered
  hit@5 (0.65 → 0.55): they are often generic ("Totaal", "Mannen") and crowded out the useful
  text in the embedding model's 128-token window. An ablation showed the real problem was on
  the question side: users say *inwoners*, *failliet* and *inflatie*; CBS titles say
  *bevolking*, *faillissementen* and *consumentenprijzen*.
- **LLM query rewriting fixed most of that vocabulary gap.** The prompt's examples deliberately
  do not overlap with the test questions, so the evaluation measures the method, not the prompt.
- **A small boost for CBS's core tables.** The EU designates 47 CBS tables as *High Value
  Datasets* (population, labour, prices…). When one of them is already among the results, it
  gets a small extra rank in the fusion. This helped a little (one question = 0.05 on this
  set), and I found the idea by looking at misses, so take the gain with some caution.
- Most remaining misses rank 6–10; the LLM chooses from the top 8, so the effective recall is
  about 0.85.

## End-to-end evaluation

Retrieval is only one step, so the whole agent is evaluated too: 12 questions, including
follow-up questions, a municipality/province ambiguity and an off-topic question
([`evals/agent_cases.json`](evals/agent_cases.json)).

- **Ground truth without an LLM.** Each case has a *reference query*: a fixed CBS lookup
  (table, measure, region codes) computed live during the evaluation. The agent's own output is
  never used as the truth, and the cases keep working when CBS publishes new figures.
- **Answer accuracy:** does the answer contain the reference number? Dutch number formats
  ("673.804", "56,1%", "1,4 miljoen") are parsed with a small rounding tolerance.
- **Groundedness:** a second model (LLM-as-judge) checks whether every claim in the answer is
  supported by the query result.
- Also measured: chosen table, SQL attempts, time, LLM calls and tokens per question.

<!-- AGENT_EVAL:START -->
_Run `uv run python evals/run_agent_eval.py` to fill in this table._
<!-- AGENT_EVAL:END -->

## Observability

Tracing with [LangSmith](https://smith.langchain.com) (free Developer plan) is optional: set
`LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` in `.env`. Every question then becomes a
trace with all nodes, each LLM call (prompt, output, tokens) and the search results. Runs are
tagged by interface (`cli`, `streamlit`, `eval`) and carry the conversation's `thread_id`.

## Tech stack (all free)

| | |
|---|---|
| Agent | LangGraph 1.x, LangChain |
| LLM | Groq free tier (`openai/gpt-oss-120b`) or local Ollama |
| Retrieval | fastembed (local multilingual embeddings), BM25 with Dutch stemming, LangChain retriever |
| Data | CBS OData v4 API, DuckDB, pandas |
| Charts | Plotly |
| UI | Streamlit chat app (with `AppTest` UI tests) |
| Observability | LangSmith tracing (optional) |
| Quality | pytest (130+ tests, no network or API key needed), mypy `--strict`, ruff, pre-commit, GitHub Actions |

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and a free [Groq API key](https://console.groq.com).

```bash
git clone https://github.com/MunzurAtak/statline-agent.git
cd statline-agent
cp .env.example .env            # add your GROQ_API_KEY
uv sync
uv run python scripts/build_catalog_index.py    # ~2-5 min the first time
uv run streamlit run streamlit_app.py           # chat app on http://localhost:8501
```

Or use the command line (leave out the question for a chat with follow-ups):

```bash
uv run statline-agent "Wat is de gemiddelde WOZ-waarde in Den Haag?" --open
```

Run the checks and the evaluations:

```bash
uv run pytest
uv run mypy
uv run python evals/run_retrieval_eval.py
uv run python evals/run_agent_eval.py
```

## Project structure

```text
src/statline_agent/
├── agent/        # LangGraph state, nodes, graph and runtime context
├── catalog/      # catalog index, hybrid retriever, query rewriting
├── cbs/          # typed CBS OData client and DuckDB cache (with safe SQL execution)
├── ui/           # Streamlit chat app (rendering) and testable helpers
├── charts.py     # chart spec -> Plotly figure
├── evaluation.py # retrieval metrics (hit@k, MRR)
├── agent_eval.py # end-to-end evaluation: reference queries, answer checks, LLM judge
├── observability.py  # optional LangSmith tracing
└── cli.py        # streaming command line interface
streamlit_app.py  # entry point of the chat app
evals/            # test questions, evaluation runners and published results
scripts/          # index builder, search comparison, diagram generator
tests/            # unit and graph tests with recorded CBS responses and a fake LLM
```

## Roadmap

- [x] Typed CBS client, DuckDB cache, CI
- [x] Hybrid catalog search with evaluation
- [x] Agent: table selection, query planning, self-correcting SQL, answer with chart
- [x] Conversation memory, follow-up questions and human-in-the-loop (checkpointer, interrupts)
- [x] Streamlit chat app with live progress, charts and explanations
- [x] End-to-end evaluation and tracing with LangSmith
- [ ] Public demo and Docker image

## License

MIT. Data: © CBS, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
