"""End-to-end evaluation: ask the agent real questions and check the final answers.

Usage:
    uv run python evals/run_agent_eval.py               # all cases (takes a few minutes)
    uv run python evals/run_agent_eval.py --only woz-den-haag --no-judge

For every case it measures: the chosen table, whether the answer contains the reference
value (computed live by an LLM-free lookup), SQL attempts, time, LLM calls and tokens.
A second model judges whether the answer is grounded in the query result.

With the published setup (Groq, gpt-oss-120b) it writes evals/results/agent.json/.md and
the per-case table in the README. Other models (gpt-oss-20b, a local Ollama model) write
evals/results/agent-<model>.json/.md, so they never replace the published numbers. Every
full run updates the model comparison table in the README.
"""

import argparse
import json
import re
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from statline_agent.agent import AgentContext, AgentState, build_graph, create_context
from statline_agent.agent.memory import memory_checkpointer
from statline_agent.agent_eval import (
    AgentCase,
    CaseResult,
    UsageCounter,
    compute_reference,
    judge_answer,
    judge_context,
    mentions,
    summarize,
)
from statline_agent.config import Settings, get_settings
from statline_agent.llm import get_chat_model

EVALS_DIR = Path(__file__).parent
RESULTS_DIR = EVALS_DIR / "results"
README = EVALS_DIR.parent / "README.md"
START, END = "<!-- AGENT_EVAL:START -->", "<!-- AGENT_EVAL:END -->"
COMPARE_START, COMPARE_END = "<!-- MODEL_COMPARISON:START -->", "<!-- MODEL_COMPARISON:END -->"
WHERE = {"groq": "Groq (cloud)", "ollama": "Ollama (laptop CPU)"}


class RateLimitReachedError(RuntimeError):
    """The provider refused more requests (e.g. Groq's free daily token limit).

    Every later case would fail too, so the run stops instead of publishing bad results.
    """


def is_rate_limit(exc: BaseException) -> bool:
    # By name, so the runner does not depend on one provider's SDK (groq, openai, ...).
    return type(exc).__name__ == "RateLimitError"


def results_name(settings: Settings) -> str:
    """'agent' for the published setup (Groq, gpt-oss-120b); 'agent-<model>' for others."""
    if settings.is_published_setup:
        return "agent"
    model = settings.llm_model.split("/")[-1]
    return "agent-" + re.sub(r"[^\w.-]+", "-", model).strip("-")


def load_cases(path: Path) -> list[AgentCase]:
    return [AgentCase.model_validate(item) for item in json.loads(path.read_text("utf-8"))]


def run_case(
    graph: Any, context: AgentContext, judge: BaseChatModel | None, case: AgentCase
) -> CaseResult:
    result = CaseResult(case_id=case.id, question=case.turns[-1])
    try:
        expected = [compute_reference(context.cbs, ref) for ref in case.references]
    except Exception as exc:
        result.error = f"reference query failed: {exc}"
        return result
    result.expected = [round(e.target, 2) for e in expected]

    counter = UsageCounter()
    config: RunnableConfig = {
        "configurable": {"thread_id": f"eval-{case.id}-{uuid.uuid4().hex[:8]}"},
        "callbacks": [counter],
        "run_name": f"eval:{case.id}",
        "tags": ["eval"],
        "metadata": {"case": case.id},
    }
    started = time.perf_counter()
    state: AgentState = {}
    try:
        for turn in case.turns:
            state = graph.invoke({"question": turn}, config, context=context)
    except Exception as exc:
        if is_rate_limit(exc):
            raise RateLimitReachedError(str(exc)[:300]) from exc
        result.error = f"{type(exc).__name__}: {str(exc)[:200]}"
    result.seconds = round(time.perf_counter() - started, 1)
    result.llm_calls, result.tokens = counter.calls, counter.tokens

    table = state.get("table_id")
    result.table_id = table
    result.answer = state.get("answer", "")
    result.attempts = state.get("attempts", 0)
    wanted = {t.lower() for t in case.expected_tables}
    if wanted:
        result.table_ok = bool(table) and table.lower() in wanted
        result.answer_ok = any(mentions(result.answer, e) for e in expected)
    else:  # e.g. off-topic: the right behaviour is not to answer with data
        result.table_ok = table is None
        result.answer_ok = table is None and state.get("source") is None

    rows = state.get("result_rows", [])
    if judge is not None and rows and result.answer:
        try:
            verdict = judge_answer(
                judge,
                result.question,
                state.get("result_columns", []),
                rows,
                result.answer,
                context=judge_context(state),
            )
            result.grounded, result.issues = verdict.grounded, verdict.issues
        except Exception as exc:
            if is_rate_limit(exc):
                raise RateLimitReachedError(str(exc)[:300]) from exc
            result.issues = [f"judge failed: {type(exc).__name__}"]
    return result


def markdown_report(results: list[CaseResult], summary: dict[str, float]) -> str:
    def mark(ok: bool | None) -> str:
        return "-" if ok is None else ("✅" if ok else "❌")

    lines = [
        "| Case | Table | Answer | Grounded | SQL attempts | Seconds | LLM calls |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.case_id} | {mark(r.table_ok)} {r.table_id or ''} | {mark(r.answer_ok)} "
            f"| {mark(r.grounded)} | {r.attempts or '-'} | {r.seconds} | {r.llm_calls} |"
        )
    lines += [
        "",
        f"**Table accuracy:** {summary['table_accuracy']:.0%} · "
        f"**Answer accuracy:** {summary['answer_accuracy']:.0%} · "
        f"**Groundedness:** {summary['groundedness']:.0%} · "
        f"**Median time:** {summary['median_seconds']} s · "
        f"**Avg LLM calls:** {summary['avg_llm_calls']}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="Run only these cases (comma-separated ids)")
    parser.add_argument("--no-judge", action="store_true", help="Skip the LLM-as-judge step")
    parser.add_argument(
        "--pause",
        type=float,
        default=None,
        help="Seconds between cases (default: 10 for Groq's rate limits, 0 for Ollama)",
    )
    args = parser.parse_args()

    settings = get_settings()
    pause = args.pause if args.pause is not None else (10 if settings.llm_provider == "groq" else 0)
    context = create_context(settings)
    graph = build_graph(checkpointer=memory_checkpointer())
    judge = None
    if not args.no_judge:
        judge = get_chat_model(
            settings, model=settings.judge_model, provider=settings.judge_provider, check=True
        )
    wanted = set(args.only.split(",")) if args.only else None
    cases = [
        c for c in load_cases(EVALS_DIR / "agent_cases.json") if wanted is None or c.id in wanted
    ]

    results = []
    for number, case in enumerate(cases, start=1):
        try:
            result = run_case(graph, context, judge, case)
        except RateLimitReachedError as exc:
            print(f"[{number}/{len(cases)}] STOP {case.id}: rate limit reached.\n      {exc}")
            print(
                "\nResults were not saved. The free Groq tier has a daily token limit per model; "
                "try again later, or run a few cases with --only."
            )
            return
        results.append(result)
        status = "OK " if result.table_ok and result.answer_ok else "MIS"
        print(
            f"[{number}/{len(cases)}] {status} {case.id}: table={result.table_id} "
            f"expected={result.expected} grounded={result.grounded} {result.error}"
        )
        if not result.answer_ok and result.answer:
            print(f"      answer: {result.answer[:200]}")
        if number < len(cases):
            time.sleep(pause)

    summary = summarize(results)
    print("\n" + json.dumps(summary, indent=2))
    if args.only:
        return  # partial runs do not overwrite the published results
    RESULTS_DIR.mkdir(exist_ok=True)
    name = results_name(settings)
    payload = {
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "judge": settings.judge_model,
        "summary": summary,
        "cases": [asdict(r) for r in results],
    }
    (RESULTS_DIR / f"{name}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), "utf-8", newline="\n"
    )
    report = markdown_report(results, summary)
    (RESULTS_DIR / f"{name}.md").write_text(report, "utf-8", newline="\n")
    print(f"Saved results to {RESULTS_DIR / name}.json/.md")
    if name == "agent":
        update_readme(report)
    update_readme(comparison_table(RESULTS_DIR), COMPARE_START, COMPARE_END)


def comparison_table(results_dir: Path) -> str:
    """One row per model that has a full evaluation run in results_dir."""
    rows = []
    for path in sorted(results_dir.glob("agent*.json")):
        payload = json.loads(path.read_text("utf-8"))
        rows.append((path.stem != "agent", -payload["summary"]["answer_accuracy"], payload))
    lines = [
        "| Model | Runs on | Table | Answer | Grounded | Median time | Judge |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, _, payload in sorted(rows, key=lambda row: row[:2]):  # published setup first
        summary = payload["summary"]
        provider = payload.get("provider", "groq")  # older results have no provider
        lines.append(
            f"| `{payload['model']}` | {WHERE.get(provider, provider)} "
            f"| {summary['table_accuracy']:.0%} | {summary['answer_accuracy']:.0%} "
            f"| {summary['groundedness']:.0%} | {summary['median_seconds']} s "
            f"| `{payload['judge']}` |"
        )
    return "\n".join(lines) + "\n"


def update_readme(report: str, start: str = START, end: str = END) -> None:
    """Put a generated table in the README, between two markers."""
    text = README.read_text("utf-8")
    if start not in text:
        return
    before, rest = text.split(start, 1)
    _, after = rest.split(end, 1)
    README.write_text(f"{before}{start}\n{report}{end}{after}", "utf-8", newline="\n")
    print(f"Updated {README.name} ({start})")


if __name__ == "__main__":
    main()
