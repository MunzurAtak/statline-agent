"""End-to-end evaluation: ask the agent real questions and check the final answers.

Usage:
    uv run python evals/run_agent_eval.py               # all cases (takes a few minutes)
    uv run python evals/run_agent_eval.py --only woz-den-haag --no-judge

For every case it measures: the chosen table, whether the answer contains the reference
value (computed live by an LLM-free lookup), SQL attempts, time, LLM calls and tokens.
A second model judges whether the answer is grounded in the query result.
Writes evals/results/agent.json and evals/results/agent.md (used in the README).
"""

import argparse
import json
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
    mentions,
    summarize,
)
from statline_agent.config import get_settings
from statline_agent.llm import get_chat_model

EVALS_DIR = Path(__file__).parent
RESULTS_DIR = EVALS_DIR / "results"
README = EVALS_DIR.parent / "README.md"
START, END = "<!-- AGENT_EVAL:START -->", "<!-- AGENT_EVAL:END -->"


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
    result.expected = [round(e.value, 2) for e in expected]

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
                judge, result.question, state.get("result_columns", []), rows, result.answer
            )
            result.grounded, result.issues = verdict.grounded, verdict.issues
        except Exception as exc:
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
    parser.add_argument("--only", help="Run a single case by id")
    parser.add_argument("--no-judge", action="store_true", help="Skip the LLM-as-judge step")
    parser.add_argument(
        "--pause", type=float, default=10, help="Seconds between cases (rate limits)"
    )
    args = parser.parse_args()

    settings = get_settings()
    context = create_context(settings)
    graph = build_graph(checkpointer=memory_checkpointer())
    judge = None if args.no_judge else get_chat_model(settings, model=settings.judge_model)
    cases = [c for c in load_cases(EVALS_DIR / "agent_cases.json") if args.only in (None, c.id)]

    results = []
    for number, case in enumerate(cases, start=1):
        result = run_case(graph, context, judge, case)
        results.append(result)
        status = "OK " if result.table_ok and result.answer_ok else "MIS"
        print(
            f"[{number}/{len(cases)}] {status} {case.id}: table={result.table_id} "
            f"expected={result.expected} grounded={result.grounded} {result.error}"
        )
        if not result.answer_ok and result.answer:
            print(f"      answer: {result.answer[:200]}")
        if number < len(cases):
            time.sleep(args.pause)

    summary = summarize(results)
    print("\n" + json.dumps(summary, indent=2))
    if args.only:
        return  # partial runs do not overwrite the published results
    RESULTS_DIR.mkdir(exist_ok=True)
    payload = {
        "model": settings.llm_model,
        "judge": settings.judge_model,
        "summary": summary,
        "cases": [asdict(r) for r in results],
    }
    (RESULTS_DIR / "agent.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), "utf-8"
    )
    (RESULTS_DIR / "agent.md").write_text(markdown_report(results, summary), "utf-8")
    print(f"Saved results to {RESULTS_DIR}")
    update_readme(markdown_report(results, summary))


def update_readme(report: str) -> None:
    """Put the latest results table in the README, between the AGENT_EVAL markers."""
    text = README.read_text("utf-8")
    if START not in text:
        return
    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    README.write_text(f"{before}{START}\n{report}{END}{after}", "utf-8")
    print(f"Updated results in {README}")


if __name__ == "__main__":
    main()
