import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from statline_agent.config import Settings

RUNNER = Path(__file__).parents[1] / "evals" / "run_agent_eval.py"


def load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_agent_eval", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_update_readme_replaces_only_the_results_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = load_runner()
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{runner.START}\nold\n{runner.END}\noutro\n", "utf-8")
    monkeypatch.setattr(runner, "README", readme)

    runner.update_readme("| new |\n")

    assert readme.read_bytes() == (
        f"intro\n{runner.START}\n| new |\n{runner.END}\noutro\n".encode()
    )  # LF line endings, also on Windows


def test_repository_cases_load() -> None:
    runner = load_runner()
    cases = runner.load_cases(RUNNER.parent / "agent_cases.json")
    assert len({case.id for case in cases}) == len(cases)


class RateLimitError(Exception):
    """Same name as the error the Groq SDK raises on HTTP 429."""


class LimitedGraph:
    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        raise RateLimitError("Rate limit reached for model on tokens per day (TPD)")


class BrokenGraph:
    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        raise ValueError("boom")


def test_rate_limit_stops_the_run() -> None:
    runner = load_runner()
    case = runner.AgentCase(id="x", turns=["Hoeveel?"], expected_tables=["T1"])
    context = SimpleNamespace(cbs=None)
    with pytest.raises(runner.RateLimitReachedError, match="tokens per day"):
        runner.run_case(LimitedGraph(), context, None, case)


def test_other_errors_are_recorded_per_case() -> None:
    runner = load_runner()
    case = runner.AgentCase(id="x", turns=["Hoeveel?"], expected_tables=["T1"])
    result = runner.run_case(BrokenGraph(), SimpleNamespace(cbs=None), None, case)
    assert result.error == "ValueError: boom"
    assert not result.answer_ok


def test_local_runs_get_their_own_results_file() -> None:
    runner = load_runner()
    assert runner.results_name(Settings(llm_provider="groq", _env_file=None)) == "agent"
    local = Settings(llm_provider="ollama", llm_model="qwen3.5:4b", _env_file=None)
    assert runner.results_name(local) == "agent-qwen3.5-4b"
    dev = Settings(llm_provider="groq", llm_model="openai/gpt-oss-20b", _env_file=None)
    assert runner.results_name(dev) == "agent-gpt-oss-20b"  # never overwrites the published run


def write_result(path: Path, model: str, answer: float, provider: str | None = None) -> None:
    summary = {
        "table_accuracy": 1.0,
        "answer_accuracy": answer,
        "groundedness": 0.9,
        "median_seconds": 20.0,
    }
    payload = {"model": model, "judge": "openai/gpt-oss-20b", "summary": summary}
    if provider:
        payload["provider"] = provider
    path.write_text(json.dumps(payload), "utf-8")


def test_comparison_table_lists_the_published_setup_first(tmp_path: Path) -> None:
    runner = load_runner()
    write_result(tmp_path / "agent-qwen3.5-4b.json", "qwen3.5:4b", 0.95, "ollama")
    write_result(tmp_path / "agent.json", "openai/gpt-oss-120b", 0.9)  # older: no provider
    write_result(tmp_path / "agent-gpt-oss-20b.json", "openai/gpt-oss-20b", 0.8, "groq")
    lines = runner.comparison_table(tmp_path).splitlines()
    assert "`openai/gpt-oss-120b` | Groq (cloud) | 100% | 90% | 90% | 20.0 s" in lines[2]
    assert "`qwen3.5:4b` | Ollama (laptop CPU) | 100% | 95%" in lines[3]
    assert "`openai/gpt-oss-20b`" in lines[4]
