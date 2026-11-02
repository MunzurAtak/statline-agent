import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

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
