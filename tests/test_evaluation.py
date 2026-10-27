from pathlib import Path

import pytest

from statline_agent.evaluation import CaseResult, RetrievalCase, evaluate, load_cases

CASE = RetrievalCase("vraag", frozenset({"B", "C"}))


def test_first_hit_rank_uses_any_expected_table() -> None:
    assert CaseResult(CASE, ["A", "C", "B"]).first_hit_rank == 2
    assert CaseResult(CASE, ["A", "D"]).first_hit_rank is None


def test_hit_at_k_and_reciprocal_rank() -> None:
    result = CaseResult(CASE, ["A", "A2", "B"])
    assert not result.hit_at(2)
    assert result.hit_at(3)
    assert result.reciprocal_rank == pytest.approx(1 / 3)


def test_report_aggregates_over_cases() -> None:
    cases = [RetrievalCase("q1", frozenset({"X"})), RetrievalCase("q2", frozenset({"Y"}))]
    rankings = {"q1": ["X"], "q2": ["Z", "Y"]}
    report = evaluate("test", cases, lambda q: rankings[q])
    assert report.hit_rate(1) == 0.5
    assert report.hit_rate(5) == 1.0
    assert report.mrr == pytest.approx(0.75)
    assert report.summary()["method"] == "test"


def test_load_cases(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text('{"question": "q", "expected": ["A", "B"]}\n\n', encoding="utf-8")
    assert load_cases(path) == [RetrievalCase("q", frozenset({"A", "B"}))]


def test_repository_cases_file_is_valid() -> None:
    cases = load_cases(Path(__file__).parents[1] / "evals" / "retrieval_cases.jsonl")
    assert len(cases) >= 15
    assert all(case.expected for case in cases)
