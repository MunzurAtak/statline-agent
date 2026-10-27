"""Retrieval metrics: does the search put an acceptable CBS table near the top?"""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RetrievalCase:
    question: str
    expected: frozenset[str]  # any of these table ids counts as a correct answer


@dataclass(frozen=True)
class CaseResult:
    case: RetrievalCase
    ranking: list[str]

    @property
    def first_hit_rank(self) -> int | None:
        """1-based rank of the first acceptable table, or None if it was not found."""
        for rank, table_id in enumerate(self.ranking, start=1):
            if table_id in self.case.expected:
                return rank
        return None

    def hit_at(self, k: int) -> bool:
        rank = self.first_hit_rank
        return rank is not None and rank <= k

    @property
    def reciprocal_rank(self) -> float:
        rank = self.first_hit_rank
        return 1.0 / rank if rank else 0.0


@dataclass(frozen=True)
class RetrievalReport:
    method: str
    results: list[CaseResult]

    def hit_rate(self, k: int) -> float:
        return sum(r.hit_at(k) for r in self.results) / len(self.results)

    @property
    def mrr(self) -> float:
        """Mean Reciprocal Rank: 1.0 = always first, 0.5 = on average second, ..."""
        return sum(r.reciprocal_rank for r in self.results) / len(self.results)

    def summary(self) -> dict[str, float | str]:
        return {
            "method": self.method,
            "hit@1": round(self.hit_rate(1), 3),
            "hit@5": round(self.hit_rate(5), 3),
            "mrr": round(self.mrr, 3),
        }


def load_cases(path: Path) -> list[RetrievalCase]:
    """Read a JSONL file with {"question": ..., "expected": [...]} per line."""
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            cases.append(RetrievalCase(item["question"], frozenset(item["expected"])))
    return cases


def evaluate(
    method: str, cases: Sequence[RetrievalCase], rank: Callable[[str], list[str]]
) -> RetrievalReport:
    """Run `rank(question) -> [table_id, ...]` for every case and collect the results."""
    return RetrievalReport(method, [CaseResult(case, rank(case.question)) for case in cases])
