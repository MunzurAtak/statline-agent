"""End-to-end evaluation of the agent: is the final answer right, and is it grounded?

Ground truth comes from *reference queries*: fixed CBS lookups without any LLM, computed live
at evaluation time. So the agent's own output is never used as the truth, and the cases keep
working when CBS publishes new figures.
"""

import re
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from statline_agent.agent.nodes.answer import format_result
from statline_agent.agent.resolve import default_code, resolve_names, select_periods
from statline_agent.cbs import CBSClient
from statline_agent.llm import with_structured_output

# -- cases ------------------------------------------------------------------------------


class ReferenceQuery(BaseModel):
    """A fixed, LLM-free lookup that produces the expected number.

    `measure` and the values in `filters` may be codes or titles (resolved deterministically).
    Dimensions that are not mentioned get their total, like in the agent.
    """

    table_id: str
    measure: str
    filters: dict[str, list[str]] = {}
    kind: Literal["latest", "year", "change_pct"] = "latest"
    year: int = 0  # for kind="year"
    start_year: int = 0  # for kind="change_pct"
    end_year: int = 0  # for kind="change_pct"; 0 = latest available


class AgentCase(BaseModel):
    id: str
    turns: list[str]  # the last turn is evaluated; earlier turns set up a conversation
    expected_tables: list[str] = []  # empty: the agent should not answer with data
    references: list[ReferenceQuery] = []  # any of these values counts as correct


@dataclass(frozen=True)
class Expected:
    value: float  # as stored by CBS, e.g. 374 for a WOZ value in "x 1 000 euro"
    is_percentage: bool = False
    scale: float = 1.0  # from the CBS unit: 1000 for "x 1 000 euro"

    @property
    def target(self) -> float:
        """The number a reader should see: the unit applied (374 -> 374.000 euro)."""
        return self.value if self.is_percentage else self.value * self.scale


# -- reference values ----------------------------------------------------------------------

_SPACES = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _SPACES.sub(" ", text).strip().lower()


def unit_scale(unit: str) -> float:
    unit = _norm(unit)
    if "mln" in unit or "miljoen" in unit:
        return 1_000_000.0
    if "1 000" in unit or "1000" in unit:
        return 1_000.0
    return 1.0


def reference_value(ref: ReferenceQuery, values: dict[str, float]) -> float:
    """Apply the reference kind to {period code: value} (year periods only)."""
    by_year = {int(code[:4]): value for code, value in sorted(values.items())}
    if not by_year:
        raise ValueError(f"No data for reference on {ref.table_id}")
    if ref.kind == "latest":
        return by_year[max(by_year)]
    if ref.kind == "year":
        return by_year[ref.year]
    end = by_year[ref.end_year] if ref.end_year else by_year[max(by_year)]
    start = by_year[ref.start_year]
    return (end - start) / start * 100


def compute_reference(cbs: CBSClient, ref: ReferenceQuery) -> Expected:
    measures = cbs.get_measures(ref.table_id)
    measure = next(
        (
            m
            for m in measures
            if ref.measure in (m.identifier, m.title) or _norm(m.title) == _norm(ref.measure)
        ),
        None,
    )
    if measure is None:
        raise ValueError(f"Measure {ref.measure!r} not found in {ref.table_id}")

    filters: dict[str, list[str]] = {"Measure": [measure.identifier]}
    period_dimension = ""
    for dim in cbs.get_dimensions(ref.table_id):
        codes = cbs.get_dimension_codes(ref.table_id, dim.identifier)
        if "Time" in dim.kind:
            period_dimension = dim.identifier
            filters[dim.identifier] = select_periods(codes, "jaar")
            continue
        names = ref.filters.get(dim.identifier) or ref.filters.get(dim.title) or []
        found, missing = resolve_names(names, codes)
        if missing:
            raise ValueError(f"Reference values {missing} not found in {dim.title}")
        default = default_code(codes)
        filters[dim.identifier] = found or ([default] if default else [])

    rows = cbs.get_observations(ref.table_id, filters)
    values = {
        str(row[period_dimension]): float(row["Value"])
        for row in rows
        if row.get("Measure") == measure.identifier and row.get("Value") is not None
    }
    return Expected(
        value=reference_value(ref, values),
        is_percentage=ref.kind == "change_pct",
        scale=unit_scale(measure.unit),
    )


# -- checking answers ----------------------------------------------------------------------

# 673.804 | 673 804 | 1.415.104 | 56,1 | 2026 | 1,4 miljoen  (\s also covers no-break spaces)
_NUMBER = re.compile(
    r"(?<![\d,.])(\d{1,3}(?:[.\s\xa0\u202f]\d{3})+|\d+)(?:,(\d+))?(?:\s*(miljoen|mln|duizend))?",
    re.IGNORECASE,
)
_MULTIPLIERS = {"miljoen": 1e6, "mln": 1e6, "duizend": 1e3}


def extract_numbers(text: str) -> list[tuple[float, bool]]:
    """Numbers in Dutch notation; the flag marks approximate wording ("1,4 miljoen")."""
    numbers = []
    for match in _NUMBER.finditer(text):
        integer = re.sub(r"[.\s\xa0\u202f]", "", match.group(1))
        value = float(f"{integer}.{match.group(2)}" if match.group(2) else integer)
        multiplier = match.group(3)
        if multiplier:
            value *= _MULTIPLIERS[multiplier.lower()]
        numbers.append((value, multiplier is not None))
    return numbers


def mentions(answer: str, expected: Expected) -> bool:
    """Does the answer contain the expected number (with sensible rounding tolerance)?

    The unit must be applied: for a WOZ value of 374 in "x 1 000 euro", only 374.000 (or
    "374 duizend") counts. "€ 374" would mislead a reader, so it is marked wrong.
    """
    target = expected.target
    for number, approximate in extract_numbers(answer):
        if expected.is_percentage:
            if abs(number - target) <= 0.6:
                return True
            continue
        # Exact figures may be rounded a little (674.000 for 673.804); "1,4 miljoen" more.
        tolerance = 0.02 if approximate else 0.001
        if abs(number - target) <= max(tolerance * abs(target), 0.5):
            return True
    return False


# -- LLM-as-judge --------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """\
Je controleert antwoorden van een data-assistent die vragen beantwoordt met CBS-cijfers. \
Beoordeel of de getallen en feitelijke beweringen in het antwoord kloppen met het \
queryresultaat en de context.

Dit is géén probleem:
- Nederlandse notatie en afronding: 673.804 is 673804, 56,1 is 56.1, "ruim 1,4 miljoen" is \
1415104.
- Het toepassen van de eenheid uit de context: 409 met eenheid "x 1 000 euro" is € 409.000.
- Eenvoudige berekeningen op het resultaat, zoals verschillen of percentages.
- Bron, tabel, onderwerp, regio, periode en eenheid die in de context staan.
- Uitleg over hoe een cijfer berekend is, zolang die klopt.

Een probleem is alleen een getal of bewering die het resultaat of de context tegenspreekt, of \
er niet uit af te leiden is. Zet grounded op true als er geen problemen zijn."""

JUDGE_HUMAN_PROMPT = """\
Vraag: {question}

Context:
{context}

Queryresultaat:
{result}

Antwoord: {answer}"""

JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [("system", JUDGE_SYSTEM_PROMPT), ("human", JUDGE_HUMAN_PROMPT)]
)


class Judgement(BaseModel):
    issues: list[str] = Field(
        description="Getallen of beweringen die niet kloppen of niet te onderbouwen zijn; "
        "leeg als er geen zijn"
    )
    grounded: bool = Field(description="true als de lijst met problemen leeg is")


def judge_context(state: Mapping[str, Any]) -> str:
    """What the agent knew besides the query result: table, measures with units, filters."""
    titles = state.get("measure_titles", [])
    units = state.get("measure_units", [])
    lines = [f"Tabel: {state.get('table_id', '')} {state.get('table_title', '')}".rstrip()]
    for i, title in enumerate(titles):
        unit = units[i] if i < len(units) else ""
        lines.append(f"Onderwerp: {title}" + (f" (eenheid: {unit})" if unit else ""))
    for dimension, values in state.get("requested_filters", {}).items():
        lines.append(f"{dimension}: {', '.join(values)}")
    return "\n".join(lines)


def judge_answer(
    llm: BaseChatModel,
    question: str,
    columns: list[str],
    rows: list[dict[str, Any]],
    answer: str,
    context: str = "",
) -> Judgement:
    chain = JUDGE_PROMPT | with_structured_output(llm, Judgement)
    verdict = chain.invoke(
        {
            "question": question,
            "context": context or "geen",
            "result": format_result(columns, rows),
            "answer": answer,
        }
    )
    # Small judge models sometimes say "not grounded" without naming a problem.
    if verdict.issues:
        return verdict.model_copy(update={"grounded": False})
    return verdict.model_copy(update={"grounded": True})


# -- bookkeeping ---------------------------------------------------------------------------


class UsageCounter(BaseCallbackHandler):
    """Counts LLM calls and tokens during a run (passed as a LangChain callback)."""

    def __init__(self) -> None:
        self.calls = 0
        self.tokens = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self.calls += 1
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
                if usage:
                    self.tokens += int(usage.get("total_tokens", 0))


@dataclass
class CaseResult:
    case_id: str
    question: str
    table_id: str | None = None
    table_ok: bool = False
    answer: str = ""
    expected: list[float] = field(default_factory=list)
    answer_ok: bool = False
    grounded: bool | None = None  # None: not judged (no data in the answer)
    issues: list[str] = field(default_factory=list)
    attempts: int = 0
    seconds: float = 0.0
    llm_calls: int = 0
    tokens: int = 0
    error: str = ""


def summarize(results: Sequence[CaseResult]) -> dict[str, float]:
    judged = [r for r in results if r.grounded is not None]
    with_sql = [r.attempts for r in results if r.attempts]
    return {
        "cases": len(results),
        "table_accuracy": round(sum(r.table_ok for r in results) / len(results), 3),
        "answer_accuracy": round(sum(r.answer_ok for r in results) / len(results), 3),
        "groundedness": round(sum(bool(r.grounded) for r in judged) / len(judged), 3)
        if judged
        else 0.0,
        "avg_sql_attempts": round(statistics.mean(with_sql), 2) if with_sql else 0.0,
        "median_seconds": round(statistics.median(r.seconds for r in results), 1),
        "avg_llm_calls": round(statistics.mean(r.llm_calls for r in results), 1),
        "avg_tokens": round(statistics.mean(r.tokens for r in results)),
    }
