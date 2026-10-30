"""Turn the question + chosen table into exact OData filters (measures and dimension codes)."""

import difflib
from collections.abc import Sequence
from typing import Literal

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from statline_agent.agent.context import AgentContext
from statline_agent.agent.resolve import (
    add_qualifiers,
    default_code,
    resolve_names,
    select_periods,
)
from statline_agent.agent.state import AgentState
from statline_agent.cbs import Dimension, DimensionCode, Measure
from statline_agent.llm import with_structured_output

MAX_MEASURES_IN_PROMPT = 25
MAX_CODES_IN_PROMPT = 30


def shortlist_measures(
    question: str,
    measures: Sequence[Measure],
    embeddings: Embeddings,
    k: int = MAX_MEASURES_IN_PROMPT,
) -> list[Measure]:
    """Big tables have hundreds of measures; only the most similar ones go into the prompt."""
    if len(measures) <= k:
        return list(measures)
    query = np.array(embeddings.embed_query(question))
    matrix = np.array(embeddings.embed_documents([m.title for m in measures]))
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
    scores = matrix @ query / np.where(norms == 0, 1, norms)
    top = np.argsort(-scores)[:k]
    return [measures[i] for i in sorted(top)]  # keep the table's own (logical) order


class DimensionChoice(BaseModel):
    dimension: str = Field(description="Naam van de indeling, bijvoorbeeld Regio's")
    values: list[str] = Field(description="Gewenste waarden in gewone taal, bijv. Rotterdam")


class QueryPlan(BaseModel):
    measures: list[str] = Field(description="Codes van de benodigde onderwerpen (1 tot 3)")
    filters: list[DimensionChoice] = Field(
        description="Alleen indelingen die in de vraag genoemd worden"
    )
    frequency: Literal["jaar", "kwartaal", "maand"]
    year_from: int = Field(description="Eerste jaar, of 0 als de vraag geen begin noemt")
    year_to: int = Field(description="Laatste jaar, of 0 als de vraag geen einde noemt")
    latest_only: bool = Field(description="true als alleen de meest recente periode nodig is")


PLAN_SYSTEM_PROMPT = """\
Je bent een data-analist. Bepaal welke gegevens uit de CBS-tabel nodig zijn om de vraag te \
beantwoorden.

- Kies 1 tot 3 onderwerpen (gebruik de codes tussen haakjes).
- Noem bij filters alleen indelingen die in de vraag voorkomen, met waarden in gewone taal \
(bijvoorbeeld de naam van een gemeente). Neem "provincie" of "gemeente" over als de vraag \
dat noemt, bijvoorbeeld "provincie Utrecht". Niet-genoemde indelingen krijgen het totaal.
- Vraagt de vraag naar een actuele stand ("hoeveel", zonder jaartal), zet latest_only op true.
- Vraagt de vraag naar een ontwikkeling of een periode, vul year_from en/of year_to in."""

PLAN_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", PLAN_SYSTEM_PROMPT),
        (
            "human",
            "Vraag: {question}\n\nTabel: {table}\n\nOnderwerpen:\n{measures}\n\n{dimensions}",
        ),
    ]
)


def is_time_dimension(dimension: Dimension) -> bool:
    return "Time" in dimension.kind


def describe_dimensions(
    dimensions: Sequence[Dimension], codes: dict[str, list[DimensionCode]]
) -> str:
    lines = ["Indelingen:"]
    for dim in dimensions:
        dim_codes = codes[dim.identifier]
        if is_time_dimension(dim):
            lines.append(f"- {dim.title}: {dim_codes[0].title} t/m {dim_codes[-1].title}")
        elif len(dim_codes) <= MAX_CODES_IN_PROMPT:
            lines.append(f"- {dim.title}: " + "; ".join(c.title for c in dim_codes))
        else:
            lines.append(f"- {dim.title}: {len(dim_codes)} waarden (noem ze bij naam)")
    return "\n".join(lines)


def _requested_values(plan: QueryPlan, dimension: Dimension) -> list[str]:
    """Find the filter the LLM gave for this dimension (matching title or identifier)."""
    names = {dimension.title.lower(), dimension.identifier.lower()}
    for choice in plan.filters:
        key = choice.dimension.strip().lower()
        if key in names or difflib.get_close_matches(key, list(names), n=1, cutoff=0.8):
            return choice.values
    return []


def plan_query(state: AgentState, runtime: Runtime[AgentContext]) -> AgentState:
    ctx = runtime.context
    table_id = state["table_id"]
    assert table_id is not None  # guaranteed by the graph's routing

    measures = ctx.cbs.get_measures(table_id)
    dimensions = ctx.cbs.get_dimensions(table_id)
    codes = {d.identifier: ctx.cbs.get_dimension_codes(table_id, d.identifier) for d in dimensions}
    shortlist = shortlist_measures(state["question"], measures, ctx.embeddings)

    chain = PLAN_PROMPT | with_structured_output(ctx.llm, QueryPlan)
    plan = chain.invoke(
        {
            "question": state["question"],
            "table": f"{table_id} {state.get('table_title', '')}",
            "measures": "\n".join(f"- {m.title} ({m.identifier}) [{m.unit}]" for m in shortlist),
            "dimensions": describe_dimensions(dimensions, codes),
        }
    )

    notes: list[str] = []
    by_code = {m.identifier.lower(): m for m in shortlist}
    chosen = [by_code[c.lower()] for c in plan.measures if c.lower() in by_code]
    if not chosen:
        chosen = shortlist[:1]
        notes.append(f"Onbekende onderwerpen {plan.measures}; '{chosen[0].title}' gebruikt.")
    filters: dict[str, list[str]] = {"Measure": [m.identifier for m in chosen]}
    requested: dict[str, list[str]] = {}  # what the LLM asked for, in plain language

    for dim in dimensions:
        dim_codes = codes[dim.identifier]
        if is_time_dimension(dim):
            periods = select_periods(
                dim_codes, plan.frequency, plan.year_from, plan.year_to, plan.latest_only
            )
            if not periods:
                periods = select_periods(dim_codes, plan.frequency, latest_only=True)
                notes.append("Gevraagde periode niet beschikbaar; meest recente periode gebruikt.")
            filters[dim.identifier] = periods
            continue

        names = add_qualifiers(_requested_values(plan, dim), state["question"])
        if names:
            requested[dim.title] = names
        found, missing = resolve_names(names, dim_codes)
        if missing:
            notes.append(f"Niet gevonden in {dim.title}: {', '.join(missing)}.")
        default = default_code(dim_codes)
        if found:
            filters[dim.identifier] = found
        elif default:
            filters[dim.identifier] = [default]

    return {
        "filters": filters,
        "requested_filters": requested,
        "measure_titles": [m.title for m in chosen],
        "plan_notes": notes,
    }
