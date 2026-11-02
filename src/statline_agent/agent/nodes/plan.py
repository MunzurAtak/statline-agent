"""Turn the question + chosen table into exact OData filters (measures and dimension codes)."""

import difflib
import re
from collections.abc import Mapping, Sequence
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
from statline_agent.cbs import (
    CBSClient,
    CBSError,
    Dimension,
    DimensionCode,
    Measure,
    MeasureGroup,
    normalize_unit,
)
from statline_agent.llm import with_structured_output

MAX_MEASURES_IN_PROMPT = 25
MAX_TOTALS_IN_PROMPT = 5
TOTAL_PREFIXES = ("totaal", "totale")
MAX_CODES_IN_PROMPT = 30
GROUP_SEPARATOR = " > "


def measure_labels(measures: Sequence[Measure], groups: Sequence[MeasureGroup]) -> dict[str, str]:
    """Measure code -> title with its group path, e.g. 'Herkomst > Nederland'.

    Without the group, 'Nederland' looks like the population of the country (it is the
    number of people with a Dutch origin) and 'Totale groei' looks like a population size.
    """
    by_id = {g.id: g for g in groups}
    labels = {}
    for measure in measures:
        path: list[str] = []
        group_id = measure.measure_group_id
        for _ in range(len(by_id)):  # bounded, so a cycle in the groups cannot hang the agent
            if group_id is None or group_id not in by_id:
                break
            group = by_id[group_id]
            if group.title and group.title != measure.title:
                path.insert(0, group.title)
            group_id = group.parent_id
        labels[measure.identifier] = GROUP_SEPARATOR.join([*path, measure.title])
    return labels


def get_measure_groups(cbs: CBSClient, table_id: str) -> list[MeasureGroup]:
    """Measure groups are nice to have: without them the agent falls back to plain titles."""
    try:
        return cbs.get_measure_groups(table_id)
    except CBSError:
        return []


def is_total(measure: Measure) -> bool:
    return measure.title.strip().lower().startswith(TOTAL_PREFIXES)


def shortlist_measures(
    question: str,
    measures: Sequence[Measure],
    embeddings: Embeddings,
    k: int = MAX_MEASURES_IN_PROMPT,
) -> list[Measure]:
    """Big tables have hundreds of measures; only the most similar ones go into the prompt.

    The best-matching totals ("Totale bevolking") are always included. Similarity alone can
    push them out: in "Hoeveel inwoners heeft Nederland?" the place name matches dozens of
    origin measures ("Nederland", "Europa (exclusief Nederland)") better than the total.
    """
    if len(measures) <= k:
        return list(measures)
    query = np.array(embeddings.embed_query(question))
    matrix = np.array(embeddings.embed_documents([m.title for m in measures]))
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
    scores = matrix @ query / np.where(norms == 0, 1, norms)
    ranked = [int(i) for i in np.argsort(-scores, kind="stable")]

    totals = [i for i in ranked if is_total(measures[i])][: min(MAX_TOTALS_IN_PROMPT, k)]
    chosen = set(totals)
    for i in ranked:
        if len(chosen) >= k:
            break
        chosen.add(i)
    return [measures[i] for i in sorted(chosen)]  # keep the table's own (logical) order


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

- Kies 1 tot 3 onderwerpen (gebruik de codes tussen haakjes). Onderwerpen staan als \
"groep > onderwerp"; lees ze als geheel. "Herkomst > Nederland" gaat over mensen met een \
Nederlandse herkomst, niet over heel Nederland.
- Een plaatsnaam in de vraag (Nederland, een provincie of gemeente) is een regio, geen \
onderwerp. Heeft de tabel geen regio-indeling, dan gaan de cijfers over heel Nederland.
- Vraagt de vraag naar groei of verandering tussen jaren, kies dan het onderwerp zelf (de \
stand, zoals "Bevolking op 1 januari" of een gemiddelde waarde); de verandering wordt later \
berekend. Kies een groei- of mutatie-onderwerp (zoals "Totale groei") alleen als de vraag \
om dat cijfer zelf vraagt, bijvoorbeeld "hoeveel groeide de bevolking in 2024".
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


_CODE_IN_PARENTHESES = re.compile(r"\(([^()]+)\)\s*(?:\[[^\]]*\])?\s*$")


def match_measure(
    answer: str, shortlist: Sequence[Measure], labels: Mapping[str, str]
) -> Measure | None:
    """Map what the LLM returned to a measure: normally a code, but smaller models sometimes
    copy the whole prompt line ("... > Totaal bevolking (M000351_3) [x 1 000]") or a title.
    """
    text = answer.strip()
    by_code = {m.identifier.lower(): m for m in shortlist}
    found = _CODE_IN_PARENTHESES.search(text)
    for candidate in (text, found.group(1) if found else ""):
        if candidate.strip().lower() in by_code:
            return by_code[candidate.strip().lower()]
    by_label = {labels.get(m.identifier, m.title).lower(): m for m in shortlist}
    if text.lower() in by_label:
        return by_label[text.lower()]
    same_title = [m for m in shortlist if m.title.lower() == text.lower()]
    return same_title[0] if len(same_title) == 1 else None  # ambiguous titles: no guess


def _matches(name: str, dimension: Dimension) -> bool:
    names = {dimension.title.lower(), dimension.identifier.lower()}
    key = name.strip().lower()
    return key in names or bool(difflib.get_close_matches(key, list(names), n=1, cutoff=0.8))


def _requested_values(plan: QueryPlan, dimension: Dimension) -> list[str]:
    """Find the filter the LLM gave for this dimension (matching title or identifier)."""
    for choice in plan.filters:
        if _matches(choice.dimension, dimension):
            return choice.values
    return []


def unmatched_filters(plan: QueryPlan, dimensions: Sequence[Dimension]) -> list[DimensionChoice]:
    """Filters on a dimension the table does not have, e.g. a municipality in a national table."""
    return [
        choice
        for choice in plan.filters
        if choice.values and not any(_matches(choice.dimension, d) for d in dimensions)
    ]


def plan_query(state: AgentState, runtime: Runtime[AgentContext]) -> AgentState:
    ctx = runtime.context
    table_id = state["table_id"]
    assert table_id is not None  # guaranteed by the graph's routing

    measures = ctx.cbs.get_measures(table_id)
    dimensions = ctx.cbs.get_dimensions(table_id)
    codes = {d.identifier: ctx.cbs.get_dimension_codes(table_id, d.identifier) for d in dimensions}
    labels = measure_labels(measures, get_measure_groups(ctx.cbs, table_id))
    shortlist = shortlist_measures(state["question"], measures, ctx.embeddings)

    chain = PLAN_PROMPT | with_structured_output(ctx.llm, QueryPlan)
    plan = chain.invoke(
        {
            "question": state["question"],
            "table": f"{table_id} {state.get('table_title', '')}",
            "measures": "\n".join(
                f"- {labels[m.identifier]} ({m.identifier}) [{m.unit}]" for m in shortlist
            ),
            "dimensions": describe_dimensions(dimensions, codes),
        }
    )

    notes: list[str] = []
    chosen: list[Measure] = []
    for answer in plan.measures:
        measure = match_measure(answer, shortlist, labels)
        if measure is not None and measure not in chosen:
            chosen.append(measure)
    if not chosen:
        chosen = shortlist[:1]
        label = labels[chosen[0].identifier]
        notes.append(f"Onbekende onderwerpen {plan.measures}; '{label}' gebruikt.")
    filters: dict[str, list[str]] = {"Measure": [m.identifier for m in chosen]}
    requested: dict[str, list[str]] = {}  # what the LLM asked for, in plain language
    for choice in unmatched_filters(plan, dimensions):
        notes.append(
            f"De tabel heeft geen indeling {choice.dimension}; {', '.join(choice.values)} kon "
            "niet worden geselecteerd, dus de cijfers gelden voor het totaal."
        )

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
        "measure_titles": [labels[m.identifier] for m in chosen],
        "measure_units": [normalize_unit(m.unit)[1] for m in chosen],  # as in the data
        "plan_notes": notes,
    }
