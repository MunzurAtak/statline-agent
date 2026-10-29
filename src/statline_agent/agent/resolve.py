"""Deterministic helpers that turn the LLM's plan into exact CBS codes.

The LLM names things in plain language ("Rotterdam", "2015 tot nu"); code does the exact
mapping to CBS codes (GM0599, 2015JJ00 ...). This keeps prompts small and makes the
mapping testable and reliable.
"""

import difflib
import re
from collections.abc import Sequence
from typing import Literal

from statline_agent.cbs import DimensionCode

Frequency = Literal["jaar", "kwartaal", "maand"]
FREQUENCY_CODES: dict[Frequency, str] = {"jaar": "JJ", "kwartaal": "KW", "maand": "MM"}
_PERIOD = re.compile(r"^(\d{4})([A-Z]{2})(\d{2})$")
_DEFAULT_TITLE = re.compile(r"^(totaal|waarde\b|nederland\b)", re.IGNORECASE)
# Common names that differ from the official CBS titles.
ALIASES = {"den haag": "'s-gravenhage", "den bosch": "'s-hertogenbosch"}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def resolve_names(
    names: Sequence[str], codes: Sequence[DimensionCode], cutoff: float = 0.8
) -> tuple[list[str], list[str]]:
    """Map plain-language names to code identifiers.

    Tries, in order: exact title, title starting with the name (e.g. "Zuid-Holland" ->
    "Zuid-Holland (PV)"), then fuzzy matching. Returns (codes, names that were not found).
    """
    by_title = {_normalize(c.title): c.identifier for c in codes}
    by_identifier = {c.identifier.lower(): c.identifier for c in codes}
    resolved: list[str] = []
    missing: list[str] = []
    for name in names:
        key = _normalize(name)
        key = ALIASES.get(key, key)
        if key in by_identifier:  # the LLM may also give a code directly
            match: str | None = by_identifier[key]
        elif key in by_title:
            match = by_title[key]
        else:
            prefixed = [title for title in by_title if title.startswith(key)]
            fuzzy = difflib.get_close_matches(key, list(by_title), n=1, cutoff=cutoff)
            best = min(prefixed, key=len) if prefixed else (fuzzy[0] if fuzzy else None)
            match = by_title[best] if best else None
        if match is None:
            missing.append(name)
        elif match not in resolved:
            resolved.append(match)
    return resolved, missing


def default_code(codes: Sequence[DimensionCode]) -> str | None:
    """The code to use when the question does not mention a dimension.

    CBS dimensions almost always contain a total ("Totaal mannen en vrouwen", "Nederland",
    "Waarde" for confidence-interval dimensions); otherwise the first code is the total.
    """
    for code in codes:
        if _DEFAULT_TITLE.match(code.title):
            return code.identifier
    return codes[0].identifier if codes else None


def select_periods(
    codes: Sequence[DimensionCode],
    frequency: Frequency,
    year_from: int = 0,
    year_to: int = 0,
    latest_only: bool = False,
) -> list[str]:
    """Pick period codes like 2024JJ00 (year), 2024KW02 (quarter) or 2024MM07 (month).

    year_from / year_to of 0 mean "no limit". Falls back to any frequency if the table
    does not have the requested one (e.g. a yearly table when asked per month).
    """
    parsed = [(c.identifier, m) for c in codes if (m := _PERIOD.match(c.identifier))]
    wanted = [(code, m) for code, m in parsed if m.group(2) == FREQUENCY_CODES[frequency]]
    candidates = wanted or parsed

    selected = [
        code
        for code, m in candidates
        if (not year_from or int(m.group(1)) >= year_from)
        and (not year_to or int(m.group(1)) <= year_to)
    ]
    if not selected:
        return []
    selected.sort()
    return selected[-1:] if latest_only else selected
