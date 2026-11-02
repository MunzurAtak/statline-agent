"""CBS publishes many figures in thousands or millions ("x 1 000 euro", "mln euro").

The agent converts them to plain numbers when the data is loaded, so neither the SQL nor the
answer has to apply the unit. Leaving that to the LLM went wrong in both directions: a WOZ
value of 409 (x 1 000 euro) became "€ 409", and a population in "x 1 000" got multiplied twice.
"""

import re

# Only a leading multiplier is a scale: "per 1 000 inwoners" is a rate and stays as it is.
# CBS writes both "x 1 000 euro" and "1 000 euro" (e.g. the WOZ value in 85036NED).
_X = r"^(?:x\s*)?"
_SCALES = [
    (re.compile(_X + r"(?:mld|miljard\b|1[\s.]?000[\s.]?000[\s.]?000\b)"), 1e9),
    (re.compile(_X + r"(?:mln|miljoen\b|1[\s.]?000[\s.]?000\b)"), 1e6),
    (re.compile(_X + r"1[\s.]?000\b"), 1e3),
]
DEFAULT_UNIT = "aantal"


def normalize_unit(unit: str) -> tuple[float, str]:
    """'x 1 000 euro' -> (1000.0, 'euro'); 'x 1 000' -> (1000.0, 'aantal'); '%' -> (1.0, '%')."""
    text = " ".join(unit.split())
    for pattern, factor in _SCALES:
        match = pattern.match(text.lower())
        if match:
            rest = text[match.end() :].strip()
            return factor, rest or DEFAULT_UNIT
    return 1.0, text
