import pytest

from statline_agent.cbs import normalize_unit


@pytest.mark.parametrize(
    ("unit", "expected"),
    [
        ("x 1 000 euro", (1_000.0, "euro")),
        ("1 000 euro", (1_000.0, "euro")),  # the WOZ value in 85036NED, without "x"
        ("10 000 euro", (1.0, "10 000 euro")),
        ("x 1 000", (1_000.0, "aantal")),
        ("x 1000 personen", (1_000.0, "personen")),
        ("x 1.000 euro", (1_000.0, "euro")),
        ("mln euro", (1_000_000.0, "euro")),
        ("x mln euro", (1_000_000.0, "euro")),
        ("x 1 000 000 euro", (1_000_000.0, "euro")),
        ("mld euro", (1_000_000_000.0, "euro")),
        ("aantal", (1.0, "aantal")),
        ("%", (1.0, "%")),
        ("per 1 000 inwoners", (1.0, "per 1 000 inwoners")),  # a rate, not a multiplier
        ("", (1.0, "")),
    ],
)
def test_normalize_unit(unit: str, expected: tuple[float, str]) -> None:
    assert normalize_unit(unit) == expected
