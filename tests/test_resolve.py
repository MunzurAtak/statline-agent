from statline_agent.agent.resolve import default_code, resolve_names, select_periods
from statline_agent.cbs import DimensionCode


def codes(*pairs: tuple[str, str]) -> list[DimensionCode]:
    return [DimensionCode(identifier=i, title=t) for i, t in pairs]


REGIONS = codes(
    ("NL01", "Nederland"),
    ("PV28", "Zuid-Holland (PV)"),
    ("GM0599", "Rotterdam"),
    ("GM0344", "Utrecht"),
    ("PV26", "Utrecht (PV)"),
    ("GM0518", "'s-Gravenhage"),
)


def test_resolve_exact_prefix_fuzzy_and_code() -> None:
    found, missing = resolve_names(["rotterdam", "Zuid-Holland", "Roterdam", "GM0518"], REGIONS)
    assert found == ["GM0599", "PV28", "GM0518"]  # "Roterdam" (typo) -> Rotterdam, deduplicated
    assert missing == []


def test_aliases_for_official_names() -> None:
    assert resolve_names(["Den Haag"], REGIONS)[0] == ["GM0518"]


def test_exact_title_wins_over_prefix() -> None:
    assert resolve_names(["Utrecht"], REGIONS)[0] == ["GM0344"]


def test_unknown_names_are_reported() -> None:
    found, missing = resolve_names(["Atlantis"], REGIONS)
    assert found == []
    assert missing == ["Atlantis"]


def test_default_code_prefers_total() -> None:
    sexes = codes(("3000", "Mannen"), ("T001038", "Totaal mannen en vrouwen"))
    assert default_code(sexes) == "T001038"
    assert default_code(REGIONS) == "NL01"
    assert default_code(codes(("A", "Eerste"), ("B", "Tweede"))) == "A"
    assert default_code([]) is None


PERIODS = codes(
    ("2022JJ00", "2022"),
    ("2023JJ00", "2023"),
    ("2024JJ00", "2024"),
    ("2024KW01", "2024 1e kwartaal"),
    ("2024KW02", "2024 2e kwartaal"),
)


def test_select_periods_by_frequency_and_range() -> None:
    assert select_periods(PERIODS, "jaar", 2023, 0) == ["2023JJ00", "2024JJ00"]
    assert select_periods(PERIODS, "kwartaal") == ["2024KW01", "2024KW02"]
    assert select_periods(PERIODS, "jaar", latest_only=True) == ["2024JJ00"]


def test_select_periods_falls_back_to_available_frequency() -> None:
    yearly = codes(("2023JJ00", "2023"), ("2024JJ00", "2024"))
    assert select_periods(yearly, "maand", latest_only=True) == ["2024JJ00"]
    assert select_periods(yearly, "jaar", 2030, 0) == []
