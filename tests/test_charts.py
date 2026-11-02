from statline_agent.charts import ChartSpec, build_figure

ROWS = [
    {"jaar": "2024", "regio": "Rotterdam", "inwoners": 670000.0},
    {"jaar": "2024", "regio": "Amsterdam", "inwoners": 930000.0},
    {"jaar": "2025", "regio": "Rotterdam", "inwoners": 673000.0},
    {"jaar": "2025", "regio": "Amsterdam", "inwoners": 941000.0},
]


def test_line_chart_has_one_trace_per_color_group() -> None:
    spec = ChartSpec(type="line", x="jaar", y="inwoners", color="regio", title="Bevolking")
    figure = build_figure(spec, ROWS)
    assert len(figure.data) == 2
    assert figure.layout.title.text == "Bevolking"


def test_unit_is_shown_on_the_y_axis() -> None:
    spec = ChartSpec(type="line", x="jaar", y="woz", color="", title="", unit="x 1 000 euro")
    rows = [{"jaar": "2024", "woz": 374.0}, {"jaar": "2025", "woz": 395.0}]
    assert build_figure(spec, rows).layout.yaxis.title.text == "woz (x 1 000 euro)"


def test_unit_is_not_part_of_the_llm_schema() -> None:
    assert "unit" not in ChartSpec.model_json_schema()["properties"]


def test_bar_chart_without_color() -> None:
    spec = ChartSpec(type="bar", x="regio", y="inwoners", color="", title="")
    assert build_figure(spec, ROWS[:2]).data[0].type == "bar"


def test_spec_validation() -> None:
    columns = ["jaar", "regio", "inwoners"]
    assert ChartSpec(type="line", x="jaar", y="inwoners", color="regio", title="").is_valid_for(
        columns
    )
    assert not ChartSpec(type="line", x="jaar", y="x", color="", title="").is_valid_for(columns)
    assert not ChartSpec(type="none", x="", y="", color="", title="").is_valid_for(columns)
