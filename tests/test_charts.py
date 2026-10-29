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
