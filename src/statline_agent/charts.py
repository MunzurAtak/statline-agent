"""Turn the agent's declarative chart spec into a Plotly figure.

The LLM only describes the chart (type and which columns); rendering is plain code. This is
safer than letting the model write plotting code, and the same spec works in the CLI and UI.
"""

from typing import Any, Literal

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pydantic import BaseModel, Field


class ChartSpec(BaseModel):
    type: Literal["line", "bar", "none"] = Field(
        description="line: ontwikkeling in de tijd; bar: vergelijking; none: één getal"
    )
    x: str = Field(description="Kolom uit het resultaat voor de x-as")
    y: str = Field(description="Kolom uit het resultaat met de getallen")
    color: str = Field(description="Kolom om lijnen/staven op te splitsen, of lege tekst")
    title: str = Field(description="Korte titel van de grafiek")

    def is_valid_for(self, columns: list[str]) -> bool:
        if self.type == "none":
            return False
        needed = [self.x, self.y] + ([self.color] if self.color else [])
        return all(column in columns for column in needed)


def build_figure(spec: ChartSpec, rows: list[dict[str, Any]]) -> go.Figure:
    df = pd.DataFrame(rows)
    plot = px.line if spec.type == "line" else px.bar
    figure: go.Figure = plot(
        df,
        x=spec.x,
        y=spec.y,
        color=spec.color or None,
        title=spec.title,
        **({"markers": True} if spec.type == "line" else {"barmode": "group"}),
    )
    figure.update_layout(template="plotly_white", legend_title_text="")
    return figure
