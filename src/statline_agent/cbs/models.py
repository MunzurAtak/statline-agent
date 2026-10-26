"""Typed models for the CBS OData v4 API (https://datasets.cbs.nl/odata/v1/CBS)."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_pascal


class CBSModel(BaseModel):
    """Base model: CBS uses PascalCase field names, we use snake_case in Python."""

    model_config = ConfigDict(alias_generator=to_pascal, populate_by_name=True, extra="ignore")


class Distribution(CBSModel):
    format: str
    download_url: str


class DatasetInfo(CBSModel):
    """One entry in the CBS catalog."""

    identifier: str
    title: str
    description: str = ""
    status: str = ""
    modified: datetime | None = None
    observation_count: int | None = None
    distributions: list[Distribution] = []

    @property
    def has_odata(self) -> bool:
        """Some (mostly old) tables are only available as CSV download."""
        return any(d.format == "odata" for d in self.distributions)

    @property
    def is_active(self) -> bool:
        return self.status != "Gediscontinueerd"


class DatasetProperties(CBSModel):
    """Detailed metadata of a single table."""

    identifier: str
    title: str
    summary: str = ""
    description: str = ""
    temporal_coverage: str = ""
    frequency: str = ""
    status: str = ""
    modified: datetime | None = None
    observation_count: int | None = None


class Dimension(CBSModel):
    identifier: str
    title: str
    kind: str


class DimensionCode(CBSModel):
    """A value within a dimension, e.g. GM0599 (Rotterdam) in RegioS."""

    identifier: str
    title: str
    description: str = ""
    dimension_group_id: str | None = None


class Measure(CBSModel):
    """A measure ('onderwerp'), e.g. M000352_3 = Totale bevolking."""

    identifier: str
    title: str
    description: str = ""
    unit: str = ""
    decimals: int | None = None
    data_type: str = ""
    measure_group_id: str | None = None
