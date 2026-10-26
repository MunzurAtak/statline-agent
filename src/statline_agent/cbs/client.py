"""Small, synchronous client for the CBS OData v4 API."""

from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Self

import httpx

from statline_agent.cbs.models import (
    DatasetInfo,
    DatasetProperties,
    Dimension,
    DimensionCode,
    Measure,
)

CBS_ODATA_URL = "https://datasets.cbs.nl/odata/v1/CBS"
DEFAULT_MAX_ROWS = 50_000


class CBSError(RuntimeError):
    """Raised when the CBS API returns an error."""


class QueryTooLargeError(CBSError):
    """Raised when a query returns more rows than allowed; the caller should filter more."""


def build_filter(filters: Mapping[str, Sequence[str]]) -> str:
    """Build an OData $filter, e.g. {"RegioS": ["GM0599", "GM0363"]} ->
    "(RegioS eq 'GM0599' or RegioS eq 'GM0363')".
    """
    clauses = []
    for field, values in filters.items():
        if not values:
            continue
        escaped = [v.replace("'", "''") for v in values]
        clauses.append("(" + " or ".join(f"{field} eq '{v}'" for v in escaped) + ")")
    return " and ".join(clauses)


class CBSClient:
    """Fetches catalog, metadata and observations from CBS StatLine."""

    def __init__(self, base_url: str = CBS_ODATA_URL, http: httpx.Client | None = None) -> None:
        self._http = http or httpx.Client(base_url=base_url, timeout=60, follow_redirects=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    # -- low level ---------------------------------------------------------

    def _get(self, url: str, params: Mapping[str, str | int] | None = None) -> dict[str, Any]:
        try:
            response = self._http.get(url, params=params)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise CBSError(f"CBS request failed: {url} ({exc})") from exc
        data: dict[str, Any] = response.json()
        return data

    def _iter_pages(
        self, url: str, params: Mapping[str, str | int] | None = None
    ) -> Iterator[dict[str, Any]]:
        """Yield all rows, following @odata.nextLink pagination."""
        next_url: str | None = url
        next_params = params
        while next_url:
            page = self._get(next_url, next_params)
            yield from page.get("value", [])
            next_url = page.get("@odata.nextLink")
            next_params = None  # the nextLink already contains the query

    # -- catalog & metadata --------------------------------------------------

    def list_datasets(self) -> list[DatasetInfo]:
        return [DatasetInfo.model_validate(row) for row in self._iter_pages("Datasets")]

    def get_properties(self, table_id: str) -> DatasetProperties:
        return DatasetProperties.model_validate(self._get(f"{table_id}/Properties"))

    def get_dimensions(self, table_id: str) -> list[Dimension]:
        return [Dimension.model_validate(r) for r in self._iter_pages(f"{table_id}/Dimensions")]

    def get_measures(self, table_id: str) -> list[Measure]:
        return [Measure.model_validate(r) for r in self._iter_pages(f"{table_id}/MeasureCodes")]

    def get_dimension_codes(self, table_id: str, dimension_id: str) -> list[DimensionCode]:
        rows = self._iter_pages(f"{table_id}/{dimension_id}Codes")
        return [DimensionCode.model_validate(r) for r in rows]

    # -- data ----------------------------------------------------------------

    def get_observations(
        self,
        table_id: str,
        filters: Mapping[str, Sequence[str]] | None = None,
        max_rows: int = DEFAULT_MAX_ROWS,
    ) -> list[dict[str, Any]]:
        """Fetch observations in long format (one row per measure x dimension combination).

        `filters` maps a dimension (or "Measure") to the codes to keep, e.g.
        {"Measure": ["M000352_3"], "RegioS": ["GM0599"]}.
        """
        params: dict[str, str | int] = {}
        odata_filter = build_filter(filters or {})
        if odata_filter:
            params["$filter"] = odata_filter

        rows: list[dict[str, Any]] = []
        for row in self._iter_pages(f"{table_id}/Observations", params):
            rows.append(row)
            if len(rows) > max_rows:
                raise QueryTooLargeError(
                    f"Query on {table_id} returns more than {max_rows} rows; add filters."
                )
        return rows
