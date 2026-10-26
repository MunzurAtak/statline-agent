"""Local DuckDB cache for CBS observations, enriched with human-readable labels."""

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import duckdb
import pandas as pd

from statline_agent.cbs.client import CBSClient, CBSError

DEFAULT_DB_PATH = Path(".cache/statline.duckdb")


class NoDataError(CBSError):
    """Raised when a query matches no observations (e.g. wrong codes in the filter)."""


def cache_table_name(table_id: str, filters: Mapping[str, Sequence[str]] | None = None) -> str:
    """Deterministic DuckDB table name for a (table, filters) combination."""
    key = json.dumps({k: sorted(v) for k, v in sorted((filters or {}).items())})
    digest = hashlib.sha256(key.encode()).hexdigest()[:10]
    safe_id = re.sub(r"\W", "_", table_id.lower())
    return f"obs_{safe_id}_{digest}"


def _label(codes: "pd.Series[str]", titles: Mapping[str, str]) -> "pd.Series[str]":
    """Map codes to titles; unknown codes keep their code as label."""
    return codes.map(titles).fillna(codes)


class DataStore:
    """Fetches CBS observations once, stores them in DuckDB and serves them via SQL.

    Each cached table has the CBS codes plus readable titles, e.g.
    Measure='M000352_3', Measure_title='Totale bevolking', RegioS='GM0599', ...
    """

    def __init__(self, client: CBSClient, path: str | Path = DEFAULT_DB_PATH) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._client = client
        self._con = duckdb.connect(str(path))

    def close(self) -> None:
        self._con.close()

    def has_table(self, name: str) -> bool:
        rows = self._con.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = ?", [name]
        ).fetchall()
        return len(rows) > 0

    def load(self, table_id: str, filters: Mapping[str, Sequence[str]] | None = None) -> str:
        """Make sure the data is cached and return the DuckDB table name."""
        name = cache_table_name(table_id, filters)
        if not self.has_table(name):
            df = self._fetch_labelled(table_id, filters)
            self._con.register("incoming", df)
            self._con.execute(f'CREATE TABLE "{name}" AS SELECT * FROM incoming')
            self._con.unregister("incoming")
        return name

    def query(self, sql: str) -> pd.DataFrame:
        return self._con.execute(sql).df()

    def _fetch_labelled(
        self, table_id: str, filters: Mapping[str, Sequence[str]] | None
    ) -> pd.DataFrame:
        rows = self._client.get_observations(table_id, filters)
        if not rows:
            raise NoDataError(f"No observations in {table_id} match filters {filters}.")
        df = pd.DataFrame(rows).drop(columns=["Id"], errors="ignore")

        measures = self._client.get_measures(table_id)
        df["Measure_title"] = _label(df["Measure"], {m.identifier: m.title for m in measures})
        df["Unit"] = df["Measure"].map({m.identifier: m.unit for m in measures}).fillna("")

        for dim in self._client.get_dimensions(table_id):
            if dim.identifier not in df.columns:
                continue
            codes = self._client.get_dimension_codes(table_id, dim.identifier)
            titles = {c.identifier: c.title for c in codes}
            df[f"{dim.identifier}_title"] = _label(df[dim.identifier], titles)

        return df