"""Local DuckDB cache for CBS observations, enriched with human-readable labels."""

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from statline_agent.cbs.client import CBSClient, CBSError

DEFAULT_DB_PATH = Path(".cache/statline.duckdb")
QUERY_VIEW = "data"  # generated SQL always reads from this view
MAX_DISTINCT_SHOWN = 12


class NoDataError(CBSError):
    """Raised when a query matches no observations (e.g. wrong codes in the filter)."""


class QueryError(ValueError):
    """Raised when generated SQL is not allowed or fails; the message goes back to the LLM."""


def _to_json_value(value: Any) -> Any:
    """DuckDB can return Decimal for some aggregates; keep results JSON-serializable."""
    return float(value) if isinstance(value, Decimal) else value


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
        # No file system or network access from SQL: generated queries can only read the
        # tables in this database (e.g. read_csv('/etc/passwd') fails).
        self._con = duckdb.connect(str(path), config={"enable_external_access": False})

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

    def _rows(self, sql: str) -> list[tuple[Any, ...]]:
        return self._con.execute(sql).fetchall()

    def describe(self, table: str) -> str:
        """Compact overview of a cached table for the LLM: columns and example values."""
        count = self._rows(f'SELECT count(*) FROM "{table}"')[0][0]
        columns = [row[0] for row in self._rows(f'DESCRIBE "{table}"')]
        lines = [f"Tabel {QUERY_VIEW} ({count} rijen). Kolommen:"]
        for column in columns:
            if column.endswith("_title") or column == "Unit":
                limit = MAX_DISTINCT_SHOWN + 1
                sql = f'SELECT DISTINCT "{column}" FROM "{table}" ORDER BY 1 LIMIT {limit}'
                values = [str(row[0]) for row in self._rows(sql)]
                more = " ..." if len(values) > MAX_DISTINCT_SHOWN else ""
                lines.append(f"- {column}: {'; '.join(values[:MAX_DISTINCT_SHOWN])}{more}")
            elif column == "Value":
                low, high = self._rows(f'SELECT min("Value"), max("Value") FROM "{table}"')[0]
                lines.append(f"- Value: getal (min {low}, max {high})")
            else:
                lines.append(f"- {column}")
        return "\n".join(lines)

    def run_select(
        self, sql: str, source_table: str, max_rows: int = 200
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Run one read-only SELECT against `data` (a view on source_table).

        Returns (columns, rows). Raises QueryError with a message the LLM can act on.
        """
        sql = sql.strip().rstrip(";")
        try:
            statements = self._con.extract_statements(sql)
        except duckdb.Error as exc:
            raise QueryError(f"SQL kan niet worden gelezen: {exc}") from exc
        if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
            raise QueryError("Alleen precies één SELECT-query is toegestaan.")

        # A cursor is a separate connection: the temporary view is private to this query.
        cursor = self._con.cursor()
        try:
            cursor.execute(f'CREATE TEMP VIEW {QUERY_VIEW} AS SELECT * FROM "{source_table}"')
            cursor.execute(sql)
            columns = [d[0] for d in cursor.description or []]
            rows = cursor.fetchmany(max_rows)
        except duckdb.Error as exc:
            raise QueryError(str(exc)) from exc
        finally:
            cursor.close()
        return columns, [dict(zip(columns, map(_to_json_value, row), strict=True)) for row in rows]

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
