"""Fetch extra metadata per table (summary, measures, dimensions) to make tables findable.

Users ask about "inwoners" or "personeel"; those words are usually in the measure titles
of a table, not in its title. Fetching them takes a few thousand API calls, so the result
is cached on disk.
"""

import json
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from statline_agent.cbs import CBSClient, CBSError

MAX_MEASURES = 40


@dataclass(frozen=True)
class TableMetadata:
    table_id: str
    summary: str = ""
    measures: list[str] = field(default_factory=list)
    dimensions: list[str] = field(default_factory=list)


def fetch_table_metadata(client: CBSClient, table_id: str) -> TableMetadata:
    properties = client.get_properties(table_id)
    measures = list(dict.fromkeys(m.title.strip() for m in client.get_measures(table_id)))
    dimensions = [d.title for d in client.get_dimensions(table_id)]
    return TableMetadata(
        table_id=table_id,
        summary=properties.summary.strip(),
        measures=measures[:MAX_MEASURES],
        dimensions=dimensions,
    )


def fetch_all_metadata(
    client: CBSClient,
    table_ids: Iterable[str],
    cache_path: Path,
    max_workers: int = 6,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, TableMetadata]:
    """Fetch metadata for all tables, reusing (and extending) the on-disk cache.

    Tables that fail (e.g. a temporary API error) are skipped; the next run retries them.
    """
    cached: dict[str, TableMetadata] = {}
    if cache_path.exists():
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        cached = {item["table_id"]: TableMetadata(**item) for item in raw}

    todo = [table_id for table_id in table_ids if table_id not in cached]

    def fetch(table_id: str) -> TableMetadata | None:
        try:
            return fetch_table_metadata(client, table_id)
        except CBSError:
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for done, result in enumerate(pool.map(fetch, todo), start=1):
            if result is not None:
                cached[result.table_id] = result
            if on_progress:
                on_progress(done, len(todo))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(meta) for meta in cached.values()]
    cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return cached
