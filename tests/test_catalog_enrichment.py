from pathlib import Path

from statline_agent.catalog import TableMetadata, dataset_to_document, fetch_all_metadata
from statline_agent.cbs import CBSClient, DatasetInfo
from tests.conftest import RecordingHandler


def test_fetch_all_metadata_collects_summary_measures_and_dimensions(
    cbs_client: CBSClient, tmp_path: Path
) -> None:
    result = fetch_all_metadata(cbs_client, ["70072ned"], tmp_path / "meta.json")
    meta = result["70072ned"]
    assert meta.summary.startswith("Regionale kerncijfers")
    assert "Totale bevolking" in meta.measures
    assert meta.dimensions == ["Regio's", "Perioden"]


def test_failed_tables_are_skipped(cbs_client: CBSClient, tmp_path: Path) -> None:
    result = fetch_all_metadata(cbs_client, ["70072ned", "unknown"], tmp_path / "meta.json")
    assert set(result) == {"70072ned"}


def test_cache_prevents_refetching(
    cbs_client: CBSClient, cbs_handler: RecordingHandler, tmp_path: Path
) -> None:
    cache = tmp_path / "meta.json"
    fetch_all_metadata(cbs_client, ["70072ned"], cache)
    calls = len(cbs_handler.requests)
    result = fetch_all_metadata(cbs_client, ["70072ned"], cache)
    assert len(cbs_handler.requests) == calls
    assert "70072ned" in result


def test_document_includes_measures_before_description() -> None:
    dataset = DatasetInfo(identifier="T1", title="Kerncijfers", description="Lang verhaal")
    meta = TableMetadata("T1", "Kerncijfers per regio", ["Totale bevolking"], ["Regio's"])
    text = dataset_to_document(dataset, meta).page_content
    assert text.index("Onderwerpen: Totale bevolking") < text.index("Lang verhaal")
    assert "Indeling: Regio's" in text
