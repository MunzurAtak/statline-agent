from collections.abc import Iterator

import pytest

from statline_agent.cbs import CBSClient, DataStore, NoDataError, cache_table_name
from tests.conftest import RecordingHandler


@pytest.fixture
def store(cbs_client: CBSClient) -> Iterator[DataStore]:
    data_store = DataStore(cbs_client, path=":memory:")
    yield data_store
    data_store.close()


def test_cache_table_name_is_stable_and_order_independent() -> None:
    a = cache_table_name("70072ned", {"RegioS": ["GM0599", "GM0363"], "Measure": ["M1"]})
    b = cache_table_name("70072ned", {"Measure": ["M1"], "RegioS": ["GM0363", "GM0599"]})
    assert a == b
    assert a.startswith("obs_70072ned_")
    assert a != cache_table_name("70072ned", {"RegioS": ["GM0599"]})


def test_load_adds_readable_labels(store: DataStore) -> None:
    name = store.load("70072ned", {"RegioS": ["GM0599"]})
    df = store.query(f'SELECT * FROM "{name}"')

    total = df[df["Measure"] == "M000352_3"].iloc[0]
    assert total["Measure_title"] == "Totale bevolking"
    assert total["Unit"] == "aantal"
    assert total["Perioden_title"] == "1995"
    assert total["Value"] == 598239.0
    # GM0599 is not in the (truncated) RegioS fixture, so the code is used as fallback
    assert total["RegioS_title"] == "GM0599"


def test_second_load_uses_cache(store: DataStore, cbs_handler: RecordingHandler) -> None:
    store.load("70072ned", {"RegioS": ["GM0599"]})
    calls_after_first_load = len(cbs_handler.requests)
    store.load("70072ned", {"RegioS": ["GM0599"]})
    assert len(cbs_handler.requests) == calls_after_first_load


def test_empty_result_raises(cbs_client: CBSClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cbs_client, "get_observations", lambda *args, **kwargs: [])
    store = DataStore(cbs_client, path=":memory:")
    with pytest.raises(NoDataError):
        store.load("70072ned", {"RegioS": ["XX0000"]})
    store.close()