from typing import Any

import httpx
import pytest

from statline_agent.cbs import CBSClient, CBSError, QueryTooLargeError, build_filter
from tests.conftest import RecordingHandler


def test_build_filter_combines_dimensions_with_and_and_codes_with_or() -> None:
    result = build_filter({"RegioS": ["GM0599", "GM0363"], "Measure": ["M000352_3"]})
    assert result == "(RegioS eq 'GM0599' or RegioS eq 'GM0363') and (Measure eq 'M000352_3')"


def test_build_filter_escapes_quotes_and_skips_empty() -> None:
    assert build_filter({"RegioS": ["'s-Hertogenbosch"], "Perioden": []}) == (
        "(RegioS eq '''s-Hertogenbosch')"
    )


def test_list_datasets(cbs_client: CBSClient) -> None:
    datasets = cbs_client.list_datasets()
    assert [d.identifier for d in datasets] == ["00370", "00371edu", "00372"]
    assert not datasets[0].has_odata  # CSV only
    assert datasets[2].has_odata
    assert not datasets[2].is_active


def test_get_properties(cbs_client: CBSClient) -> None:
    props = cbs_client.get_properties("70072ned")
    assert props.title == "Regionale kerncijfers Nederland"
    assert props.observation_count is not None and props.observation_count > 1_000_000


def test_get_dimensions(cbs_client: CBSClient) -> None:
    dims = cbs_client.get_dimensions("70072ned")
    assert [(d.identifier, d.kind) for d in dims] == [
        ("RegioS", "GeoDimension"),
        ("Perioden", "TimeDimension"),
    ]


def test_get_measures(cbs_client: CBSClient) -> None:
    measures = {m.identifier: m for m in cbs_client.get_measures("70072ned")}
    assert measures["M000352_3"].title == "Totale bevolking"
    assert measures["M000352_3"].unit == "aantal"


def test_get_dimension_codes(cbs_client: CBSClient) -> None:
    codes = cbs_client.get_dimension_codes("70072ned", "RegioS")
    assert codes[0].identifier == "NL01"
    assert codes[0].title == "Nederland"


def test_get_observations_sends_filter(
    cbs_client: CBSClient, cbs_handler: RecordingHandler
) -> None:
    rows = cbs_client.get_observations("70072ned", filters={"RegioS": ["GM0599"]})
    assert rows[0]["RegioS"] == "GM0599"
    assert rows[0]["Measure"] == "M000352_3"
    sent = cbs_handler.requests[-1].url.params["$filter"]
    assert sent == "(RegioS eq 'GM0599')"


def test_get_observations_raises_when_too_large(cbs_client: CBSClient) -> None:
    with pytest.raises(QueryTooLargeError):
        cbs_client.get_observations("70072ned", max_rows=5)


def test_pagination_follows_next_link() -> None:
    pages: dict[str, dict[str, Any]] = {
        "/odata/v1/CBS/T1/Observations": {
            "value": [{"Id": 0}],
            "@odata.nextLink": "https://datasets.cbs.nl/odata/v1/CBS/T1/Observations?$skip=1",
        },
        "/odata/v1/CBS/T1/Observations?$skip=1": {"value": [{"Id": 1}]},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        key = request.url.raw_path.decode().replace("%24", "$")
        return httpx.Response(200, json=pages[key])

    http = httpx.Client(
        base_url="https://datasets.cbs.nl/odata/v1/CBS", transport=httpx.MockTransport(handler)
    )
    with CBSClient(http=http) as client:
        assert [r["Id"] for r in client.get_observations("T1")] == [0, 1]


def test_http_errors_become_cbs_errors(cbs_client: CBSClient) -> None:
    with pytest.raises(CBSError):
        cbs_client.get_properties("does-not-exist")
