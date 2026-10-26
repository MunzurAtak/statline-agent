import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from statline_agent.cbs import CBSClient

FIXTURES = Path(__file__).parent / "fixtures" / "cbs"

# Request path (after /odata/v1/CBS/) -> recorded response file
ROUTES = {
    "Datasets": "catalog.json",
    "70072ned/Properties": "70072ned_properties.json",
    "70072ned/Dimensions": "70072ned_dimensions.json",
    "70072ned/MeasureCodes": "70072ned_measure_codes.json",
    "70072ned/RegioSCodes": "70072ned_regios_codes.json",
    "70072ned/PeriodenCodes": "70072ned_perioden_codes.json",
    "70072ned/Observations": "70072ned_observations.json",
}


def load_fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class RecordingHandler:
    """Serves recorded CBS responses and remembers which requests were made."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path.removeprefix("/odata/v1/CBS/")
        if path not in ROUTES:
            return httpx.Response(404, json={"error": f"no fixture for {path}"})
        return httpx.Response(200, json=load_fixture(ROUTES[path]))


@pytest.fixture
def cbs_handler() -> RecordingHandler:
    return RecordingHandler()


@pytest.fixture
def cbs_client(cbs_handler: RecordingHandler) -> Iterator[CBSClient]:
    http = httpx.Client(
        base_url="https://datasets.cbs.nl/odata/v1/CBS",
        transport=httpx.MockTransport(cbs_handler),
    )
    with CBSClient(http=http) as client:
        yield client
