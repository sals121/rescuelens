from fastapi.testclient import TestClient

from rescuelens import __version__
from rescuelens.api.app import create_app


def test_health() -> None:
    client = TestClient(create_app())
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}
