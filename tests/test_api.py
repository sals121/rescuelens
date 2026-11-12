from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rescuelens import __version__
from rescuelens.api.app import create_app
from rescuelens.config import Settings
from rescuelens.schemas import Candidate, Frame, Run, RunConfig
from rescuelens.verify.schema import Verification


@pytest.fixture
def runs_dir(tmp_path: Path) -> Path:
    run = Run(
        run_id="20261112-120000-demo",
        created_at="2026-11-12T12:00:00+00:00",
        status="done",
        config=RunConfig(
            source_dir="x",
            weights="yolo26s.pt",
            use_sahi=True,
            confidence=0.25,
            llm_enabled=True,
            model="claude-opus-5-5",
            prompt_version="v1",
            geo_mode="simulated",
        ),
        frames=[
            Frame(
                id="F001",
                filename="a.jpg",
                width=4000,
                height=3000,
                lat=42.6,
                lon=-0.04,
                geo_source="simulated",
                preview="frames/F001.jpg",
            )
        ],
        candidates=[
            Candidate(
                id="F001-C01",
                frame_id="F001",
                bbox=(10, 20, 30, 60),
                yolo_score=0.8,
                lat=42.6,
                lon=-0.04,
                geo_source="simulated",
                crop="crops/F001-C01.jpg",
                verify_status="verified",
                verification=Verification(
                    is_person=True,
                    confidence=0.9,
                    evidence="Red jacket, head and shadow visible.",
                    posture="lying",
                    priority="high",
                ),
            )
        ],
    )
    d = tmp_path / run.run_id
    (d / "crops").mkdir(parents=True)
    (d / "crops" / "F001-C01.jpg").write_bytes(b"\xff\xd8fakejpeg")
    (d / "run.json").write_text(run.model_dump_json(), encoding="utf-8")
    (tmp_path / ".cache").mkdir()  # must be ignored
    return tmp_path


@pytest.fixture
def client(runs_dir: Path) -> TestClient:
    settings = Settings(_env_file=None, RESCUELENS_RUNS_DIR=runs_dir)
    return TestClient(create_app(settings))


def test_health(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


def test_list_runs(client: TestClient) -> None:
    runs = client.get("/api/runs").json()
    assert [r["run_id"] for r in runs] == ["20261112-120000-demo"]
    assert runs[0]["model"] == "claude-opus-5-5"


def test_get_run(client: TestClient) -> None:
    run = client.get("/api/runs/20261112-120000-demo").json()
    assert run["candidates"][0]["verification"]["priority"] == "high"


def test_get_asset(client: TestClient) -> None:
    r = client.get("/api/runs/20261112-120000-demo/crops/F001-C01.jpg")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"


@pytest.mark.parametrize(
    "url",
    [
        "/api/runs/missing",
        "/api/runs/.cache",
        "/api/runs/20261112-120000-demo/run.json/x",
        "/api/runs/20261112-120000-demo/crops/..%2Frun.json",
        "/api/runs/20261112-120000-demo/crops/nope.jpg",
    ],
)
def test_not_found_and_traversal(client: TestClient, url: str) -> None:
    assert client.get(url).status_code == 404
