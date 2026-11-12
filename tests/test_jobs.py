import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from rescuelens.api.app import create_app
from rescuelens.api.jobs import BusyError, JobManager
from rescuelens.config import Settings
from rescuelens.pipeline import AnalysisOptions, Event


def _opts(run_id: str) -> AnalysisOptions:
    return AnalysisOptions(source_dir=Path("x"), weights=Path("w.pt"), run_id=run_id)


async def _fake_run(opts: AnalysisOptions) -> AsyncIterator[Event]:
    yield Event(type="run_started", data={"run_id": opts.run_id})
    await asyncio.sleep(0.01)
    yield Event(type="run_finished", data={"run_id": opts.run_id})


async def _failing_run(opts: AnalysisOptions) -> AsyncIterator[Event]:
    yield Event(type="run_started", data={"run_id": opts.run_id})
    raise RuntimeError("GPU on fire")


async def test_late_subscriber_gets_every_event() -> None:
    jobs = JobManager(_fake_run)
    job = jobs.start(_opts("r1"))
    assert jobs.busy and jobs.active_run_id == "r1"
    with pytest.raises(BusyError):
        jobs.start(_opts("r2"))
    first = [e["type"] async for e in job.follow()]
    late = [e["type"] async for e in job.follow()]  # job already finished
    assert first == late == ["run_started", "run_finished"]
    await asyncio.sleep(0)
    assert not jobs.busy and jobs.active_run_id is None


async def test_failure_becomes_error_event() -> None:
    job = JobManager(_failing_run).start(_opts("r1"))
    events = [e async for e in job.follow()]
    assert [e["type"] for e in events] == ["run_started", "error"]
    assert "GPU on fire" in events[-1]["data"]["message"]


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    folder = tmp_path / "data" / "samples" / "flight-a"
    folder.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(folder / "a.jpg")
    (tmp_path / "data" / "samples" / "empty").mkdir()
    settings = Settings(
        _env_file=None,
        RESCUELENS_RUNS_DIR=tmp_path / "runs",
        RESCUELENS_DATA_DIR=tmp_path / "data",
    )
    return TestClient(create_app(settings, runner=_fake_run))


def test_sources_lists_only_folders_with_images(client: TestClient) -> None:
    assert client.get("/api/sources").json() == [{"name": "flight-a", "n_images": 1}]


def test_analyze_streams_events(client: TestClient) -> None:
    r = client.post("/api/analyze", json={"source": "flight-a", "llm": False})
    assert r.status_code == 202
    run_id = r.json()["run_id"]
    assert run_id.endswith("-flight-a")
    with client.stream("GET", f"/api/runs/{run_id}/events") as stream:
        body = "".join(stream.iter_text())
    types = [ln.split(":", 1)[1].strip() for ln in body.splitlines() if ln.startswith("event:")]
    assert types == ["run_started", "run_finished"]
    data = [json.loads(ln.split(":", 1)[1]) for ln in body.splitlines() if ln.startswith("data:")]
    assert data[0]["run_id"] == run_id


def test_analyze_rejects_unknown_source_and_paths(client: TestClient) -> None:
    for source in ("nope", "../runs", "C:/Windows"):
        assert client.post("/api/analyze", json={"source": source}).status_code == 404


def test_events_for_unknown_run(client: TestClient) -> None:
    assert client.get("/api/runs/whatever/events").status_code == 404


def test_analyze_refuses_foreign_origins_and_hosts(client: TestClient) -> None:
    body = {"source": "flight-a"}
    r = client.post("/api/analyze", json=body, headers={"origin": "https://evil.example"})
    assert r.status_code == 403
    r = client.post("/api/analyze", json=body, headers={"host": "evil.example"})
    assert r.status_code == 400  # DNS rebinding: unknown Host header
