"""FastAPI application: serves analysis runs and their images."""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ValidationError
from sse_starlette.sse import EventSourceResponse

from rescuelens import __version__
from rescuelens.api.jobs import BusyError, JobManager, Runner, default_runner
from rescuelens.config import Settings, get_settings
from rescuelens.ingest.loader import list_images
from rescuelens.logging import configure_logging, get_logger
from rescuelens.pipeline import AnalysisOptions, new_run_id
from rescuelens.schemas import Run, RunStats

log = get_logger(__name__)

_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_ASSET_DIRS = ("frames", "crops")
# Local tool: only answer to local host names (blocks DNS rebinding) and local web origins.
ALLOWED_HOSTS = ["localhost", "127.0.0.1", "testserver"]
ALLOWED_ORIGINS = ["http://localhost:5179", "http://127.0.0.1:5179"]


class Health(BaseModel):
    status: str
    version: str


class Source(BaseModel):
    name: str
    n_images: int


class AnalyzeRequest(BaseModel):
    source: str
    limit: int | None = Field(default=None, ge=1, le=500)
    confidence: float = Field(default=0.4, gt=0, lt=1)
    llm: bool = True
    geo: Literal["auto", "exif", "simulated"] = "auto"


class AnalyzeResponse(BaseModel):
    run_id: str


class Status(BaseModel):
    busy: bool
    active_run_id: str | None
    weights: str


def list_sources(settings: Settings) -> dict[str, Path]:
    """Image folders the UI may analyse: data/samples/* plus the HERIDAL test frames.

    The API never takes a filesystem path from the client, only one of these names.
    """
    found: dict[str, Path] = {}
    if settings.samples_dir.is_dir():
        for d in sorted(settings.samples_dir.iterdir()):
            if d.is_dir() and _SAFE_ID.fullmatch(d.name):
                found[d.name] = d
    heridal = settings.data_dir / "heridal" / "test_full" / "images"
    if heridal.is_dir():
        found["heridal-test"] = heridal
    return {k: v for k, v in found.items() if list_images(v, 1)}


class RunSummary(BaseModel):
    run_id: str
    created_at: str
    status: str
    stats: RunStats
    model: str | None
    geo_mode: str


def _run_dir(settings: Settings, run_id: str) -> Path:
    if not _SAFE_ID.fullmatch(run_id) or run_id.startswith("."):
        raise HTTPException(status_code=404, detail="run not found")
    path = settings.runs_dir / run_id
    if not (path / "run.json").is_file():
        raise HTTPException(status_code=404, detail="run not found")
    return path


def _load_run(path: Path) -> Run:
    try:
        return Run.model_validate_json((path / "run.json").read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        log.error("run_unreadable", path=str(path), error=str(exc))
        raise HTTPException(status_code=500, detail="run file is unreadable") from exc


def create_app(settings: Settings | None = None, runner: Runner | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, json=settings.log_json)
    cfg = settings
    jobs = JobManager(runner or default_runner(lambda: cfg))

    app = FastAPI(title="RescueLens", version=__version__)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> Health:
        return Health(status="ok", version=__version__)

    @app.get("/api/status")
    def status() -> Status:
        return Status(
            busy=jobs.busy, active_run_id=jobs.active_run_id, weights=settings.default_weights.name
        )

    @app.get("/api/sources")
    def sources() -> list[Source]:
        found = list_sources(settings)
        return [Source(name=k, n_images=len(list_images(v))) for k, v in found.items()]

    @app.post("/api/analyze", status_code=202)
    async def analyze(req: AnalyzeRequest, request: Request) -> AnalyzeResponse:
        from rescuelens.detect.detector import auto_device

        # starting a job spends GPU time and Claude budget: refuse other web pages
        origin = request.headers.get("origin")
        if origin is not None and origin not in ALLOWED_ORIGINS:
            raise HTTPException(status_code=403, detail="origin not allowed")

        source = list_sources(settings).get(req.source)
        if source is None:
            raise HTTPException(status_code=404, detail=f"unknown source {req.source!r}")
        opts = AnalysisOptions(
            source_dir=source,
            weights=settings.default_weights,
            device=auto_device(),
            confidence=req.confidence,
            llm_enabled=req.llm,
            geo_mode=req.geo,
            limit=req.limit,
            run_id=new_run_id(source),
        )
        try:
            job = jobs.start(opts)
        except BusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return AnalyzeResponse(run_id=job.run_id)

    @app.get("/api/runs/{run_id}/events")
    async def events(run_id: str) -> EventSourceResponse:
        job = jobs.get(run_id)
        if job is None:
            raise HTTPException(status_code=404, detail="no live analysis with that id")

        async def stream() -> AsyncIterator[dict[str, str]]:
            async for ev in job.follow():
                yield {"event": ev["type"], "data": json.dumps(ev["data"])}

        return EventSourceResponse(stream(), ping=15)

    @app.get("/api/runs")
    def list_runs() -> list[RunSummary]:
        runs: list[RunSummary] = []
        if not settings.runs_dir.is_dir():
            return runs
        for d in settings.runs_dir.iterdir():
            if d.name.startswith(".") or not (d / "run.json").is_file():
                continue
            try:
                run = _load_run(d)
            except HTTPException:
                continue
            runs.append(
                RunSummary(
                    run_id=run.run_id,
                    created_at=run.created_at,
                    status=run.status,
                    stats=run.stats,
                    model=run.config.model,
                    geo_mode=run.config.geo_mode,
                )
            )
        return sorted(runs, key=lambda r: r.created_at, reverse=True)

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> Run:
        return _load_run(_run_dir(settings, run_id))

    @app.get("/api/runs/{run_id}/{kind}/{filename}")
    def get_asset(run_id: str, kind: str, filename: str) -> FileResponse:
        if kind not in _ASSET_DIRS or not _SAFE_ID.fullmatch(filename):
            raise HTTPException(status_code=404, detail="asset not found")
        path = _run_dir(settings, run_id) / kind / filename
        if not path.is_file():
            raise HTTPException(status_code=404, detail="asset not found")
        return FileResponse(path, media_type="image/jpeg")

    return app


app = create_app()
