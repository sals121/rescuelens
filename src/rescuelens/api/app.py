"""FastAPI application: serves analysis runs and their images."""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, ValidationError

from rescuelens import __version__
from rescuelens.config import Settings, get_settings
from rescuelens.logging import configure_logging, get_logger
from rescuelens.schemas import Run, RunStats

log = get_logger(__name__)

_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_ASSET_DIRS = ("frames", "crops")


class Health(BaseModel):
    status: str
    version: str


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


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, json=settings.log_json)

    app = FastAPI(title="RescueLens", version=__version__)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5179", "http://127.0.0.1:5179"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> Health:
        return Health(status="ok", version=__version__)

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
