"""Data contract for analysis runs, persisted as JSON and served by the API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from rescuelens.verify.schema import Verification

GeoSource = Literal["exif", "simulated"]
VerifyStatus = Literal["pending", "verified", "cached", "error", "skipped"]


class Frame(BaseModel):
    id: str
    filename: str
    width: int
    height: int
    lat: float
    lon: float
    alt_m: float | None = None
    geo_source: GeoSource
    preview: str  # path relative to the run directory
    n_candidates: int = 0
    detect_s: float = 0.0
    # annotated people (x, y, w, h), when the folder ships an annotations.json (HERIDAL samples)
    ground_truth: list[tuple[float, float, float, float]] | None = None


class Candidate(BaseModel):
    id: str
    frame_id: str
    bbox: tuple[float, float, float, float]  # x, y, w, h in original-frame pixels
    yolo_score: float
    lat: float
    lon: float
    geo_source: GeoSource
    crop: str  # path relative to the run directory
    verify_status: VerifyStatus = "pending"
    verification: Verification | None = None
    verify_error: str | None = None
    llm_cost_usd: float = 0.0
    llm_latency_s: float = 0.0


class RunStats(BaseModel):
    n_frames: int = 0
    n_candidates: int = 0
    n_verified: int = 0
    n_confirmed: int = 0  # LLM says is_person
    n_high_priority: int = 0
    n_errors: int = 0
    duration_s: float = 0.0
    detect_s: float = 0.0
    cost_usd: float = 0.0  # API list-price equivalent (not billed on the claude-code backend)
    cost_full_run_usd: float = (
        0.0  # same run with an empty cache (cached verdicts at their original price)
    )
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    cache_hits: int = 0


class RunConfig(BaseModel):
    source_dir: str  # folder name only, so run files never leak local paths
    weights: str
    use_sahi: bool
    confidence: float
    llm_enabled: bool
    model: str | None
    llm_backend: str | None = None  # "claude-code" (subscription) or "api"
    prompt_version: str | None
    geo_mode: GeoSource


class Run(BaseModel):
    run_id: str
    created_at: str
    status: Literal["running", "done", "failed"] = "running"
    config: RunConfig
    stats: RunStats = Field(default_factory=RunStats)
    frames: list[Frame] = Field(default_factory=list)
    candidates: list[Candidate] = Field(default_factory=list)
    report_markdown: str | None = None
    error: str | None = None
