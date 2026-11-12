"""End-to-end analysis: ingest → detect (YOLO+SAHI) → verify (Claude) → report.

`run_analysis` is an async generator of progress events so the CLI, the API (SSE) and the
demo mode all share one code path. The final `Run` is written to `<runs_dir>/<run_id>/run.json`.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel

from rescuelens.config import Settings
from rescuelens.detect.crops import make_context, make_crop, make_thumbnail
from rescuelens.detect.detector import Detection, DetectorConfig, PersonDetector
from rescuelens.ingest.georef_simulated import pixel_to_latlon
from rescuelens.ingest.loader import GeoMode, SourceFrame, list_images, load_frames
from rescuelens.llm import make_client
from rescuelens.logging import get_logger
from rescuelens.report.generator import llm_report, offline_report
from rescuelens.schemas import Candidate, Frame, Run, RunConfig
from rescuelens.verify.cache import VerdictCache
from rescuelens.verify.client import (
    Verifier,
    VerifyRequest,
    load_prompt,
)
from rescuelens.verify.cost import BudgetExceededError, CostTracker

log = get_logger(__name__)

PREVIEW_MAX_SIDE = 1600


@dataclass(frozen=True, slots=True)
class AnalysisOptions:
    source_dir: Path
    weights: Path
    device: str = "cpu"
    use_sahi: bool = True
    confidence: float = 0.25
    max_candidates_per_frame: int = 15
    llm_enabled: bool = True
    prompt_version: str = "v1"
    geo_mode: GeoMode = "auto"
    limit: int | None = None
    run_id: str | None = None


class Event(BaseModel):
    type: Literal[
        "run_started", "frame_done", "candidate_verified", "report_ready", "run_finished", "error"
    ]
    data: dict[str, Any]


def new_run_id(source_dir: Path) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{source_dir.name}"[:64]


def _candidate_position(frame: SourceFrame, det: Detection) -> tuple[float, float]:
    """Simulated frames know heading + GSD, so we project the pixel. With real EXIF we only know
    the camera position, so the candidate inherits it (error ≈ half the frame footprint)."""
    if frame.pose is not None:
        cx, cy = det.center
        return pixel_to_latlon(frame.pose, cx, cy, frame.width, frame.height)
    return frame.lat, frame.lon


def _save_preview(img: Image.Image, path: Path) -> None:
    scale = min(1.0, PREVIEW_MAX_SIDE / max(img.width, img.height))
    preview = img.resize((round(img.width * scale), round(img.height * scale)))
    path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(path, format="JPEG", quality=85)


def _write_run(run: Run, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp = run_dir / "run.json.tmp"
    tmp.write_text(run.model_dump_json(indent=1), encoding="utf-8")
    tmp.replace(run_dir / "run.json")


async def run_analysis(opts: AnalysisOptions, settings: Settings) -> AsyncIterator[Event]:
    t_start = time.perf_counter()
    run_id = opts.run_id or new_run_id(opts.source_dir)
    run_dir = settings.runs_dir / run_id
    llm_on = opts.llm_enabled and settings.anthropic_api_key is not None
    if opts.llm_enabled and not llm_on:
        log.warning("llm_disabled_no_api_key")

    paths = list_images(opts.source_dir, opts.limit)
    if not paths:
        raise FileNotFoundError(f"no images found in {opts.source_dir}")
    frames_src = load_frames(paths, geo_mode=opts.geo_mode)

    run = Run(
        run_id=run_id,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        config=RunConfig(
            source_dir=str(opts.source_dir),
            weights=opts.weights.name,
            use_sahi=opts.use_sahi,
            confidence=opts.confidence,
            llm_enabled=llm_on,
            model=settings.anthropic_model if llm_on else None,
            prompt_version=opts.prompt_version if llm_on else None,
            geo_mode=frames_src[0].geo_source,
        ),
    )
    run.stats.n_frames = len(frames_src)
    yield Event(
        type="run_started",
        data={"run_id": run_id, "config": run.config.model_dump(), "n_frames": len(frames_src)},
    )

    detector = await asyncio.to_thread(
        PersonDetector,
        DetectorConfig(
            weights=opts.weights,
            confidence=opts.confidence,
            device=opts.device,
            use_sahi=opts.use_sahi,
        ),
    )

    tracker = CostTracker(max_cost_usd=settings.llm_max_cost_usd)
    client = make_client(settings) if llm_on else None
    verifier = (
        Verifier(
            client=client,
            model=settings.anthropic_model,
            system_prompt=load_prompt(settings.prompts_dir, opts.prompt_version),
            prompt_version=opts.prompt_version,
            cache=VerdictCache(settings.runs_dir / ".cache" / "llm"),
            tracker=tracker,
            max_concurrency=settings.llm_max_concurrency,
        )
        if client is not None
        else None
    )

    queue: asyncio.Queue[Event] = asyncio.Queue()
    budget_hit = asyncio.Event()

    async def verify_one(cand: Candidate, req: VerifyRequest) -> None:
        assert verifier is not None
        if budget_hit.is_set():
            cand.verify_status = "skipped"
            cand.verify_error = "LLM budget exhausted"
        else:
            try:
                res = await verifier.verify(req)
            except BudgetExceededError as exc:
                budget_hit.set()
                cand.verify_status, cand.verify_error = "skipped", str(exc)
                log.warning("llm_budget_exhausted", spent=tracker.spent_usd)
            except Exception as exc:  # one bad candidate must not kill the run
                cand.verify_status, cand.verify_error = "error", f"{type(exc).__name__}: {exc}"
                log.error("verify_failed", candidate=cand.id, error=cand.verify_error)
            else:
                cand.verification = res.verification
                cand.verify_status = "cached" if res.cached else "verified"
                cand.llm_cost_usd, cand.llm_latency_s = res.cost_usd, res.latency_s
        await queue.put(
            Event(
                type="candidate_verified",
                data={"candidate": cand.model_dump(), "cost_usd": tracker.spent_usd},
            )
        )

    async def produce() -> None:
        tasks: list[asyncio.Task[None]] = []
        for idx, src in enumerate(frames_src):
            frame_id = f"F{idx + 1:03d}"
            t0 = time.perf_counter()
            with Image.open(src.path) as raw:
                img = raw.convert("RGB")
            dets = await asyncio.to_thread(detector.detect, np.asarray(img))
            dets = dets[: opts.max_candidates_per_frame]
            detect_s = time.perf_counter() - t0

            preview_rel = f"frames/{frame_id}.jpg"
            await asyncio.to_thread(_save_preview, img, run_dir / preview_rel)
            frame = Frame(
                id=frame_id,
                filename=src.path.name,
                width=src.width,
                height=src.height,
                lat=src.lat,
                lon=src.lon,
                alt_m=src.alt_m,
                geo_source=src.geo_source,
                preview=preview_rel,
                n_candidates=len(dets),
                detect_s=round(detect_s, 3),
            )
            run.frames.append(frame)
            run.stats.detect_s += detect_s

            thumb, scale = make_thumbnail(img) if verifier is not None and dets else (img, 1.0)
            new_cands: list[Candidate] = []
            for j, det in enumerate(dets):
                cand_id = f"{frame_id}-C{j + 1:02d}"
                crop_bytes = make_crop(img, det)
                crop_rel = f"crops/{cand_id}.jpg"
                (run_dir / "crops").mkdir(parents=True, exist_ok=True)
                (run_dir / crop_rel).write_bytes(crop_bytes)
                lat, lon = _candidate_position(src, det)
                cand = Candidate(
                    id=cand_id,
                    frame_id=frame_id,
                    bbox=(round(det.x, 1), round(det.y, 1), round(det.w, 1), round(det.h, 1)),
                    yolo_score=round(det.score, 4),
                    lat=lat,
                    lon=lon,
                    geo_source=src.geo_source,
                    crop=crop_rel,
                    verify_status="pending" if verifier else "skipped",
                )
                run.candidates.append(cand)
                new_cands.append(cand)
                if verifier is not None:
                    ctx = make_context(thumb, scale, det, src.width, src.height)
                    req = VerifyRequest(crop_jpeg=crop_bytes, context_jpeg=ctx)
                    tasks.append(asyncio.create_task(verify_one(cand, req)))

            await queue.put(
                Event(
                    type="frame_done",
                    data={
                        "frame": frame.model_dump(),
                        "candidates": [c.model_dump() for c in new_cands],
                        "progress": (idx + 1) / len(frames_src),
                    },
                )
            )
        if tasks:
            await asyncio.gather(*tasks)

    producer = asyncio.create_task(produce())
    try:
        while not (producer.done() and queue.empty()):
            get = asyncio.create_task(queue.get())
            done, _ = await asyncio.wait({get, producer}, return_when=asyncio.FIRST_COMPLETED)
            if get in done:
                yield get.result()
            else:
                get.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await get
        producer.result()  # re-raise detection errors
    except BaseException as exc:
        producer.cancel()
        run.status, run.error = "failed", f"{type(exc).__name__}: {exc}"
        _write_run(run, run_dir)
        raise

    # Stats + report
    s = run.stats
    s.n_candidates = len(run.candidates)
    s.n_verified = sum(c.verification is not None for c in run.candidates)
    s.n_confirmed = sum(bool(c.verification and c.verification.is_person) for c in run.candidates)
    s.n_high_priority = sum(
        bool(c.verification and c.verification.is_person and c.verification.priority == "high")
        for c in run.candidates
    )
    s.n_errors = sum(c.verify_status == "error" for c in run.candidates)
    s.detect_s = round(s.detect_s, 2)

    if client is not None and s.n_verified and not budget_hit.is_set():
        try:
            run.report_markdown = await llm_report(
                run,
                client=client,
                model=settings.anthropic_model,
                prompts_dir=settings.prompts_dir,
                tracker=tracker,
            )
        except Exception as exc:
            log.error("report_failed", error=str(exc))
            run.report_markdown = offline_report(run)
    else:
        run.report_markdown = offline_report(run)
    yield Event(type="report_ready", data={"markdown": run.report_markdown})

    s.cost_usd = round(tracker.spent_usd, 5)
    s.input_tokens = tracker.usage.input_tokens + tracker.usage.cache_read_input_tokens
    s.output_tokens = tracker.usage.output_tokens
    s.llm_calls, s.cache_hits = tracker.calls, tracker.cache_hits
    s.duration_s = round(time.perf_counter() - t_start, 2)
    run.status = "done"
    _write_run(run, run_dir)
    log.info("run_finished", run_id=run_id, **s.model_dump())
    yield Event(type="run_finished", data={"run_id": run_id, "stats": s.model_dump()})
