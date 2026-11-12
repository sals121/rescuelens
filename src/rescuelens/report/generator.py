"""Rescue briefing: written by Claude from the run data, with a deterministic offline fallback."""

from __future__ import annotations

import json
from pathlib import Path

from rescuelens.logging import get_logger
from rescuelens.schemas import Candidate, Run
from rescuelens.verify.backends import BackendError, LLMBackend, RetryableBackendError
from rescuelens.verify.cost import CostTracker

log = get_logger(__name__)

_PRIORITY_RANK = {"high": 0, "medium": 1, "low": 2}


def ranked_people(run: Run) -> list[Candidate]:
    people = [c for c in run.candidates if c.verification and c.verification.is_person]
    return sorted(
        people,
        key=lambda c: (
            _PRIORITY_RANK[c.verification.priority] if c.verification else 3,
            -(c.verification.confidence if c.verification else 0.0),
        ),
    )


def run_digest(run: Run) -> str:
    """Compact JSON the model reads; only facts produced by the pipeline."""
    simulated = any(f.geo_source == "simulated" for f in run.frames)
    targets = [
        {
            "id": c.id,
            "frame": c.frame_id,
            "priority": c.verification.priority,
            "posture": c.verification.posture,
            "confidence": round(c.verification.confidence, 2),
            "lat": round(c.lat, 5),
            "lon": round(c.lon, 5),
            "evidence": c.verification.evidence,
        }
        for c in ranked_people(run)
        if c.verification
    ]
    return json.dumps(
        {
            "frames_analysed": run.stats.n_frames,
            "candidates_detected": run.stats.n_candidates,
            "candidates_verified": run.stats.n_verified,
            "judged_people": len(targets),
            "coordinates_simulated": simulated,
            "targets": targets,
        },
        indent=1,
    )


def offline_report(run: Run) -> str:
    """Template briefing used when the LLM is disabled or unavailable."""
    people = ranked_people(run)
    simulated = any(f.geo_source == "simulated" for f in run.frames)
    lines = [
        "## Summary",
        f"{run.stats.n_frames} frames analysed, {run.stats.n_candidates} candidates detected, "
        f"{len(people)} judged to be people.",
        "",
        "## Priority targets",
    ]
    if people:
        lines += [
            "| Rank | ID | Priority | Posture | Confidence | Lat | Lon |",
            "|---|---|---|---|---|---|---|",
        ]
        for i, c in enumerate(people[:15], 1):
            v = c.verification
            assert v is not None
            lines.append(
                f"| {i} | {c.id} | {v.priority} | {v.posture} | {v.confidence:.2f} "
                f"| {c.lat:.5f} | {c.lon:.5f} |"
            )
    else:
        lines.append("No verified people. Review unverified candidates manually.")
    lines += ["", "## Caveats", "- Automated verdicts: confirm every target on the ground."]
    if simulated:
        lines.append("- Coordinates are SIMULATED for demonstration purposes.")
    return "\n".join(lines)


async def llm_report(
    run: Run,
    *,
    backend: LLMBackend,
    prompts_dir: Path,
    tracker: CostTracker,
    version: str = "v1",
) -> str:
    system = (prompts_dir / f"report_{version}.md").read_text(encoding="utf-8").strip()
    tracker.check_budget()
    try:
        response = await backend.write(
            system=system,
            text=f"Survey results:\n```json\n{run_digest(run)}\n```",
            effort="low",
        )
    except (BackendError, RetryableBackendError) as exc:
        log.warning("report_failed", error=str(exc))
        return offline_report(run)
    tracker.record_cost(response.usage, response.cost_usd)
    return response.text.strip() or offline_report(run)
