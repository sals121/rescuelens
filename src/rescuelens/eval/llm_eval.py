"""Ablation: does a Claude verification pass improve detector precision on HERIDAL?

Every detection above an operating threshold is sent to Claude exactly as in production
(crop + context thumbnail, same prompt, same schema). We then compare the detector alone
against the detector filtered by Claude's verdict, and against detections re-ranked by
Claude's confidence, on the same ground truth.
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

from rescuelens.config import Settings
from rescuelens.data.voc import Box
from rescuelens.detect.crops import make_context, make_crop, make_thumbnail
from rescuelens.detect.detector import Detection
from rescuelens.eval.metrics import Pred, evaluate
from rescuelens.llm import make_backend
from rescuelens.verify.cache import VerdictCache
from rescuelens.verify.client import Verifier, VerifyRequest, load_prompt
from rescuelens.verify.cost import BudgetExceededError, CostTracker, cost_usd

# Measured with the claude-code backend (2026-11-12): ~$0.023 per Opus 5.5 verification,
# including Claude Code's own system prompt. Other models are scaled by input price.
_MEASURED_OPUS_CALL_USD = 0.023


@dataclass(slots=True)
class Verdict:
    stem: str
    box: list[float]
    score: float
    status: str  # verified | cached | error | skipped
    is_person: bool | None = None
    confidence: float | None = None
    posture: str | None = None
    priority: str | None = None
    cost_usd: float = 0.0
    latency_s: float = 0.0
    error: str | None = None

    @property
    def p_person(self) -> float | None:
        if self.is_person is None or self.confidence is None:
            return None
        return self.confidence if self.is_person else 1.0 - self.confidence


def candidates(preds: dict[str, list[Pred]], conf: float) -> dict[str, list[Pred]]:
    return {k: [p for p in v if p.score >= conf] for k, v in preds.items()}


def estimate_cost(n: int, model: str) -> float:
    from rescuelens.verify.cost import PRICES_PER_MTOK

    ratio = PRICES_PER_MTOK[model][0] / PRICES_PER_MTOK["claude-opus-5-5"][0]
    return n * _MEASURED_OPUS_CALL_USD * ratio


async def verify_candidates(
    cands: dict[str, list[Pred]],
    frames_dir: Path,
    settings: Settings,
    *,
    model: str,
    prompt_version: str = "v1",
    concurrency: int = 6,
) -> tuple[list[Verdict], CostTracker]:
    backend = make_backend(settings, model=model)
    if backend is None:
        raise RuntimeError("no Claude backend available (install Claude Code or set an API key)")
    tracker = CostTracker(max_cost_usd=settings.llm_max_cost_usd)
    verifier = Verifier(
        backend=backend,
        system_prompt=load_prompt(settings.prompts_dir, "verify", prompt_version),
        prompt_version=prompt_version,
        cache=VerdictCache(settings.runs_dir / ".cache" / "llm"),
        tracker=tracker,
        max_concurrency=concurrency,
    )
    budget_hit = asyncio.Event()
    verdicts: list[Verdict] = []
    bar = tqdm(total=sum(len(v) for v in cands.values()), desc=f"verify {model}")

    async def one(v: Verdict, req: VerifyRequest) -> None:
        if budget_hit.is_set():
            v.status, v.error = "skipped", "budget"
        else:
            try:
                res = await verifier.verify(req)
            except BudgetExceededError as exc:
                budget_hit.set()
                v.status, v.error = "skipped", str(exc)
            except Exception as exc:  # keep going; failures are reported, not hidden
                v.status, v.error = "error", f"{type(exc).__name__}: {exc}"
            else:
                r = res.verification
                v.status = "cached" if res.cached else "verified"
                v.is_person, v.confidence = r.is_person, r.confidence
                v.posture, v.priority = r.posture, r.priority
                v.cost_usd, v.latency_s = res.source_cost_usd, res.source_latency_s
        bar.update(1)

    tasks: list[asyncio.Task[None]] = []
    for stem, ps in cands.items():
        if not ps:
            continue
        with Image.open(frames_dir / f"{stem}.jpg") as im:
            img = im.convert("RGB")
        thumb, scale = make_thumbnail(img)
        for p in ps:
            det = Detection(p.box.xmin, p.box.ymin, p.box.w, p.box.h, p.score)
            req = VerifyRequest(
                crop_jpeg=make_crop(img, det),
                context_jpeg=make_context(thumb, scale, det, img.width, img.height),
            )
            v = Verdict(stem, [p.box.xmin, p.box.ymin, p.box.xmax, p.box.ymax], p.score, "pending")
            verdicts.append(v)
            tasks.append(asyncio.create_task(one(v, req)))
    await asyncio.gather(*tasks)
    bar.close()
    return verdicts, tracker


def ablation(
    verdicts: list[Verdict], gts: dict[str, list[Box]], conf: float
) -> dict[str, dict[str, float]]:
    """Detector alone vs. filtered by Claude vs. re-ranked by Claude, same candidates."""
    yolo: dict[str, list[Pred]] = defaultdict(list)
    kept: dict[str, list[Pred]] = defaultdict(list)
    rescored: dict[str, list[Pred]] = defaultdict(list)
    for v in verdicts:
        box = Box(*v.box)
        yolo[v.stem].append(Pred(box, v.score))
        # Fail open: an unverified candidate stays in the list (a SAR tool must not drop it).
        if v.is_person is not False:
            kept[v.stem].append(Pred(box, v.score))
        pp = v.p_person
        rescored[v.stem].append(Pred(box, pp if pp is not None else v.score))
    return {
        "yolo": evaluate(yolo, gts, conf=conf),
        "yolo+claude_filter": evaluate(kept, gts, conf=conf),
        # threshold 0.5 on p(person) == "Claude says person"; AP uses Claude's ranking
        "yolo+claude_rescored": evaluate(rescored, gts, conf=0.5),
    }


def summarize(verdicts: list[Verdict], tracker: CostTracker) -> dict[str, Any]:
    done = [v for v in verdicts if v.status in ("verified", "cached")]
    # cost/latency of the original calls, so a re-run from cache reports the same numbers
    lat = sorted(v.latency_s for v in done if v.latency_s > 0)
    cost = sum(v.cost_usd for v in done)
    return {
        "candidates": len(verdicts),
        "verified": len(done),
        "errors": sum(1 for v in verdicts if v.status == "error"),
        "skipped": sum(1 for v in verdicts if v.status == "skipped"),
        "said_person": sum(1 for v in done if v.is_person),
        "cost_usd": cost,
        "cost_per_call_usd": cost / len(done) if done else 0.0,
        "spent_this_run_usd": tracker.spent_usd,
        "llm_calls": tracker.calls,
        "cache_hits": tracker.cache_hits,
        "latency_p50_s": lat[len(lat) // 2] if lat else None,
        "latency_p90_s": lat[int(len(lat) * 0.9)] if lat else None,
    }


def save_verdicts(path: Path, verdicts: list[Verdict], meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"meta": meta, "verdicts": [asdict(v) for v in verdicts]}, indent=1),
        encoding="utf-8",
    )


__all__ = [
    "Verdict",
    "ablation",
    "candidates",
    "cost_usd",
    "estimate_cost",
    "save_verdicts",
    "summarize",
    "verify_candidates",
]
