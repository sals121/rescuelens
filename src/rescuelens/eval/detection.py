"""Run a detector over the full-resolution test frames and score it."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from rescuelens.data.voc import Box
from rescuelens.eval.metrics import Pred, evaluate, pr_curve

# Low floor so AP sees the whole score range; P/R/F1 are reported at an operating threshold.
SCORE_FLOOR = 0.05


def predict_frames(
    weights: Path, frames_dir: Path, *, use_sahi: bool, device: str
) -> tuple[dict[str, list[Pred]], float]:
    """Predictions per frame stem, and mean seconds per frame (after warm-up)."""
    preds, timings = predict_frames_timed(weights, frames_dir, use_sahi=use_sahi, device=device)
    times = [t["detect_s"] for t in timings.values()]
    return preds, (sum(times) / len(times) if times else float("nan"))


def predict_frames_timed(
    weights: Path,
    frames_dir: Path,
    *,
    use_sahi: bool,
    device: str,
) -> tuple[dict[str, list[Pred]], dict[str, dict[str, float]]]:
    """Predictions per frame stem, and per-frame timings in seconds.

    ``load_s`` is JPEG decoding, ``detect_s`` the detector (tiling, inference, merging).
    A warm-up pass on the first frame runs before timing so CUDA start-up is not counted.
    """
    from rescuelens.detect.detector import DetectorConfig, PersonDetector

    det = PersonDetector(
        DetectorConfig(weights=weights, confidence=SCORE_FLOOR, device=device, use_sahi=use_sahi)
    )
    paths = sorted(frames_dir.glob("*.jpg"))
    out: dict[str, list[Pred]] = {}
    timings: dict[str, dict[str, float]] = {}
    for i, p in enumerate(tqdm(paths, desc=f"{weights.stem} {'sahi' if use_sahi else 'full'}")):
        t0 = time.perf_counter()
        with Image.open(p) as im:
            arr = np.asarray(im.convert("RGB"))
        load_s = time.perf_counter() - t0
        if i == 0:
            det.detect(arr)  # warm-up
        t1 = time.perf_counter()
        dets = det.detect(arr)
        timings[p.stem] = {"load_s": load_s, "detect_s": time.perf_counter() - t1}
        out[p.stem] = [Pred(Box(d.x, d.y, d.x + d.w, d.y + d.h), d.score) for d in dets]
    return out, timings


def save_preds(path: Path, preds: dict[str, list[Pred]], meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "meta": meta,
        "preds": {
            k: [[p.box.xmin, p.box.ymin, p.box.xmax, p.box.ymax, p.score] for p in v]
            for k, v in preds.items()
        },
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def load_preds(path: Path) -> tuple[dict[str, list[Pred]], dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    preds = {k: [Pred(Box(*r[:4]), r[4]) for r in v] for k, v in data["preds"].items()}
    return preds, data["meta"]


def best_f1(
    preds: dict[str, list[Pred]], gts: dict[str, list[Box]]
) -> tuple[float, float, float, float]:
    """(conf, precision, recall, f1) at the score threshold that maximises F1."""
    best = (0.0, 0.0, 0.0, 0.0)
    for score, p, r in pr_curve(preds, gts):
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        if f1 > best[3]:
            best = (score, p, r, f1)
    return best


def score_run(
    preds: dict[str, list[Pred]], gts: dict[str, list[Box]], conf: float
) -> dict[str, float]:
    m = evaluate(preds, gts, conf=conf)
    c, p, r, f1 = best_f1(preds, gts)
    return {**m, "best_f1": f1, "best_f1_conf": c, "best_f1_precision": p, "best_f1_recall": r}
