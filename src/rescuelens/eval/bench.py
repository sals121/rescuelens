"""Person-level benchmark: two hit criteria, bootstrap intervals, recall vs. false alarms.

Rules this module enforces:

* Every prediction of a split is labelled once (greedy, one-to-one, highest score first).
  A prediction's label only depends on higher-scored predictions, so any confidence
  threshold is just a filter over the same labels and curves are exact.
* Operating thresholds are chosen on val and then applied, unchanged, to test.
* Every rate comes with a 95 % percentile bootstrap interval, resampling whole images.

Two criteria, both reported:

``iou50``
    The literature's: a detection counts if IoU >= 0.5 with an unmatched person. Extra
    detections of an already-matched person are false positives (PASCAL VOC).
``found``
    Operational: a person counts as found if a detection overlaps them with IoU >= 0.1 or
    either box's centre lies inside the other. Someone checking that detection would see
    the person. A second detection on the same person is a *duplicate*: not a person
    found twice, and not a false alarm either. False alarms are detections on no person.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from rescuelens.data.voc import Box
from rescuelens.eval.metrics import Pred, iou

Criterion = Literal["iou50", "found"]
CRITERIA: tuple[Criterion, ...] = ("iou50", "found")

FP, TP, DUP = 0, 1, 2
# per-image count columns
C_GT, C_TP, C_DUP, C_FP = 0, 1, 2, 3

FOUND_MIN_IOU = 0.1


def _centre_in(a: Box, b: Box) -> bool:
    cx, cy = (a.xmin + a.xmax) / 2, (a.ymin + a.ymax) / 2
    return b.xmin <= cx <= b.xmax and b.ymin <= cy <= b.ymax


def match_quality(criterion: Criterion, p: Box, g: Box) -> float:
    """> 0 if ``p`` hits ``g`` under ``criterion``; larger is a better match."""
    v = iou(p, g)
    if criterion == "iou50":
        return v if v >= 0.5 else 0.0
    if v >= FOUND_MIN_IOU or _centre_in(p, g) or _centre_in(g, p):
        return max(v, 1e-9)
    return 0.0


def label_image(
    preds: Sequence[Pred], gts: Sequence[Box], criterion: Criterion
) -> list[tuple[float, int, int]]:
    """(score, label, matched gt index or -1) per prediction, highest score first."""
    used = [False] * len(gts)
    out: list[tuple[float, int, int]] = []
    for p in sorted(preds, key=lambda q: q.score, reverse=True):
        best, best_j, any_hit = 0.0, -1, False
        for j, g in enumerate(gts):
            q = match_quality(criterion, p.box, g)
            if q <= 0:
                continue
            any_hit = True
            if not used[j] and q > best:
                best, best_j = q, j
        if best_j >= 0:
            used[best_j] = True
            out.append((p.score, TP, best_j))
        else:
            out.append((p.score, DUP if any_hit else FP, -1))
    return out


@dataclass(frozen=True, slots=True)
class Labelled:
    """All predictions of one split under one criterion."""

    criterion: Criterion
    stems: tuple[str, ...]
    n_gt: NDArray[np.int64]  # per image
    image: NDArray[np.int64]  # per prediction: index into stems
    score: NDArray[np.float64]
    label: NDArray[np.int64]
    gt_index: NDArray[np.int64]  # matched person within its image, -1 if none

    @property
    def n_images(self) -> int:
        return len(self.stems)

    def counts(self, conf: float) -> NDArray[np.int64]:
        """Per-image (n_gt, tp, dup, fp) for detections with score >= conf."""
        keep = self.score >= conf
        out = np.zeros((self.n_images, 4), dtype=np.int64)
        out[:, C_GT] = self.n_gt
        for col, lab in ((C_TP, TP), (C_DUP, DUP), (C_FP, FP)):
            sel = keep & (self.label == lab)
            out[:, col] = np.bincount(self.image[sel], minlength=self.n_images)
        return out

    def found(self, conf: float) -> dict[str, list[int]]:
        """Indices of the people found at ``conf``, per image."""
        sel = (self.score >= conf) & (self.label == TP)
        out: dict[str, list[int]] = {s: [] for s in self.stems}
        for i, j in zip(self.image[sel], self.gt_index[sel], strict=True):
            out[self.stems[i]].append(int(j))
        return out


def label_split(
    preds: Mapping[str, Sequence[Pred]], gts: Mapping[str, Sequence[Box]], criterion: Criterion
) -> Labelled:
    """Ground truth defines the image set; predictions for other images are ignored."""
    stems = tuple(sorted(gts))
    image: list[int] = []
    score: list[float] = []
    label: list[int] = []
    gt_index: list[int] = []
    for i, stem in enumerate(stems):
        for s, lab, j in label_image(preds.get(stem, ()), gts[stem], criterion):
            image.append(i)
            score.append(s)
            label.append(lab)
            gt_index.append(j)
    return Labelled(
        criterion=criterion,
        stems=stems,
        n_gt=np.array([len(gts[s]) for s in stems], dtype=np.int64),
        image=np.array(image, dtype=np.int64),
        score=np.array(score, dtype=np.float64),
        label=np.array(label, dtype=np.int64),
        gt_index=np.array(gt_index, dtype=np.int64),
    )


def _rates(
    c: NDArray[np.int64], criterion: Criterion, n_images: int, area_km2: float | None
) -> dict[str, NDArray[np.float64]]:
    """Metrics from summed counts. ``c`` is (..., 4); works on bootstrap stacks too."""
    gt, tp, dup, fp = (c[..., k].astype(np.float64) for k in (C_GT, C_TP, C_DUP, C_FP))
    det = tp + dup + fp
    # iou50: duplicates are false positives (VOC). found: they point at a real person.
    false_alarms = fp + dup if criterion == "iou50" else fp
    on_person = tp if criterion == "iou50" else tp + dup
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(gt > 0, tp / gt, np.nan)
        precision = np.where(det > 0, on_person / det, np.nan)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    out = {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "fp_per_image": false_alarms / n_images,
    }
    if area_km2:
        out["fp_per_km2"] = false_alarms / area_km2
    return out


def _finite(x: float) -> float | None:
    return x if math.isfinite(x) else None


def operating_point(
    lab: Labelled,
    conf: float,
    *,
    area_km2: NDArray[np.float64] | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Counts, rates and 95 % bootstrap intervals (resampling images) at ``conf``.

    ``area_km2`` is the ground footprint of each image (same order as ``lab.stems``).
    """
    c = lab.counts(conf)
    total = c.sum(axis=0)
    total_area = float(area_km2.sum()) if area_km2 is not None else None
    point = _rates(total, lab.criterion, lab.n_images, total_area)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, lab.n_images, size=(n_boot, lab.n_images))
    boot_c = c[idx].sum(axis=1)
    boot_area = area_km2[idx].sum(axis=1) if area_km2 is not None else None
    boot = _rates(boot_c, lab.criterion, lab.n_images, None)
    if boot_area is not None:
        fa = boot_c[:, C_FP] + (boot_c[:, C_DUP] if lab.criterion == "iou50" else 0)
        boot["fp_per_km2"] = fa / boot_area

    out: dict[str, Any] = {
        "criterion": lab.criterion,
        "conf": conf,
        "images": lab.n_images,
        "people": int(total[C_GT]),
        "detections": int(total[C_TP] + total[C_DUP] + total[C_FP]),
        "tp": int(total[C_TP]),
        "duplicates": int(total[C_DUP]),
        "fp": int(total[C_FP]),
        "missed": int(total[C_GT] - total[C_TP]),
    }
    for k, v in point.items():
        b = boot[k][np.isfinite(boot[k])]
        lo, hi = np.percentile(b, [2.5, 97.5]) if b.size else (math.nan, math.nan)
        out[k] = _finite(float(v))
        out[f"{k}_ci"] = [_finite(float(lo)), _finite(float(hi))]
    return out


def threshold_for_recall(lab: Labelled, target: float) -> float | None:
    """Highest threshold whose recall is >= ``target`` (None if never reached)."""
    n_gt = int(lab.n_gt.sum())
    if n_gt == 0:
        return None
    order = np.argsort(-lab.score, kind="stable")
    cum_tp = np.cumsum(lab.label[order] == TP)
    hit = np.nonzero(cum_tp >= target * n_gt - 1e-9)[0]
    return float(lab.score[order][hit[0]]) if hit.size else None


def threshold_max_f1(lab: Labelled) -> float:
    grid = np.unique(lab.score)
    if grid.size == 0:
        return 1.0
    best_conf, best = float(grid[-1]), -1.0
    for t in grid[::-1]:
        f1 = float(
            _rates(lab.counts(float(t)).sum(axis=0), lab.criterion, lab.n_images, None)["f1"]
        )
        if f1 > best:
            best_conf, best = float(t), f1
    return best_conf


def recall_fp_curve(lab: Labelled, *, step: float = 0.01) -> list[dict[str, float]]:
    """Recall and false alarms per image on a fixed confidence grid (floor .. 0.99)."""
    floor = float(lab.score.min()) if lab.score.size else 0.0
    grid = np.round(np.arange(max(floor, step), 1.0, step), 4)
    rows: list[dict[str, float]] = []
    for t in grid:
        r = _rates(lab.counts(float(t)).sum(axis=0), lab.criterion, lab.n_images, None)
        rows.append(
            {
                "conf": float(t),
                "recall": float(r["recall"]),
                "fp_per_image": float(r["fp_per_image"]),
                "precision": float(np.nan_to_num(r["precision"], nan=1.0)),
            }
        )
    return rows


def percentile(values: Sequence[float], q: float) -> float | None:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else None


def threshold_for_fp_budget(lab: Labelled, fp_per_image: float) -> float | None:
    """Lowest threshold whose false alarms per image stay within the budget.

    That is the threshold with the highest recall under the budget. Ties in score are kept
    together (a threshold cannot split them). None if even the top score exceeds it.
    """
    order = np.argsort(-lab.score, kind="stable")
    s = lab.score[order]
    bad = lab.label[order] == FP
    if lab.criterion == "iou50":
        bad |= lab.label[order] == DUP
    cum_fa = np.cumsum(bad)
    last_of_tie = np.r_[s[1:] != s[:-1], True] if s.size else np.array([], dtype=bool)
    ok = (cum_fa <= fp_per_image * lab.n_images + 1e-9) & last_of_tie
    idx = np.nonzero(ok)[0]
    return float(s[idx[-1]]) if idx.size else None


def paired_difference(
    a: Labelled,
    conf_a: float,
    b: Labelled,
    conf_b: float,
    metric: str = "recall",
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """``metric(b) - metric(a)`` on the same photos, with a paired 95 % bootstrap interval.

    Both configurations are resampled with the same photos in every replicate, so photo
    difficulty cancels out. ``share_above_zero`` is the fraction of replicates where b wins.
    """
    if a.stems != b.stems or a.criterion != b.criterion:
        raise ValueError("paired comparison needs the same photos and criterion")
    ca, cb = a.counts(conf_a), b.counts(conf_b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.n_images, size=(n_boot, a.n_images))
    ra = _rates(ca[idx].sum(axis=1), a.criterion, a.n_images, None)[metric]
    rb = _rates(cb[idx].sum(axis=1), b.criterion, b.n_images, None)[metric]
    d = rb - ra
    d = d[np.isfinite(d)]
    point = float(
        _rates(cb.sum(axis=0), b.criterion, b.n_images, None)[metric]
        - _rates(ca.sum(axis=0), a.criterion, a.n_images, None)[metric]
    )
    lo, hi = np.percentile(d, [2.5, 97.5]) if d.size else (math.nan, math.nan)
    return {
        "difference": _finite(point),
        "ci": [_finite(float(lo)), _finite(float(hi))],
        "share_above_zero": float((d > 0).mean()) if d.size else None,
    }
