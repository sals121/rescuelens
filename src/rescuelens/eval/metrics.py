"""Detection metrics on full frames: greedy IoU matching, AP (all-point), P/R/F1.

Kept independent of Ultralytics so YOLO, YOLO+SAHI and YOLO+Claude are scored with the
exact same code, on the original 4000x3000 frames rather than on tiles.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rescuelens.data.voc import Box

IOU_THRESHOLDS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))  # 0.50 ... 0.95


@dataclass(frozen=True, slots=True)
class Pred:
    box: Box
    score: float


def iou(a: Box, b: Box) -> float:
    iw = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    ih = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / (a.area + b.area - inter)


def match_image(preds: Sequence[Pred], gts: Sequence[Box], iou_thr: float) -> list[bool]:
    """Greedy matching in descending score order; each ground truth is matched once.

    Returns a TP flag per prediction, in the order of ``sorted(preds, key=score, desc)``.
    """
    order = sorted(preds, key=lambda p: p.score, reverse=True)
    used = [False] * len(gts)
    flags: list[bool] = []
    for p in order:
        best, best_j = iou_thr, -1
        for j, g in enumerate(gts):
            if not used[j] and (v := iou(p.box, g)) >= best:
                best, best_j = v, j
        if best_j >= 0:
            used[best_j] = True
        flags.append(best_j >= 0)
    return flags


def average_precision(scored: Sequence[tuple[float, bool]], n_gt: int) -> float:
    """All-point interpolated AP (PASCAL VOC 2010+) from (score, is_tp) pairs."""
    if n_gt == 0:
        return float("nan")
    ranked = sorted(scored, key=lambda t: t[0], reverse=True)
    tp = fp = 0
    recalls: list[float] = []
    precisions: list[float] = []
    for _, is_tp in ranked:
        tp += is_tp
        fp += not is_tp
        recalls.append(tp / n_gt)
        precisions.append(tp / (tp + fp))
    # precision envelope, right to left
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    ap, prev_r = 0.0, 0.0
    for r, p in zip(recalls, precisions, strict=True):
        ap += (r - prev_r) * p
        prev_r = r
    return ap


def _scored(
    preds: Mapping[str, Sequence[Pred]], gts: Mapping[str, Sequence[Box]], iou_thr: float
) -> list[tuple[float, bool]]:
    out: list[tuple[float, bool]] = []
    for stem, g in gts.items():
        ps = sorted(preds.get(stem, ()), key=lambda p: p.score, reverse=True)
        out.extend(zip((p.score for p in ps), match_image(ps, g, iou_thr), strict=True))
    return out


def evaluate(
    preds: Mapping[str, Sequence[Pred]],
    gts: Mapping[str, Sequence[Box]],
    *,
    conf: float,
    iou_thr: float = 0.5,
) -> dict[str, float]:
    """AP50, AP50-95 over all scores; precision/recall/F1/FP-per-image at ``conf``."""
    n_gt = sum(len(g) for g in gts.values())
    scored = _scored(preds, gts, iou_thr)
    kept = [t for t in scored if t[0] >= conf]
    tp = sum(1 for _, ok in kept if ok)
    fp = len(kept) - tp
    p = tp / len(kept) if kept else 0.0
    r = tp / n_gt if n_gt else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    ap_by_thr = [average_precision(_scored(preds, gts, t), n_gt) for t in IOU_THRESHOLDS]
    return {
        "images": len(gts),
        "ground_truth": n_gt,
        "conf": conf,
        "detections": len(kept),
        "tp": tp,
        "fp": fp,
        "fn": n_gt - tp,
        "precision": p,
        "recall": r,
        "f1": f1,
        "fp_per_image": fp / len(gts) if gts else 0.0,
        "ap50": average_precision(scored, n_gt),
        "ap50_95": sum(ap_by_thr) / len(ap_by_thr),
    }


def pr_curve(
    preds: Mapping[str, Sequence[Pred]], gts: Mapping[str, Sequence[Box]], iou_thr: float = 0.5
) -> list[tuple[float, float, float]]:
    """(score, precision, recall) at every detection, highest score first."""
    n_gt = sum(len(g) for g in gts.values())
    tp = fp = 0
    out: list[tuple[float, float, float]] = []
    for score, ok in sorted(_scored(preds, gts, iou_thr), key=lambda t: t[0], reverse=True):
        tp += ok
        fp += not ok
        out.append((score, tp / (tp + fp), tp / n_gt if n_gt else 0.0))
    return out
