"""Merge the boxes of several views of one photo (test-time augmentation, ensembles).

Weighted box fusion: boxes from all views are clustered greedily by score (IoU with the
cluster's fused box), the fused box is the score-weighted mean, and the fused score is the
mean over *all* views of each view's best score in the cluster. A person seen in every view
keeps its score; a box seen in one view of two drops to half, which is the point: noise
rarely survives a flip or a second model.
"""

from __future__ import annotations

from collections.abc import Sequence

Scored = tuple[float, float, float, float, float]  # x1, y1, x2, y2, score


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def fuse_views(views: Sequence[Sequence[Scored]], iou_thr: float = 0.55) -> list[Scored]:
    """Fused boxes, highest score first. ``views`` holds one box list per view."""
    n = len(views)
    if n == 0:
        return []
    items = sorted(((b, v) for v, boxes in enumerate(views) for b in boxes), key=lambda t: -t[0][4])
    clusters: list[list[tuple[Scored, int]]] = []
    fused: list[list[float]] = []  # running fused box per cluster
    for box, view in items:
        best, best_iou = -1, iou_thr
        for i, f in enumerate(fused):
            o = iou(box, f)
            if o >= best_iou:
                best, best_iou = i, o
        if best < 0:
            clusters.append([(box, view)])
            fused.append(list(box[:4]))
            continue
        clusters[best].append((box, view))
        w = sum(b[4] for b, _ in clusters[best])
        fused[best] = [sum(b[k] * b[4] for b, _ in clusters[best]) / w for k in range(4)]
    out: list[Scored] = []
    for members, f in zip(clusters, fused, strict=True):
        per_view: dict[int, float] = {}
        for b, v in members:
            per_view[v] = max(per_view.get(v, 0.0), b[4])
        out.append((f[0], f[1], f[2], f[3], sum(per_view.values()) / n))
    out.sort(key=lambda t: -t[4])
    return out
