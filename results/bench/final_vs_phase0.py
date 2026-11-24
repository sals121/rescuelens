"""Phase 0 detector vs the final A5 configuration on test, same photos.

Reporting only: A5 and its operating rules were fixed on val before test was scored.
Run from the repository root: ``python results/bench/final_vs_phase0.py``.
"""

from pathlib import Path

import numpy as np

from rescuelens.data.heridal import ground_truth
from rescuelens.eval.bench import label_split, paired_difference, threshold_for_fp_budget
from rescuelens.eval.detection import load_preds
from rescuelens.eval.strata import PhotoMeta

ROOT = Path("data/heridal")
PREDS = Path("results/bench/preds")
UNSEEN = {"FC550", "no EXIF"}  # cameras in test but not in train or val


def pct(x: float) -> str:
    return f"{100 * x:+.1f}"


def main() -> None:
    gv, gt = ground_truth(ROOT, "val"), ground_truth(ROOT, "test")
    runs = {
        name: (load_preds(PREDS / f"{tag}_val.json")[0], load_preds(PREDS / f"{tag}_test.json")[0])
        for name, tag in (("phase0", "rescuelens-yolo26s-heridal_sahi"), ("A5", "A5-s1024-i1280"))
    }
    (p0v, p0t), (p5v, p5t) = runs["phase0"], runs["A5"]
    t0 = threshold_for_fp_budget(label_split(p0v, gv, "found"), 1.0)
    t5 = threshold_for_fp_budget(label_split(p5v, gv, "found"), 1.0)
    assert t0 is not None and t5 is not None
    for crit in ("found", "iou50"):
        l0, l5 = label_split(p0t, gt, crit), label_split(p5t, gt, crit)
        for what, a, b in (
            ("conf 0.40 both", 0.4, 0.4),
            (f"<=1 FA/photo on val ({t0:.3f} vs {t5:.3f})", t0, t5),
        ):
            d = paired_difference(l0, a, l5, b)
            lo, hi = d["ci"]
            print(f"{what}, {crit}: {pct(d['difference'])} [{pct(lo)}, {pct(hi)}]")

    frames = ROOT / "test_full" / "images"
    meta = {s: PhotoMeta.read(frames / f"{s}.jpg") for s in gt}
    rng = np.random.default_rng(0)
    for name, (_, preds) in runs.items():
        lab = label_split(preds, gt, "found")
        hits = lab.found(0.4)
        for group in ("seen", "unseen"):
            stems = [s for s in lab.stems if (meta[s].camera in UNSEEN) == (group == "unseen")]
            n = np.array([len(gt[s]) for s in stems])
            k = np.array([len(set(hits[s])) for s in stems])
            boot = []
            for _ in range(2000):
                i = rng.integers(0, len(stems), len(stems))
                boot.append(k[i].sum() / max(1, n[i].sum()))
            lo, hi = np.percentile(boot, [2.5, 97.5])
            print(
                f"{name}, {group} cameras, found @ conf 0.40: {100 * k.sum() / n.sum():.1f} "
                f"[{100 * lo:.1f}, {100 * hi:.1f}] ({len(stems)} photos, {n.sum()} people)"
            )


if __name__ == "__main__":
    main()
