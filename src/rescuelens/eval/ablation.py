"""Detector ablations, scored on val only.

Configurations are compared by recall at a fixed false-alarm budget (per photo): the
operating question for a search is "how many people do we find if we can afford to check
N false alarms per photo?". Test is never touched here; the winner goes through
``rescuelens bench run`` once.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rescuelens.eval.bench import CRITERIA, operating_point, threshold_for_fp_budget
from rescuelens.eval.bench_run import Split, latency_summary
from rescuelens.eval.detection import SCORE_FLOOR
from rescuelens.eval.metrics import evaluate

BUDGETS = (0.5, 1.0, 2.0)


def score_val(
    split: Split, *, budgets: Sequence[float] = BUDGETS, n_boot: int = 2000
) -> dict[str, Any]:
    labs = split.labelled()
    res: dict[str, Any] = {"ceiling": {}, "budgets": {}}
    for c, lab in labs.items():
        res["ceiling"][c] = operating_point(lab, SCORE_FLOOR, n_boot=n_boot)
        res["budgets"][c] = {}
        for b in budgets:
            t = threshold_for_fp_budget(lab, b)
            res["budgets"][c][f"{b:g}"] = (
                None if t is None else operating_point(lab, t, n_boot=n_boot)
            )
    res["ap50"] = evaluate(split.preds, split.gts, conf=0.4)["ap50"]
    res["latency"] = latency_summary(split.timings)
    return res


def _cell(op: Mapping[str, Any] | None) -> str:
    if op is None:
        return "n/a"
    lo, hi = op["recall_ci"]
    return f"{100 * op['recall']:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"


def table(rows: Sequence[Mapping[str, Any]], budget: str = "1") -> str:
    out = [
        "# Detector ablations · val (155 photos, 275 people)",
        "",
        "Recall % [95 % CI] at the threshold that allows the given false alarms per photo,",
        "chosen on val. Ceiling = every box down to score 0.05. Test is not used.",
        "",
        "| Configuration | Upscale | Recall @ ≤ "
        f"{budget} FA/photo · found | Recall @ ≤ {budget} FA/photo · IoU ≥ 0.5 | "
        "Ceiling · found | FA/photo at ceiling · found | AP50 · IoU ≥ 0.5 | "
        "Detect p50 · p95 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        s = r["score"]
        lat = s["latency"]
        ceil = s["ceiling"]["found"]
        out.append(
            f"| {r['tag']} | {r['upscale']} | {_cell(s['budgets']['found'][budget])} | "
            f"{_cell(s['budgets']['iou50'][budget])} | {_cell(ceil)} | "
            f"{ceil['fp_per_image']:.1f} | {s['ap50']:.3f} | "
            f"{lat['detect_p50_s']:.2f} · {lat['detect_p95_s']:.2f} s |"
        )
    out.append("")
    for b in sorted({k for r in rows for k in r["score"]["budgets"]["found"]}):
        if b == budget:
            continue
        out += [
            f"Recall % at ≤ {b} false alarms per photo (found · IoU ≥ 0.5): "
            + "; ".join(
                f"{r['tag']} {_cell(r['score']['budgets']['found'][b])} · "
                f"{_cell(r['score']['budgets']['iou50'][b])}"
                for r in rows
            ),
            "",
        ]
    return "\n".join(out)


def collect(folder: Path) -> list[dict[str, Any]]:
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))]
    return sorted(rows, key=lambda r: r.get("order", 0))


__all__ = ["BUDGETS", "CRITERIA", "collect", "score_val", "table"]
