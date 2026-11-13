"""Benchmark protocol: choose operating thresholds on val, apply each once to test.

Inputs are plain data (predictions, ground truth, timings, verdicts) so the protocol runs
the same in CI, on a small fixture, as it does on the full splits.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rescuelens.data.voc import Box
from rescuelens.eval.bench import (
    CRITERIA,
    Labelled,
    label_split,
    operating_point,
    percentile,
    recall_fp_curve,
    threshold_for_recall,
    threshold_max_f1,
)
from rescuelens.eval.metrics import Pred

PRODUCT_CONF = 0.4  # the app's default since phase 5, fixed before this benchmark existed


@dataclass(slots=True)
class Split:
    name: str
    preds: Mapping[str, Sequence[Pred]]
    gts: Mapping[str, Sequence[Box]]
    timings: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    area_km2: Mapping[str, float] = field(default_factory=dict)

    def labelled(self) -> dict[str, Labelled]:
        return {c: label_split(self.preds, self.gts, c) for c in CRITERIA}

    def areas(self, lab: Labelled) -> np.ndarray[Any, np.dtype[np.float64]] | None:
        if not self.area_km2:
            return None
        return np.array([self.area_km2[s] for s in lab.stems], dtype=np.float64)


@dataclass(slots=True)
class ClaudeArm:
    """Detections kept by Claude (fail open), plus what the calls cost."""

    name: str
    conf: float
    kept: dict[str, list[Pred]]
    call_cost_usd: list[float]
    call_latency_s: list[float]
    per_image_latency_s: dict[str, float]


def load_claude_arm(path: Path, name: str) -> ClaudeArm:
    data = json.loads(path.read_text(encoding="utf-8"))
    kept: dict[str, list[Pred]] = defaultdict(list)
    per_image: dict[str, float] = defaultdict(float)
    costs: list[float] = []
    lats: list[float] = []
    for v in data["verdicts"]:
        if v["is_person"] is not False:  # unverified candidates stay (fail open)
            kept[v["stem"]].append(Pred(Box(*v["box"]), v["score"]))
        if v["status"] in ("verified", "cached"):
            costs.append(v["cost_usd"])
            lats.append(v["latency_s"])
            per_image[v["stem"]] += v["latency_s"]
    return ClaudeArm(name, float(data["meta"]["conf"]), dict(kept), costs, lats, dict(per_image))


def _ops(split: Split, labs: Mapping[str, Labelled], conf: float, n_boot: int) -> dict[str, Any]:
    return {
        c: operating_point(lab, conf, area_km2=split.areas(lab), n_boot=n_boot)
        for c, lab in labs.items()
    }


def latency_summary(timings: Mapping[str, Mapping[str, float]]) -> dict[str, Any] | None:
    if not timings:
        return None
    det = [t["detect_s"] for t in timings.values()]
    tot = [t["detect_s"] + t["load_s"] for t in timings.values()]
    return {
        "frames": len(det),
        "detect_p50_s": percentile(det, 50),
        "detect_p95_s": percentile(det, 95),
        "with_decode_p50_s": percentile(tot, 50),
        "with_decode_p95_s": percentile(tot, 95),
    }


def run_protocol(
    val: Split,
    test: Split,
    *,
    targets: Sequence[float] = (0.90, 0.95, 0.98),
    claude: Sequence[ClaudeArm] = (),
    n_boot: int = 2000,
) -> dict[str, Any]:
    val_labs, test_labs = val.labelled(), test.labelled()

    rules: list[tuple[str, str, float | None]] = [
        ("product default", f"fixed at {PRODUCT_CONF} before this benchmark", PRODUCT_CONF),
        ("max F1 (IoU 0.5)", "val", threshold_max_f1(val_labs["iou50"])),
    ]
    for t in targets:
        rules.append(
            (f"recall >= {t:.2f} (found)", "val", threshold_for_recall(val_labs["found"], t))
        )

    rows: list[dict[str, Any]] = []
    for name, source, conf in rules:
        row: dict[str, Any] = {"rule": name, "chosen_on": source, "conf": conf}
        if conf is not None:
            row["val"] = _ops(val, val_labs, conf, n_boot)
            row["test"] = _ops(test, test_labs, conf, n_boot)
        rows.append(row)

    arms: list[dict[str, Any]] = []
    for arm in claude:
        kept_split = Split(test.name, arm.kept, test.gts, area_km2=test.area_km2)
        labs = kept_split.labelled()
        lat = list(arm.per_image_latency_s.get(s, 0.0) for s in test_labs["found"].stems)
        arms.append(
            {
                "name": arm.name,
                "conf": arm.conf,
                "test": _ops(kept_split, labs, arm.conf, n_boot),
                "calls": len(arm.call_cost_usd),
                "cost_usd": sum(arm.call_cost_usd),
                "cost_per_image_usd": sum(arm.call_cost_usd) / test_labs["found"].n_images,
                "call_latency_p50_s": percentile(arm.call_latency_s, 50),
                "call_latency_p95_s": percentile(arm.call_latency_s, 95),
                # sequential sum per image: an upper bound (the app runs 6 calls at once)
                "image_latency_serial_p50_s": percentile(lat, 50),
                "image_latency_serial_p95_s": percentile(lat, 95),
            }
        )

    return {
        "splits": {
            s.name: {
                "images": labs["found"].n_images,
                "people": int(labs["found"].n_gt.sum()),
                "footprint_km2": sum(s.area_km2.values()) if s.area_km2 else None,
            }
            for s, labs in ((val, val_labs), (test, test_labs))
        },
        "thresholds": rows,
        "claude": arms,
        "curves": {
            s.name: {c: recall_fp_curve(lab) for c, lab in labs.items()}
            for s, labs in ((val, val_labs), (test, test_labs))
        },
        "latency": {
            val.name: latency_summary(val.timings),
            test.name: latency_summary(test.timings),
        },
        "bootstrap": {"replicates": n_boot, "interval": "95 % percentile", "unit": "image"},
    }


# ---------------------------------------------------------------- markdown


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{100 * v:.1f}"


def _ci(op: Mapping[str, Any], key: str, pct: bool = True, nd: int = 2) -> str:
    v, (lo, hi) = op.get(key), op.get(f"{key}_ci", [None, None])
    if v is None:
        return "n/a"
    if pct:
        return f"{_pct(v)} [{_pct(lo)}, {_pct(hi)}]"
    return f"{v:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def _row(label: str, conf: float | None, op: Mapping[str, Any]) -> str:
    km2 = _ci(op, "fp_per_km2", pct=False, nd=0) if "fp_per_km2" in op else "n/a"
    c = "n/a" if conf is None else f"{conf:.3f}"
    return (
        f"| {label} | {c} | {op['tp']}/{op['people']} | {_ci(op, 'recall')} | "
        f"{_ci(op, 'precision')} | {_ci(op, 'fp_per_image', pct=False)} | {km2} |"
    )


HEAD = (
    "| Rule | Conf | Found | Recall % [95 % CI] | Precision % | False alarms / image | "
    "False alarms / km² |\n|---|---:|---:|---:|---:|---:|---:|"
)


def to_markdown(res: Mapping[str, Any], title: str) -> str:
    out = [f"# {title}", ""]
    for name, s in res["splits"].items():
        fp = f", {s['footprint_km2']:.3f} km² (assumed altitude)" if s["footprint_km2"] else ""
        out.append(f"- **{name}**: {s['images']} images, {s['people']} people{fp}")
    out.append("")
    for crit, desc in (
        ("found", "person found: IoU ≥ 0.1 or a centre inside the other box (operational)"),
        ("iou50", "IoU ≥ 0.5, duplicates count as false positives (literature)"),
    ):
        for split in ("val", "test"):
            out += [f"## {split.capitalize()} · {desc}", "", HEAD]
            for row in res["thresholds"]:
                if row["conf"] is None:
                    out.append(f"| {row['rule']} | not reachable on val | | | | | |")
                    continue
                out.append(_row(row["rule"], row["conf"], row[split][crit]))
            if split == "test":
                for arm in res["claude"]:
                    out.append(
                        _row(f"YOLO + {arm['name']} (filter)", arm["conf"], arm["test"][crit])
                    )
            out.append("")
    if res["claude"]:
        out += [
            "## Cost and latency of the Claude arms (test)",
            "",
            "| Arm | Calls | Cost / image | Call p50 / p95 | Image p50 / p95 (serial) |",
            "|---|---:|---:|---:|---:|",
        ]
        for a in res["claude"]:
            out.append(
                f"| {a['name']} | {a['calls']} | ${a['cost_per_image_usd']:.4f} | "
                f"{a['call_latency_p50_s']:.1f} / {a['call_latency_p95_s']:.1f} s | "
                f"{a['image_latency_serial_p50_s']:.1f} / {a['image_latency_serial_p95_s']:.1f} s |"
            )
        out.append("")
    out += [
        "## Detector latency per image",
        "",
        "| Split | Detect p50 / p95 | + JPEG decode p50 / p95 |",
        "|---|---:|---:|",
    ]
    for split, lat in res["latency"].items():
        if lat:
            out.append(
                f"| {split} | {lat['detect_p50_s']:.2f} / {lat['detect_p95_s']:.2f} s | "
                f"{lat['with_decode_p50_s']:.2f} / {lat['with_decode_p95_s']:.2f} s |"
            )
    out.append("")
    return "\n".join(out)


def plot_curves(res: Mapping[str, Any], path: Path) -> None:
    """Recall vs. false alarms per image (val dashed, test solid), Claude arms as points."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    titles = {"found": "Person found (operational)", "iou50": "IoU ≥ 0.5 (literature)"}
    for ax, crit in zip(axes, ("found", "iou50"), strict=True):
        for split, style in (("val", "--"), ("test", "-")):
            cv = res["curves"][split][crit]
            ax.plot(
                [r["fp_per_image"] for r in cv],
                [100 * r["recall"] for r in cv],
                style,
                color="#18221f",
                lw=1.6,
                label=f"YOLO, {split}",
            )
        for row in res["thresholds"]:
            if row["conf"] is None:
                continue
            op = row["test"][crit]
            ax.plot(op["fp_per_image"], 100 * op["recall"], "o", color="#3a86d4", ms=5)
            ax.annotate(
                f"{row['conf']:.2f}",
                (op["fp_per_image"], 100 * op["recall"]),
                textcoords="offset points",
                xytext=(5, -10),
                fontsize=8,
                color="#3a86d4",
            )
        for arm, color in zip(res["claude"], ("#ff5a1f", "#c38f22"), strict=False):
            op = arm["test"][crit]
            ax.errorbar(
                op["fp_per_image"],
                100 * op["recall"],
                yerr=[
                    [100 * (op["recall"] - op["recall_ci"][0])],
                    [100 * (op["recall_ci"][1] - op["recall"])],
                ],
                fmt="D",
                color=color,
                ms=6,
                capsize=3,
                label=f"YOLO + {arm['name']}, test",
            )
        ax.set_title(titles[crit], fontsize=11)
        ax.set_xlabel("False alarms per image")
        ax.set_xlim(0, 6.5)
        ax.set_ylim(40, 100)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Recall (% of people)")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Blue dots: test at thresholds chosen on val (labels = confidence)", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
