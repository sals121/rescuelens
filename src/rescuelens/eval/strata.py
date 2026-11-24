"""Recall per stratum of people (size in pixels, ground resolution, camera).

Answers "which people does the detector miss?". Intervals resample whole photos, as in
:mod:`rescuelens.eval.bench`, restricted to the people of each stratum.
"""

from __future__ import annotations

import itertools
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rescuelens.data.voc import Box
from rescuelens.eval.bench import Labelled
from rescuelens.ingest.camera import read_camera, read_relative_altitude

SIZE_BINS_PX = (0, 30, 40, 50, 60, 80, 120, math.inf)
GSD_BINS_CM = (0, 1.25, 1.5, 1.75, 2.0, 2.5, math.inf)


@dataclass(frozen=True, slots=True)
class PhotoMeta:
    camera: str
    rel_alt_m: float | None  # DJI only, above take-off
    gsd_cm: float | None  # from rel_alt_m, flat-ground assumption

    @classmethod
    def read(cls, path: Path) -> PhotoMeta:
        cam = read_camera(path)
        alt = read_relative_altitude(path)
        gsd = 100 * cam.gsd_m(alt) if alt is not None and alt > 0 else None
        return cls(cam.model or "no EXIF", alt, gsd)


@dataclass(frozen=True, slots=True)
class Person:
    image: int
    size_px: float  # sqrt(w * h) of the annotated box
    camera: str
    gsd_cm: float | None


def people(
    lab: Labelled, gts: Mapping[str, Sequence[Box]], meta: Mapping[str, PhotoMeta]
) -> list[Person]:
    out: list[Person] = []
    for i, stem in enumerate(lab.stems):
        m = meta[stem]
        for b in gts[stem]:
            out.append(Person(i, math.sqrt(b.w * b.h), m.camera, m.gsd_cm))
    return out


def found_flags(lab: Labelled, gts: Mapping[str, Sequence[Box]], conf: float) -> list[bool]:
    """One flag per person, in the order of :func:`people`."""
    hits = lab.found(conf)
    return [j in set(hits[s]) for s in lab.stems for j in range(len(gts[s]))]


def recall_by(
    persons: Sequence[Person],
    flags: Mapping[str, Sequence[bool]],
    key: Callable[[Person], str | None],
    *,
    n_images: int,
    n_boot: int = 2000,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Recall per stratum for each set of flags (e.g. ``{"conf 0.40": ..., "any box": ...}``)."""
    keys = [key(p) for p in persons]
    strata = sorted({k for k in keys if k is not None}, key=_natural)
    img = np.array([p.image for p in persons], dtype=np.int64)
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, n_images, size=(n_boot, n_images))
    # how many times each image appears in each replicate: (n_boot, n_images)
    weight = np.zeros((n_boot, n_images), dtype=np.int64)
    np.add.at(weight, (np.arange(n_boot)[:, None], boot_idx), 1)

    rows: list[dict[str, Any]] = []
    for s in strata:
        sel = np.array([k == s for k in keys])
        row: dict[str, Any] = {
            "stratum": s,
            "people": int(sel.sum()),
            "photos": int(np.unique(img[sel]).size),
        }
        for name, f in flags.items():
            hit = np.asarray(f, dtype=bool)[sel]
            per_img_n = np.bincount(img[sel], minlength=n_images)
            per_img_hit = np.bincount(img[sel], weights=hit, minlength=n_images)
            n_b = weight @ per_img_n
            h_b = weight @ per_img_hit
            with np.errstate(invalid="ignore", divide="ignore"):
                r_b = h_b / n_b
            r_b = r_b[np.isfinite(r_b)]
            lo, hi = np.percentile(r_b, [2.5, 97.5]) if r_b.size else (math.nan, math.nan)
            row[name] = {
                "found": int(hit.sum()),
                "recall": float(hit.mean()),
                "ci": [float(lo), float(hi)],
            }
        rows.append(row)
    return rows


def _natural(s: str) -> tuple[float, str]:
    """Numeric strata in numeric order ("< 30 px" first), names alphabetically after."""
    if s.startswith("<"):
        return (-math.inf, s)
    m = re.search(r"\d+(\.\d+)?", s)
    return (float(m.group()) if m else math.inf, s)


def binner(edges: Sequence[float], unit: str) -> Callable[[float | None], str | None]:
    def f(v: float | None) -> str | None:
        if v is None:
            return None
        for lo, hi in itertools.pairwise(edges):
            if lo <= v < hi:
                if lo == 0:
                    return f"< {hi:g} {unit}"
                if math.isinf(hi):
                    return f"≥ {lo:g} {unit}"
                return f"{lo:g}-{hi:g} {unit}"
        return None

    return f


CRITERION_LABEL = {
    "found": "criterion: person found (IoU ≥ 0.1 or centre inside)",
    "iou50": "criterion: IoU ≥ 0.5",
}


def _pct(r: Mapping[str, Any]) -> str:
    lo, hi = r["ci"]
    return f"{100 * r['recall']:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"


def strata_markdown(res: Mapping[str, Any]) -> str:
    out = [f"# Recall by stratum · {res['tag']} · val", ""]
    titles = {
        "size": "person size, √(w·h) of the annotated box",
        "gsd": "ground resolution (DJI RelativeAltitude, flat ground; Canon has none)",
        "camera": "camera (EXIF model)",
    }
    for key, rows in res["val"].items():
        crit, dim = key.split("/")
        cols = [k for k in rows[0] if k not in ("stratum", "people", "photos")] if rows else []
        out += [
            f"## Val · {titles[dim]} · {CRITERION_LABEL[crit]}",
            "",
            "| Stratum | People | Photos | " + " | ".join(f"Recall % at {c}" for c in cols) + " |",
            "|---|---:|---:|" + "---:|" * len(cols),
        ]
        for r in rows:
            cells = " | ".join(f"{r[c]['found']}/{r['people']} · {_pct(r[c])}" for c in cols)
            out.append(f"| {r['stratum']} | {r['people']} | {r['photos']} | {cells} |")
        out.append("")
    out += [
        "## Test, described only (annotations + EXIF, no predictions used)",
        "",
        f"Expected recall ({CRITERION_LABEL['found']}, conf {res['conf']:.2f}) if only person "
        "size mattered: val recall per size bin, weighted by the group's sizes.",
        "",
        "| Camera | People | Median size | Share < 40 px | Median rel. altitude | "
        "Expected recall from size |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for d in res["test_described"]:
        alt = "n/a" if d["rel_alt_m_median"] is None else f"{d['rel_alt_m_median']:.0f} m"
        out.append(
            f"| {d['camera']} | {d['people']} | {d['size_px_median']:.0f} px | "
            f"{100 * d['share_below_40px']:.0f} % | {alt} | "
            f"{100 * d['recall_expected_from_size']:.1f} % |"
        )
    out.append("")
    return "\n".join(out)
