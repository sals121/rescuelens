"""HERIDAL (VOC) -> YOLO tiles for training, plus full-resolution test frames for evaluation.

Layout written under ``out_dir``::

    tiles/images/{train,val,test}/<stem>_x<X>_y<Y>.jpg   640 px native-resolution tiles
    tiles/labels/{train,val,test}/<stem>_x<X>_y<Y>.txt   YOLO labels (class 0 = person)
    tiles/dataset.yaml                                   Ultralytics dataset file
    test_full/images/<stem>.jpg                          untouched test frames (for SAHI eval)
    test_full/boxes.json                                 ground truth in pixels
    manifest.json                                        counts, parameters, provenance

Splits come from the archive's ``ImageSets/Main/{train,val,test}.txt`` (the official HERIDAL
test set and the Aalto mirror's train/val split), so they are reproducible by construction.
"""

from __future__ import annotations

import json
import os
import random
import shutil
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

from rescuelens.data.tiling import boxes_in_tile, tile_grid
from rescuelens.data.voc import Box, parse_voc, to_yolo_line
from rescuelens.ingest.exif import read_gps

SPLITS = ("train", "val", "test")


@dataclass(frozen=True, slots=True)
class PrepareConfig:
    tile: int = 640
    overlap: float = 0.2
    # Fraction of person-free tiles kept for train/val. Test keeps every tile.
    neg_keep: float = 0.05
    min_visible: float = 0.5
    seed: int = 42
    jpeg_quality: int = 95
    workers: int = field(default_factory=lambda: min(8, os.cpu_count() or 1))


@dataclass(frozen=True, slots=True)
class _Job:
    stem: str
    split: str
    image: Path
    annotation: Path
    out_dir: Path
    cfg: PrepareConfig


def find_voc_root(raw_dir: Path) -> Path:
    """Directory that contains ``ImageSets/Main`` (the archive nests one level)."""
    hits = sorted(p.parent.parent for p in raw_dir.rglob("ImageSets/Main"))
    if not hits:
        raise FileNotFoundError(f"no VOC dataset (ImageSets/Main) under {raw_dir}")
    return hits[0]


def read_split(voc_root: Path, split: str) -> list[str]:
    path = voc_root / "ImageSets" / "Main" / f"{split}.txt"
    return [s for s in path.read_text(encoding="utf-8").split() if s]


def keep_negative(seed: int, stem: str, x: int, y: int, p: float) -> bool:
    """Deterministic per-tile coin flip, independent of worker scheduling."""
    if p >= 1.0:
        return True
    return random.Random(f"{seed}:{stem}:{x}:{y}").random() < p  # noqa: S311 - sampling, not crypto


def _process(job: _Job) -> dict[str, Any]:
    cfg = job.cfg
    boxes = parse_voc(job.annotation)
    with Image.open(job.image) as im:
        img = im.convert("RGB")
    w, h = img.size
    img_dir = job.out_dir / "tiles" / "images" / job.split
    lbl_dir = job.out_dir / "tiles" / "labels" / job.split
    neg_p = 1.0 if job.split == "test" else cfg.neg_keep

    pos = neg = objs = partial = 0
    for x, y in tile_grid(w, h, cfg.tile, cfg.overlap):
        tboxes, dropped = boxes_in_tile(boxes, x, y, cfg.tile, cfg.min_visible)
        partial += dropped
        if not tboxes and not keep_negative(cfg.seed, job.stem, x, y, neg_p):
            continue
        name = f"{job.stem}_x{x}_y{y}"
        tw, th = min(cfg.tile, w - x), min(cfg.tile, h - y)
        img.crop((x, y, x + tw, y + th)).save(img_dir / f"{name}.jpg", quality=cfg.jpeg_quality)
        (lbl_dir / f"{name}.txt").write_text(
            "".join(to_yolo_line(b, tw, th) + "\n" for b in tboxes), encoding="utf-8"
        )
        pos += bool(tboxes)
        neg += not tboxes
        objs += len(tboxes)

    result: dict[str, Any] = {
        "stem": job.stem,
        "split": job.split,
        "size": [w, h],
        "boxes": [[b.xmin, b.ymin, b.xmax, b.ymax] for b in boxes],
        "pos_tiles": pos,
        "neg_tiles": neg,
        "tile_objects": objs,
        "partial_dropped": partial,
        "gps": read_gps(job.image) is not None,
    }
    if job.split == "test":
        dst = job.out_dir / "test_full" / "images" / job.image.name
        if not dst.exists():
            shutil.copy2(job.image, dst)
    return result


def _jobs(voc_root: Path, out_dir: Path, cfg: PrepareConfig) -> Iterable[_Job]:
    for split in SPLITS:
        for stem in read_split(voc_root, split):
            image = voc_root / "JPEGImages" / f"{stem}.jpg"
            ann = voc_root / "Annotations" / f"{stem}.xml"
            if not image.exists() or not ann.exists():
                raise FileNotFoundError(f"{split}/{stem}: missing image or annotation")
            yield _Job(stem, split, image, ann, out_dir, cfg)


def prepare_heridal(raw_dir: Path, out_dir: Path, cfg: PrepareConfig) -> dict[str, Any]:
    voc_root = find_voc_root(raw_dir)
    for split in SPLITS:
        for kind in ("images", "labels"):
            d = out_dir / "tiles" / kind / split
            if d.exists():
                shutil.rmtree(d)  # tiles are fully derived; rebuild from scratch
            d.mkdir(parents=True)
    (out_dir / "test_full" / "images").mkdir(parents=True, exist_ok=True)

    jobs = list(_jobs(voc_root, out_dir, cfg))
    results: list[dict[str, Any]] = []
    if cfg.workers > 1:
        with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
            results.extend(tqdm(pool.map(_process, jobs, chunksize=4), total=len(jobs)))
    else:
        results.extend(tqdm(map(_process, jobs), total=len(jobs)))

    test = {
        r["stem"]: {"size": r["size"], "boxes": r["boxes"]} for r in results if r["split"] == "test"
    }
    (out_dir / "test_full" / "boxes.json").write_text(json.dumps(test, indent=1), encoding="utf-8")

    tiles_dir = (out_dir / "tiles").resolve()
    (tiles_dir / "dataset.yaml").write_text(
        f"# Generated by `rescuelens data prepare`. HERIDAL, CC BY 3.0.\n"
        f"path: {tiles_dir.as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: person\n",
        encoding="utf-8",
    )

    manifest = {
        "dataset": "HERIDAL (Aalto VOC mirror, Zenodo 5662351)",
        "license": "CC BY 3.0",
        "config": {k: v for k, v in asdict(cfg).items() if k != "workers"},
        "splits": {s: _summary([r for r in results if r["split"] == s]) for s in SPLITS},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    widths = sorted(b[2] - b[0] for r in rows for b in r["boxes"])
    return {
        "images": len(rows),
        "images_with_people": sum(1 for r in rows if r["boxes"]),
        "people": sum(len(r["boxes"]) for r in rows),
        "images_with_gps": sum(1 for r in rows if r["gps"]),
        "pos_tiles": sum(r["pos_tiles"] for r in rows),
        "neg_tiles": sum(r["neg_tiles"] for r in rows),
        "tile_objects": sum(r["tile_objects"] for r in rows),
        "partial_boxes_dropped": sum(r["partial_dropped"] for r in rows),
        "person_width_px_median": widths[len(widths) // 2] if widths else None,
    }


def export_full_split(raw_dir: Path, out_dir: Path, split: str) -> int:
    """Full-resolution frames + ``boxes.json`` for one split, under ``<split>_full/``.

    ``prepare_heridal`` already does this for test. Val needs it too: operating thresholds
    are chosen on val frames, never on test. Images are hard-linked when possible.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    voc_root = find_voc_root(raw_dir)
    img_dir = out_dir / f"{split}_full" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    boxes: dict[str, dict[str, Any]] = {}
    for stem in read_split(voc_root, split):
        src = voc_root / "JPEGImages" / f"{stem}.jpg"
        dst = img_dir / src.name
        if not dst.exists():
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        with Image.open(src) as im:
            size = list(im.size)
        b = parse_voc(voc_root / "Annotations" / f"{stem}.xml")
        boxes[stem] = {"size": size, "boxes": [[x.xmin, x.ymin, x.xmax, x.ymax] for x in b]}
    (img_dir.parent / "boxes.json").write_text(json.dumps(boxes, indent=1), encoding="utf-8")
    return len(boxes)


def ground_truth(out_dir: Path, split: str = "test") -> dict[str, list[Box]]:
    """Full-resolution boxes of a split (``prepare_heridal`` / :func:`export_full_split`)."""
    path = out_dir / f"{split}_full" / "boxes.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return {stem: [Box(*b) for b in v["boxes"]] for stem, v in data.items()}
