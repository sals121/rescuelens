"""Hard negative mining: extra training tiles centred on the detector's false alarms.

The original tiles keep only 5 % of person-free tiles, picked at random, so most of the
ground the detector confuses with people (rocks, bushes, bags, shadows) is never shown to
it as background. Here the current detector runs over the full *train* photos; every box
that hits nobody (criterion "found") above a score floor gets a 640 px tile centred on it.
The tile keeps the true labels of any person inside, so it is a normal training tile.

Tiles go to a separate tree, ``<out>/tiles_hn/{images,labels}/train``, and a new dataset
file lists both trees for training, so the original dataset is never modified.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from rescuelens.data.tiling import boxes_in_tile
from rescuelens.data.voc import Box, to_yolo_line
from rescuelens.eval.bench import FP, label_image
from rescuelens.eval.metrics import Pred


@dataclass(frozen=True, slots=True)
class HardNegConfig:
    min_score: float = 0.10  # false alarms below this never reach an operator anyway
    tile: int = 640
    min_visible: float = 0.5
    max_per_photo: int = 8  # a few cluttered photos must not dominate the new tiles
    jpeg_quality: int = 95


def false_alarms(preds: list[Pred], gts: list[Box], min_score: float) -> list[Pred]:
    """Predictions that hit nobody (criterion "found"), highest score first."""
    ordered = sorted(preds, key=lambda q: q.score, reverse=True)
    labels = label_image(ordered, gts, "found")
    return [p for p, (_, lab, _) in zip(ordered, labels, strict=True)
            if lab == FP and p.score >= min_score]  # fmt: skip


def tile_origin(center: float, length: int, tile: int) -> int:
    """Start of a tile of size ``tile`` centred on ``center``, kept inside ``[0, length)``."""
    return int(max(0, min(length - tile, round(center - tile / 2))))


def hard_tiles(
    fas: list[Pred], width: int, height: int, cfg: HardNegConfig
) -> list[tuple[int, int]]:
    """Tile origins, one per false alarm not already inside an earlier hard tile."""
    origins: list[tuple[int, int]] = []
    for p in fas:
        cx, cy = (p.box.xmin + p.box.xmax) / 2, (p.box.ymin + p.box.ymax) / 2
        if any(x <= cx < x + cfg.tile and y <= cy < y + cfg.tile for x, y in origins):
            continue
        origins.append((tile_origin(cx, width, cfg.tile), tile_origin(cy, height, cfg.tile)))
        if len(origins) >= cfg.max_per_photo:
            break
    return origins


def export_hard_negatives(
    preds: dict[str, list[Pred]],
    gts: dict[str, list[Box]],
    frames_dir: Path,
    out_dir: Path,
    cfg: HardNegConfig,
) -> dict[str, Any]:
    """Write hard tiles + labels under ``out_dir/tiles_hn`` and a combined dataset file."""
    img_dir = out_dir / "tiles_hn" / "images" / "train"
    lbl_dir = out_dir / "tiles_hn" / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    n_fa = n_tiles = n_with_people = 0
    for stem in sorted(preds):
        fas = false_alarms(preds[stem], gts.get(stem, []), cfg.min_score)
        n_fa += len(fas)
        if not fas:
            continue
        with Image.open(frames_dir / f"{stem}.jpg") as im:
            img = im.convert("RGB")
        w, h = img.size
        for x, y in hard_tiles(fas, w, h, cfg):
            tboxes, _ = boxes_in_tile(gts.get(stem, []), x, y, cfg.tile, cfg.min_visible)
            tw, th = min(cfg.tile, w - x), min(cfg.tile, h - y)
            name = f"{stem}_hn_x{x}_y{y}"
            img.crop((x, y, x + tw, y + th)).save(img_dir / f"{name}.jpg", quality=cfg.jpeg_quality)
            (lbl_dir / f"{name}.txt").write_text(
                "".join(to_yolo_line(b, tw, th) + "\n" for b in tboxes), encoding="utf-8"
            )
            n_tiles += 1
            n_with_people += bool(tboxes)
    base = (out_dir / "tiles").resolve().as_posix()
    hn = (out_dir / "tiles_hn").resolve().as_posix()
    yaml = out_dir / "tiles_hn" / "dataset.yaml"
    yaml.write_text(
        "# HERIDAL tiles + hard negative tiles (train only). Val and test are unchanged.\n"
        f"train:\n  - {base}/images/train\n  - {hn}/images/train\n"
        f"val: {base}/images/val\n"
        f"test: {base}/images/test\n"
        "names:\n  0: person\n",
        encoding="utf-8",
    )
    summary = {
        "photos": len(preds),
        "false_alarms_mined": n_fa,
        "hard_tiles": n_tiles,
        "hard_tiles_with_people": n_with_people,
        "config": {k: getattr(cfg, k) for k in cfg.__slots__},
        "dataset": yaml.as_posix(),
    }
    (out_dir / "tiles_hn" / "summary.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8"
    )
    return summary
