"""Fine-tune a YOLO detector on HERIDAL tiles (Ultralytics)."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rescuelens.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class TrainConfig:
    data: Path  # tiles/dataset.yaml from `rescuelens data prepare`
    weights: Path  # starting checkpoint, e.g. weights/yolo26s.pt (COCO)
    out_weights: Path  # where the best checkpoint is copied
    project: Path  # Ultralytics run directory
    name: str = "heridal-yolo26s"
    epochs: int = 100
    patience: int = 30
    imgsz: int = 640  # = SAHI slice size used at inference
    batch: int = 32
    device: str = "0"
    workers: int = 8
    seed: int = 42
    # Nadir drone views have no "up": vertical flips are as plausible as horizontal ones.
    flipud: float = 0.5
    fliplr: float = 0.5


def train_detector(cfg: TrainConfig) -> dict[str, Any]:
    from ultralytics.models import YOLO  # heavy import, keep it lazy

    model = YOLO(str(cfg.weights))
    model.train(
        data=str(cfg.data),
        epochs=cfg.epochs,
        patience=cfg.patience,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        device=cfg.device,
        workers=cfg.workers,
        seed=cfg.seed,
        deterministic=True,
        project=str(cfg.project),
        name=cfg.name,
        exist_ok=False,
        single_cls=True,
        flipud=cfg.flipud,
        fliplr=cfg.fliplr,
        plots=True,
    )
    trainer = model.trainer
    if trainer is None:
        raise RuntimeError("Ultralytics did not return a trainer")
    best = Path(trainer.best)
    run_dir = Path(trainer.save_dir)

    cfg.out_weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, cfg.out_weights)

    # Tile-level metrics on the official HERIDAL test split (full-frame SAHI eval is separate).
    best_model = YOLO(str(best))
    m = best_model.val(
        data=str(cfg.data), split="test", imgsz=cfg.imgsz, batch=cfg.batch, device=cfg.device,
        project=str(cfg.project), name=f"{cfg.name}-test", plots=True,
    )  # fmt: skip
    summary: dict[str, Any] = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
        "run_dir": str(run_dir),
        "best": str(best),
        "test_tiles": {
            "precision": float(m.box.mp),
            "recall": float(m.box.mr),
            "mAP50": float(m.box.map50),
            "mAP50_95": float(m.box.map),
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    log.info("training_done", **summary["test_tiles"], weights=str(cfg.out_weights))
    return summary
