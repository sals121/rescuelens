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
    # Random zoom range is [1 - scale, 1 + scale]; Ultralytics' default is 0.5.
    scale: float = 0.5
    # Split scored after training. Val by default: test is reserved for final configurations.
    eval_split: str = "val"
    # Continue an interrupted run from <project>/<name>/weights/last.pt (saved every epoch).
    resume: bool = False


def train_detector(cfg: TrainConfig) -> dict[str, Any]:
    from ultralytics.models import YOLO  # heavy import, keep it lazy

    if cfg.resume:
        last = cfg.project / cfg.name / "weights" / "last.pt"
        if not last.exists():
            raise FileNotFoundError(f"nothing to resume: {last}")
        model = YOLO(str(last))
        model.train(resume=True)  # every other setting comes from the checkpoint
    else:
        model = YOLO(str(cfg.weights))
        _train_fresh(model, cfg)
    trainer = model.trainer
    if trainer is None:
        raise RuntimeError("Ultralytics did not return a trainer")
    return _finish(cfg, Path(trainer.best), Path(trainer.save_dir))


def _train_fresh(model: Any, cfg: TrainConfig) -> None:  # noqa: ANN401 - Ultralytics model
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
        flipud=cfg.flipud,
        fliplr=cfg.fliplr,
        scale=cfg.scale,
        plots=True,
    )


def _finish(cfg: TrainConfig, best: Path, run_dir: Path) -> dict[str, Any]:
    from ultralytics.models import YOLO

    cfg.out_weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, cfg.out_weights)

    # Tile-level metrics (full-frame SAHI evaluation is `rescuelens bench`).
    best_model = YOLO(str(best))
    m = best_model.val(
        data=str(cfg.data), split=cfg.eval_split, imgsz=cfg.imgsz, batch=cfg.batch,
        device=cfg.device, project=str(cfg.project), name=f"{cfg.name}-{cfg.eval_split}",
        plots=True,
    )  # fmt: skip
    summary: dict[str, Any] = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
        "run_dir": str(run_dir),
        "best": str(best),
        "eval_split": cfg.eval_split,
        "tiles": {
            "precision": float(m.box.mp),
            "recall": float(m.box.mr),
            "mAP50": float(m.box.map50),
            "mAP50_95": float(m.box.map),
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    log.info("training_done", **summary["tiles"], weights=str(cfg.out_weights))
    return summary
