"""Person detection on large drone frames: YOLO, optionally with SAHI slicing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rescuelens.logging import get_logger

log = get_logger(__name__)

PERSON_CLASS_NAMES = frozenset({"person", "human", "people", "pedestrian"})


def person_class_ids(names: dict[int, str]) -> set[int]:
    """Class ids that mean "person".

    A single-class checkpoint is a person detector by construction (Ultralytics renames the
    class to "item" when trained with ``single_cls=True``).
    """
    if len(names) == 1:
        return set(names)
    return {i for i, n in names.items() if n.lower() in PERSON_CLASS_NAMES}


def auto_device() -> str:
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True, slots=True)
class Detection:
    x: float  # top-left, pixels in the original frame
    y: float
    w: float
    h: float
    score: float

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2.0, self.y + self.h / 2.0


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    weights: Path
    confidence: float = 0.25
    device: str = "cpu"
    use_sahi: bool = True
    slice_size: int = 640
    overlap: float = 0.2
    full_image_size: int = 1280  # inference size when SAHI is off


class PersonDetector:
    """Thin wrapper over SAHI's Ultralytics adapter that only returns person boxes."""

    def __init__(self, config: DetectorConfig) -> None:
        from sahi import AutoDetectionModel  # heavy import, keep it lazy

        self.config = config
        self._model: Any = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=str(config.weights),
            confidence_threshold=config.confidence,
            device=config.device,
            image_size=config.full_image_size,
        )
        names: dict[int, str] = self._model.model.names
        self._person_ids = person_class_ids(names)
        if not self._person_ids:
            raise ValueError(f"weights {config.weights} have no person class: {names}")
        log.info(
            "detector_ready",
            weights=config.weights.name,
            device=config.device,
            sahi=config.use_sahi,
        )

    def detect(self, image: NDArray[np.uint8]) -> list[Detection]:
        """Detect people in an RGB HxWx3 image."""
        from sahi.predict import get_prediction, get_sliced_prediction

        if self.config.use_sahi:
            result = get_sliced_prediction(
                image,
                self._model,
                slice_height=self.config.slice_size,
                slice_width=self.config.slice_size,
                overlap_height_ratio=self.config.overlap,
                overlap_width_ratio=self.config.overlap,
                postprocess_type="NMS",
                postprocess_match_metric="IOS",
                postprocess_match_threshold=0.5,
                verbose=0,
            )
        else:
            result = get_prediction(image, self._model)

        detections: list[Detection] = []
        for pred in result.object_prediction_list:
            if pred.category.id not in self._person_ids:
                continue
            x, y, w, h = (float(v) for v in pred.bbox.to_xywh())
            detections.append(Detection(x, y, w, h, float(pred.score.value)))
        detections.sort(key=lambda d: d.score, reverse=True)
        return detections
