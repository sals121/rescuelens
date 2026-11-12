"""Pascal VOC annotations -> boxes -> YOLO label lines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from defusedxml.ElementTree import parse as parse_xml

PERSON_LABELS = frozenset({"person", "human", "people", "pedestrian"})


@dataclass(frozen=True, slots=True)
class Box:
    """Axis-aligned box in pixel coordinates (xmax/ymax exclusive)."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def w(self) -> float:
        return self.xmax - self.xmin

    @property
    def h(self) -> float:
        return self.ymax - self.ymin

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)


def parse_voc(path: Path) -> list[Box]:
    """Return the person boxes of a VOC XML file.

    HERIDAL files for images without people have no ``<object>`` (and often no ``<size>``),
    so an empty list is a valid result.
    """
    root = parse_xml(path).getroot()
    if root is None:
        raise ValueError(f"{path.name}: empty XML document")
    boxes: list[Box] = []
    for obj in root.iter("object"):
        if (obj.findtext("name") or "").strip().lower() not in PERSON_LABELS:
            continue
        bb = obj.find("bndbox")
        if bb is None:
            continue
        x0, y0 = float(bb.findtext("xmin", "nan")), float(bb.findtext("ymin", "nan"))
        x1, y1 = float(bb.findtext("xmax", "nan")), float(bb.findtext("ymax", "nan"))
        if not (x1 > x0 and y1 > y0):  # also rejects NaN
            raise ValueError(f"{path.name}: degenerate box {(x0, y0, x1, y1)}")
        boxes.append(Box(x0, y0, x1, y1))
    return boxes


def to_yolo_line(box: Box, img_w: int, img_h: int, cls: int = 0) -> str:
    """Format a box as a YOLO label line (class cx cy w h, normalised and clipped)."""
    x0, x1 = max(0.0, box.xmin), min(float(img_w), box.xmax)
    y0, y1 = max(0.0, box.ymin), min(float(img_h), box.ymax)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"box {box} lies outside a {img_w}x{img_h} image")
    cx, cy = (x0 + x1) / 2 / img_w, (y0 + y1) / 2 / img_h
    return f"{cls} {cx:.6f} {cy:.6f} {(x1 - x0) / img_w:.6f} {(y1 - y0) / img_h:.6f}"
