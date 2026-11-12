"""Camera metadata from EXIF and the ground footprint of a nadir photo.

The footprint turns "false alarms per photo" into "false alarms per km² searched", which is
what a search coordinator plans with. EXIF has the lens but not the height above ground
(``GPSAltitude`` is above sea level), so the altitude is always an explicit assumption.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

# Diagonal of a 36 x 24 mm frame: the reference for 35 mm-equivalent focal lengths.
FULL_FRAME_DIAGONAL_MM = math.hypot(36.0, 24.0)

# 35 mm-equivalent focal length for cameras whose EXIF omits it.
KNOWN_F35_MM = {
    "Canon PowerShot S100": 24.0,
    "Canon PowerShot S110": 24.0,
}
DEFAULT_F35_MM = 24.0  # typical wide drone / compact lens; used when nothing else is known


@dataclass(frozen=True, slots=True)
class CameraInfo:
    model: str | None
    width_px: int
    height_px: int
    f35_mm: float
    f35_assumed: bool  # True when the focal length did not come from EXIF

    def ground_diagonal_m(self, agl_m: float) -> float:
        """Diagonal of the ground footprint of a nadir photo taken ``agl_m`` above ground."""
        return agl_m * FULL_FRAME_DIAGONAL_MM / self.f35_mm

    def footprint_m2(self, agl_m: float) -> float:
        d = self.ground_diagonal_m(agl_m)
        w, h = self.width_px, self.height_px
        return d * d * w * h / (w * w + h * h)

    def gsd_m(self, agl_m: float) -> float:
        """Ground sampling distance (metres per pixel), square pixels."""
        return self.ground_diagonal_m(agl_m) / math.hypot(self.width_px, self.height_px)


def _clean(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).replace("\x00", "").strip()
    return text or None


def read_camera(path: Path) -> CameraInfo:
    with Image.open(path) as im:
        exif = im.getexif()
        w, h = im.size
    model = _clean(exif.get(0x0110))
    f35 = exif.get_ifd(0x8769).get(0xA405)  # FocalLengthIn35mmFilm
    if f35:
        return CameraInfo(model, w, h, float(f35), f35_assumed=False)
    known = KNOWN_F35_MM.get(model or "")
    return CameraInfo(model, w, h, known or DEFAULT_F35_MM, f35_assumed=known is None)
