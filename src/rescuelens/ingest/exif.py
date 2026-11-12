"""Read GPS position and capture metadata from image EXIF."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import ExifTags, Image

from rescuelens.logging import get_logger

log = get_logger(__name__)

_GPS = ExifTags.GPS


@dataclass(frozen=True, slots=True)
class GpsFix:
    lat: float
    lon: float
    alt_m: float | None


# EXIF payloads are untyped by nature (Pillow returns heterogeneous values), hence Any.
def _to_float(value: Any) -> float:  # noqa: ANN401
    """EXIF rationals arrive as IFDRational, (num, den) tuples, or plain numbers."""
    if isinstance(value, tuple) and len(value) == 2:
        num, den = value
        result = float(num) / float(den)
    else:
        result = float(value)
    if not math.isfinite(result):  # IFDRational with a zero denominator yields NaN
        raise ValueError(f"non-finite EXIF rational: {value!r}")
    return result


def dms_to_decimal(dms: Any, ref: str) -> float:  # noqa: ANN401
    """Convert EXIF (degrees, minutes, seconds) + hemisphere ref to signed decimal degrees."""
    degrees, minutes, seconds = (_to_float(v) for v in dms)
    decimal = degrees + minutes / 60.0 + seconds / 3600.0
    return -decimal if ref.upper() in {"S", "W"} else decimal


def parse_gps_ifd(gps: dict[int, Any]) -> GpsFix | None:
    """Parse a raw GPS IFD mapping. Returns None if latitude/longitude are missing or invalid."""
    try:
        lat = dms_to_decimal(gps[_GPS.GPSLatitude], str(gps[_GPS.GPSLatitudeRef]))
        lon = dms_to_decimal(gps[_GPS.GPSLongitude], str(gps[_GPS.GPSLongitudeRef]))
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None

    alt: float | None = None
    if _GPS.GPSAltitude in gps:
        try:
            alt = _to_float(gps[_GPS.GPSAltitude])
            # GPSAltitudeRef == 1 means below sea level.
            if gps.get(_GPS.GPSAltitudeRef) in (1, b"\x01"):
                alt = -alt
        except (TypeError, ValueError, ZeroDivisionError):
            alt = None
    return GpsFix(lat=lat, lon=lon, alt_m=alt)


def read_gps(path: Path) -> GpsFix | None:
    """Return the GPS fix stored in the image's EXIF, or None if absent/unreadable."""
    try:
        with Image.open(path) as img:
            gps_ifd = dict(img.getexif().get_ifd(ExifTags.IFD.GPSInfo))
    except OSError as exc:
        log.warning("exif_read_failed", path=str(path), error=str(exc))
        return None
    return parse_gps_ifd(gps_ifd) if gps_ifd else None
