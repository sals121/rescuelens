"""Discover images in a folder and attach a position to each one."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image

from rescuelens.ingest.exif import read_gps
from rescuelens.ingest.georef_simulated import FramePose, simulate_flight
from rescuelens.logging import get_logger

log = get_logger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
GeoMode = Literal["auto", "exif", "simulated"]


@dataclass(frozen=True, slots=True)
class SourceFrame:
    path: Path
    width: int
    height: int
    lat: float
    lon: float
    alt_m: float | None
    geo_source: Literal["exif", "simulated"]
    pose: FramePose | None  # only for simulated frames (heading + GSD known)


def list_images(folder: Path, limit: int | None = None) -> list[Path]:
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    return paths[:limit] if limit else paths


def load_frames(
    paths: list[Path], *, geo_mode: GeoMode = "auto", seed: int = 42
) -> list[SourceFrame]:
    """Read size + GPS for each image.

    ``auto`` uses real EXIF GPS when *every* frame has it; otherwise the whole batch gets a
    SIMULATED flight path (mixing real and fake positions on one map would be misleading).
    """
    sizes: list[tuple[int, int]] = []
    fixes = []
    for p in paths:
        with Image.open(p) as img:
            sizes.append(img.size)
        fixes.append(read_gps(p))

    have_all_gps = all(f is not None for f in fixes)
    if geo_mode == "exif" and not have_all_gps:
        missing = [p.name for p, f in zip(paths, fixes, strict=True) if f is None]
        raise ValueError(f"geo_mode=exif but {len(missing)} images lack GPS: {missing[:5]}")
    use_exif = geo_mode == "exif" or (geo_mode == "auto" and have_all_gps)

    if not use_exif:
        log.warning("using_simulated_georeferencing", n_frames=len(paths))
        poses = simulate_flight(len(paths), seed=seed)
        return [
            SourceFrame(p, w, h, pose.lat, pose.lon, pose.alt_agl_m, "simulated", pose)
            for p, (w, h), pose in zip(paths, sizes, poses, strict=True)
        ]

    frames: list[SourceFrame] = []
    for p, (w, h), fix in zip(paths, sizes, fixes, strict=True):
        assert fix is not None
        frames.append(SourceFrame(p, w, h, fix.lat, fix.lon, fix.alt_m, "exif", None))
    return frames
