"""SIMULATED georeferencing.

HERIDAL images carry no usable GPS, so to place them on a map we synthesise a plausible
drone survey: a boustrophedon ("lawnmower") flight over the Ordesa valley (Pyrenees, Spain).

Everything produced here is FAKE location data and is flagged with ``simulated=True`` so
that the API, the UI and the README can say so explicitly. The imagery itself was captured
in Croatia / Bosnia-Herzegovina, not in the Pyrenees.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

# Rough centre of the survey area: Valle de Ordesa, Parque Nacional de Ordesa y Monte Perdido.
ORDESA_CENTER = (42.6435, -0.0420)
EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True, slots=True)
class CameraModel:
    """Nadir camera used to derive the ground footprint of each frame."""

    image_width_px: int = 4000
    image_height_px: int = 3000
    horizontal_fov_deg: float = 73.7  # typical 1" drone camera (approx. 24 mm equivalent)


@dataclass(frozen=True, slots=True)
class FramePose:
    lat: float
    lon: float
    alt_agl_m: float
    heading_deg: float  # 0 = north, clockwise
    gsd_m: float  # ground sampling distance, metres per pixel
    simulated: bool = True


def offset_latlon(lat: float, lon: float, east_m: float, north_m: float) -> tuple[float, float]:
    """Move a WGS84 point by a local east/north offset in metres (flat-earth approximation)."""
    dlat = math.degrees(north_m / EARTH_RADIUS_M)
    dlon = math.degrees(east_m / (EARTH_RADIUS_M * math.cos(math.radians(lat))))
    return lat + dlat, lon + dlon


def ground_sampling_distance(alt_agl_m: float, camera: CameraModel) -> float:
    footprint_w = 2.0 * alt_agl_m * math.tan(math.radians(camera.horizontal_fov_deg) / 2.0)
    return footprint_w / camera.image_width_px


def simulate_flight(
    n_frames: int,
    *,
    seed: int = 42,
    center: tuple[float, float] = ORDESA_CENTER,
    alt_agl_m: float = 50.0,
    along_track_overlap: float = 0.2,
    cross_track_overlap: float = 0.2,
    frames_per_line: int = 8,
    camera: CameraModel | None = None,
) -> list[FramePose]:
    """Generate ``n_frames`` poses along a lawnmower pattern centred on ``center``.

    Deterministic for a given seed; small GPS/heading jitter makes the track look real.
    """
    if n_frames <= 0:
        return []
    cam = camera or CameraModel()
    rng = random.Random(seed)  # noqa: S311 - simulation jitter, not crypto
    gsd = ground_sampling_distance(alt_agl_m, cam)
    # Camera's long side is across-track.
    footprint_across = cam.image_width_px * gsd
    footprint_along = cam.image_height_px * gsd
    step_along = footprint_along * (1.0 - along_track_overlap)
    step_across = footprint_across * (1.0 - cross_track_overlap)

    n_lines = math.ceil(n_frames / frames_per_line)
    # Survey lines run SW→NE following the valley axis.
    track_bearing = math.radians(65.0)
    along_vec = (math.sin(track_bearing), math.cos(track_bearing))  # (east, north)
    across_vec = (math.cos(track_bearing), -math.sin(track_bearing))
    half_len = (frames_per_line - 1) * step_along / 2.0
    half_wid = (n_lines - 1) * step_across / 2.0

    poses: list[FramePose] = []
    for i in range(n_frames):
        line, k = divmod(i, frames_per_line)
        forward = line % 2 == 0
        s = -half_len + (k if forward else frames_per_line - 1 - k) * step_along
        c = -half_wid + line * step_across
        east = s * along_vec[0] + c * across_vec[0] + rng.gauss(0.0, 1.5)
        north = s * along_vec[1] + c * across_vec[1] + rng.gauss(0.0, 1.5)
        lat, lon = offset_latlon(center[0], center[1], east, north)
        heading = (math.degrees(track_bearing) + (0.0 if forward else 180.0)) % 360.0
        poses.append(
            FramePose(
                lat=lat,
                lon=lon,
                alt_agl_m=alt_agl_m + rng.gauss(0.0, 0.8),
                heading_deg=(heading + rng.gauss(0.0, 2.0)) % 360.0,
                gsd_m=gsd,
            )
        )
    return poses


def pixel_to_latlon(
    pose: FramePose,
    px: float,
    py: float,
    image_width_px: int,
    image_height_px: int,
) -> tuple[float, float]:
    """Project an image pixel to ground coordinates for a nadir camera.

    Image "up" points along the drone heading; x grows to the right.
    """
    dx_m = (px - image_width_px / 2.0) * pose.gsd_m  # right of centre
    dy_m = (image_height_px / 2.0 - py) * pose.gsd_m  # ahead of centre
    h = math.radians(pose.heading_deg)
    east = dy_m * math.sin(h) + dx_m * math.cos(h)
    north = dy_m * math.cos(h) - dx_m * math.sin(h)
    return offset_latlon(pose.lat, pose.lon, east, north)
