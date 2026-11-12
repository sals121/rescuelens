import itertools
import math

import pytest

from rescuelens.ingest.georef_simulated import (
    ORDESA_CENTER,
    CameraModel,
    FramePose,
    ground_sampling_distance,
    offset_latlon,
    pixel_to_latlon,
    simulate_flight,
)


def _haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * 6_371_000 * math.asin(math.sqrt(h))


def test_flight_is_deterministic_and_flagged() -> None:
    a = simulate_flight(20, seed=7)
    b = simulate_flight(20, seed=7)
    assert a == b
    assert len(a) == 20
    assert all(p.simulated for p in a)
    assert simulate_flight(20, seed=8) != a


def test_flight_stays_near_ordesa() -> None:
    for pose in simulate_flight(120):
        assert _haversine_m((pose.lat, pose.lon), ORDESA_CENTER) < 2_000


def test_consecutive_frames_are_spaced_like_a_survey() -> None:
    poses = simulate_flight(8, frames_per_line=8, alt_agl_m=50.0)
    gaps = [_haversine_m((p.lat, p.lon), (q.lat, q.lon)) for p, q in itertools.pairwise(poses)]
    # 50 m AGL → ~56 m along-track footprint, 20 % overlap → ~45 m steps (± GPS jitter).
    assert all(35 < g < 55 for g in gaps)


def test_empty_flight() -> None:
    assert simulate_flight(0) == []


def test_gsd_reasonable() -> None:
    gsd = ground_sampling_distance(50.0, CameraModel())
    assert 0.015 < gsd < 0.025  # ~2 cm/px, a person is ~25-90 px across


def test_offset_latlon_roundtrip_distance() -> None:
    moved = offset_latlon(*ORDESA_CENTER, east_m=300.0, north_m=400.0)
    assert _haversine_m(ORDESA_CENTER, moved) == pytest.approx(500.0, rel=1e-3)


@pytest.mark.parametrize(
    ("heading", "px", "py", "expect_east", "expect_north"),
    [
        (0.0, 2000, 0, 0.0, 30.0),  # top edge, facing north → north
        (90.0, 2000, 0, 30.0, 0.0),  # top edge, facing east → east
        (0.0, 4000, 1500, 40.0, 0.0),  # right edge, facing north → east
        (0.0, 2000, 1500, 0.0, 0.0),  # centre → drone position
    ],
)
def test_pixel_to_latlon_orientation(
    heading: float, px: float, py: float, expect_east: float, expect_north: float
) -> None:
    pose = FramePose(lat=42.0, lon=0.0, alt_agl_m=50.0, heading_deg=heading, gsd_m=0.02)
    got = pixel_to_latlon(pose, px, py, 4000, 3000)
    expected = offset_latlon(42.0, 0.0, expect_east, expect_north)
    assert _haversine_m(got, expected) < 0.01
