import pytest

from rescuelens.detect.fusion import fuse_views


def test_box_seen_in_every_view_keeps_its_score() -> None:
    a = [(100.0, 100.0, 140.0, 160.0, 0.8)]
    b = [(102.0, 101.0, 142.0, 161.0, 0.6)]
    (f,) = fuse_views([a, b])
    assert f[4] == pytest.approx(0.7)
    assert f[0] == pytest.approx((100 * 0.8 + 102 * 0.6) / 1.4)  # score-weighted mean


def test_box_seen_in_one_view_is_halved() -> None:
    a = [(100.0, 100.0, 140.0, 160.0, 0.8), (500.0, 500.0, 520.0, 530.0, 0.9)]
    b = [(101.0, 100.0, 141.0, 160.0, 0.8)]
    fused = fuse_views([a, b])
    assert [round(f[4], 3) for f in fused] == [0.8, 0.45]


def test_duplicates_within_one_view_do_not_inflate_the_score() -> None:
    a = [(0.0, 0.0, 40.0, 60.0, 0.9), (1.0, 0.0, 41.0, 60.0, 0.9)]
    (f,) = fuse_views([a, []])
    assert f[4] == pytest.approx(0.45)


def test_empty() -> None:
    assert fuse_views([]) == []
    assert fuse_views([[], []]) == []
