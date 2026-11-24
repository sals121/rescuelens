from pathlib import Path

import pytest

from rescuelens.data.voc import Box
from rescuelens.eval.bench import label_split
from rescuelens.eval.metrics import Pred
from rescuelens.eval.strata import PhotoMeta, binner, found_flags, people, recall_by
from rescuelens.ingest.camera import read_relative_altitude


def test_binner_labels_and_edges() -> None:
    b = binner((0, 30, 60, float("inf")), "px")
    assert [b(10), b(30), b(59.9), b(60), b(None)] == [
        "< 30 px",
        "30-60 px",
        "30-60 px",
        "≥ 60 px",
        None,
    ]


def test_recall_by_stratum() -> None:
    small, big = Box(0, 0, 20, 20), Box(100, 100, 180, 180)
    gts = {"a": [small, big], "b": [Box(0, 0, 25, 25)]}
    preds = {"a": [Pred(big, 0.9)], "b": [Pred(Box(0, 0, 25, 25), 0.3)]}
    lab = label_split(preds, gts, "found")
    meta = {s: PhotoMeta("cam", None, None) for s in gts}
    ps = people(lab, gts, meta)
    flags = {"0.4": found_flags(lab, gts, 0.4), "any": found_flags(lab, gts, 0.05)}
    rows = recall_by(
        ps, flags, lambda p: binner((0, 30, 1e9), "px")(p.size_px), n_images=2, n_boot=50
    )
    assert [r["stratum"] for r in rows] == ["< 30 px", "30-1e+09 px"]
    assert rows[0]["0.4"]["recall"] == 0.0 and rows[0]["any"]["recall"] == 0.5
    assert rows[1]["0.4"]["recall"] == 1.0


@pytest.mark.parametrize(
    ("xmp", "expected"),
    [
        (b'<x drone-dji:RelativeAltitude="+51.20" />', 51.2),
        (b"<drone-dji:RelativeAltitude>-3.5</drone-dji:RelativeAltitude>", -3.5),
        (b"no xmp here", None),
    ],
)
def test_relative_altitude(tmp_path: Path, xmp: bytes, expected: float | None) -> None:
    f = tmp_path / "x.jpg"
    f.write_bytes(b"\xff\xd8" + xmp + b"\xff\xd9")
    assert read_relative_altitude(f) == expected
