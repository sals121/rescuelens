import hashlib
import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from rescuelens.data.download import md5sum, safe_extract
from rescuelens.data.heridal import PrepareConfig, ground_truth, keep_negative, prepare_heridal
from rescuelens.data.tiling import boxes_in_tile, tile_grid, tile_origins
from rescuelens.data.voc import Box, parse_voc, to_yolo_line

VOC_TWO = """<?xml version="1.0" encoding="utf-8"?>
<annotation>
  <object><name>person</name><bndbox>
    <xmin>1765</xmin><xmax>1795</xmax><ymin>2326</ymin><ymax>2372</ymax></bndbox></object>
  <size><width>4000</width><height>3000</height><depth>3</depth></size>
  <object><name>person</name><bndbox>
    <xmin>2183</xmin><xmax>2225</xmax><ymin>1771</ymin><ymax>1803</ymax></bndbox></object>
  <object><name>car</name><bndbox>
    <xmin>10</xmin><xmax>20</xmax><ymin>10</ymin><ymax>20</ymax></bndbox></object>
</annotation>"""

# HERIDAL images without people: no <object>, and no <size> either.
VOC_EMPTY = '<?xml version="1.0"?><annotation><filename>x</filename></annotation>'


def _write(tmp: Path, name: str, text: str) -> Path:
    p = tmp / name
    p.write_text(text, encoding="utf-8")
    return p


# --- VOC ---------------------------------------------------------------------------------


def test_parse_voc_keeps_people_only(tmp_path: Path) -> None:
    boxes = parse_voc(_write(tmp_path, "a.xml", VOC_TWO))
    assert boxes == [Box(1765, 2326, 1795, 2372), Box(2183, 1771, 2225, 1803)]


def test_parse_voc_empty_annotation(tmp_path: Path) -> None:
    assert parse_voc(_write(tmp_path, "e.xml", VOC_EMPTY)) == []


def test_parse_voc_rejects_degenerate_box(tmp_path: Path) -> None:
    bad = VOC_TWO.replace("<xmax>1795</xmax>", "<xmax>1700</xmax>")
    with pytest.raises(ValueError, match="degenerate"):
        parse_voc(_write(tmp_path, "b.xml", bad))


def test_parse_voc_blocks_entity_expansion(tmp_path: Path) -> None:
    bomb = (
        '<?xml version="1.0"?><!DOCTYPE a [<!ENTITY x "xxxxxxxx">]>'
        "<annotation><filename>&x;</filename></annotation>"
    )
    with pytest.raises(Exception, match=r"(?i)entit"):
        parse_voc(_write(tmp_path, "bomb.xml", bomb))


def test_to_yolo_line() -> None:
    assert to_yolo_line(Box(100, 50, 200, 150), 400, 200) == "0 0.375000 0.500000 0.250000 0.500000"


def test_to_yolo_line_clips_to_image() -> None:
    assert to_yolo_line(Box(-10, 0, 10, 20), 100, 100) == "0 0.050000 0.100000 0.100000 0.200000"


# --- tiling ------------------------------------------------------------------------------


@pytest.mark.parametrize(("w", "h"), [(4000, 3000), (4000, 2250), (1300, 1000), (640, 640)])
def test_tile_grid_matches_sahi(w: int, h: int) -> None:
    from sahi.slicing import get_slice_bboxes

    sahi = get_slice_bboxes(h, w, 640, 640, overlap_height_ratio=0.2, overlap_width_ratio=0.2)
    assert [(b[0], b[1]) for b in sahi] == tile_grid(w, h, 640, 0.2)


def test_heridal_frame_has_48_tiles() -> None:
    assert len(tile_grid(4000, 3000, 640, 0.2)) == 48
    assert tile_origins(3000, 640, 0.2) == [0, 512, 1024, 1536, 2048, 2360]


def test_tile_origins_small_image() -> None:
    assert tile_origins(500, 640, 0.2) == [0]


def test_boxes_in_tile_shift_clip_and_drop() -> None:
    inside = Box(100, 100, 140, 160)
    mostly_in = Box(620, 10, 660, 50)  # 20/40 px inside -> 50% visible, kept and clipped
    barely_in = Box(630, 100, 670, 140)  # 10/40 px inside -> dropped
    outside = Box(700, 700, 720, 720)
    kept, dropped = boxes_in_tile([inside, mostly_in, barely_in, outside], 0, 0, 640)
    assert kept == [inside, Box(620, 10, 640, 50)]
    assert dropped == 1

    kept, _ = boxes_in_tile([Box(600, 600, 640, 640)], 512, 512, 640)
    assert kept == [Box(88, 88, 128, 128)]


def test_keep_negative_is_deterministic() -> None:
    a = [keep_negative(42, "img", x, 0, 0.3) for x in range(200)]
    assert a == [keep_negative(42, "img", x, 0, 0.3) for x in range(200)]
    assert 30 < sum(a) < 90
    assert keep_negative(1, "s", 0, 0, 1.0)


# --- download / extract ------------------------------------------------------------------


def test_md5sum(tmp_path: Path) -> None:
    p = tmp_path / "f.bin"
    p.write_bytes(b"rescuelens")
    assert md5sum(p) == hashlib.md5(b"rescuelens", usedforsecurity=False).hexdigest()


def test_safe_extract_skips_macos_metadata(tmp_path: Path) -> None:
    z = tmp_path / "a.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("ds/Annotations/a.xml", VOC_EMPTY)
        zf.writestr("__MACOSX/ds/._a.xml", "junk")
        zf.writestr("ds/.DS_Store", "junk")
    assert safe_extract(z, tmp_path / "out") == 1
    assert (tmp_path / "out" / "ds" / "Annotations" / "a.xml").exists()
    assert not (tmp_path / "out" / "__MACOSX").exists()


def test_safe_extract_rejects_path_traversal(tmp_path: Path) -> None:
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("../escape.txt", "x")
    with pytest.raises(ValueError, match="unsafe"):
        safe_extract(z, tmp_path / "out")


# --- end to end --------------------------------------------------------------------------


def _fake_voc(root: Path) -> None:
    """Three 1300x1000 frames: one person in train, none in val, two people in test."""
    (root / "JPEGImages").mkdir(parents=True)
    (root / "Annotations").mkdir()
    (root / "ImageSets" / "Main").mkdir(parents=True)
    obj = (
        "<object><name>person</name><bndbox><xmin>{}</xmin><xmax>{}</xmax>"
        "<ymin>{}</ymin><ymax>{}</ymax></bndbox></object>"
    )
    frames = {
        "train": ("train_A_0001", [obj.format(100, 140, 100, 160)]),
        "val": ("train_A_0002", []),
        "test": ("test_A_0001", [obj.format(700, 740, 500, 560), obj.format(1200, 1250, 900, 960)]),
    }
    for split, (stem, objs) in frames.items():
        Image.new("RGB", (1300, 1000), (60, 120, 60)).save(root / "JPEGImages" / f"{stem}.jpg")
        (root / "Annotations" / f"{stem}.xml").write_text(
            f"<annotation>{''.join(objs)}</annotation>", encoding="utf-8"
        )
        (root / "ImageSets" / "Main" / f"{split}.txt").write_text(stem + "\n", encoding="utf-8")


def test_prepare_heridal_end_to_end(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _fake_voc(raw / "heridal")
    out = tmp_path / "out"
    manifest = prepare_heridal(raw, out, PrepareConfig(neg_keep=0.0, workers=1))

    s = manifest["splits"]
    assert s["train"]["people"] == 1 and s["train"]["pos_tiles"] == 1
    assert s["train"]["neg_tiles"] == 0  # neg_keep=0 drops every empty train tile
    assert s["val"]["pos_tiles"] == 0 and s["val"]["neg_tiles"] == 0
    # 1300x1000 -> 3x2 tiles; test keeps all of them
    assert s["test"]["pos_tiles"] + s["test"]["neg_tiles"] == 6

    labels = (out / "tiles" / "labels" / "train" / "train_A_0001_x0_y0.txt").read_text()
    assert labels == "0 0.187500 0.203125 0.062500 0.093750\n"
    with Image.open(out / "tiles" / "images" / "train" / "train_A_0001_x0_y0.jpg") as im:
        assert im.size == (640, 640)

    assert (out / "test_full" / "images" / "test_A_0001.jpg").exists()
    gt = ground_truth(out)
    assert gt == {"test_A_0001": [Box(700, 500, 740, 560), Box(1200, 900, 1250, 960)]}

    yaml = (out / "tiles" / "dataset.yaml").read_text()
    assert "train: images/train" in yaml and "0: person" in yaml
    assert json.loads((out / "manifest.json").read_text())["config"]["tile"] == 640
