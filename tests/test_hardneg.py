from pathlib import Path

import numpy as np
from PIL import Image

from rescuelens.data.hardneg import (
    HardNegConfig,
    export_hard_negatives,
    false_alarms,
    hard_tiles,
    tile_origin,
)
from rescuelens.data.voc import Box
from rescuelens.eval.metrics import Pred

PERSON = Box(100, 100, 140, 160)


def test_only_boxes_on_nobody_above_the_floor_are_mined() -> None:
    preds = [
        Pred(PERSON, 0.9),  # the person
        Pred(Box(102, 101, 141, 160), 0.8),  # duplicate on the person: not a false alarm
        Pred(Box(500, 500, 530, 540), 0.5),  # rock
        Pred(Box(900, 900, 930, 940), 0.05),  # below the floor
    ]
    fas = false_alarms(preds, [PERSON], min_score=0.1)
    assert [p.score for p in fas] == [0.5]


def test_tiles_are_centred_and_clipped_to_the_photo() -> None:
    assert tile_origin(1000, 4000, 640) == 680
    assert tile_origin(100, 4000, 640) == 0
    assert tile_origin(3990, 4000, 640) == 3360


def test_one_tile_covers_nearby_false_alarms() -> None:
    cfg = HardNegConfig(max_per_photo=2)
    fas = [
        Pred(Box(990, 990, 1010, 1010), 0.9),
        Pred(Box(1100, 1000, 1120, 1020), 0.8),  # inside the first tile
        Pred(Box(3000, 2000, 3020, 2020), 0.7),
        Pred(Box(200, 2000, 220, 2020), 0.6),  # over the per-photo cap
    ]
    assert hard_tiles(fas, 4000, 3000, cfg) == [(680, 680), (2690, 1690)]


def test_export_keeps_true_labels_and_writes_a_dataset(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    Image.fromarray(np.zeros((1000, 1500, 3), np.uint8)).save(frames / "a.jpg")
    preds = {"a": [Pred(Box(300, 300, 320, 330), 0.6)]}
    gts = {"a": [Box(400, 400, 440, 460)]}  # a person next to the false alarm
    s = export_hard_negatives(preds, gts, frames, tmp_path, HardNegConfig())
    assert (s["hard_tiles"], s["hard_tiles_with_people"]) == (1, 1)
    (label,) = (tmp_path / "tiles_hn" / "labels" / "train").glob("*.txt")
    assert label.name == "a_hn_x0_y0.txt"  # centred at (310, 315), clipped to the corner
    assert label.read_text(encoding="utf-8").startswith("0 ")
    assert "tiles_hn" in (tmp_path / "tiles_hn" / "dataset.yaml").read_text(encoding="utf-8")
