import math

import numpy as np
import pytest

from rescuelens.data.voc import Box
from rescuelens.eval.bench import (
    DUP,
    FP,
    TP,
    label_image,
    label_split,
    match_quality,
    operating_point,
    recall_fp_curve,
    threshold_for_recall,
    threshold_max_f1,
)
from rescuelens.eval.bench_run import Split, run_protocol
from rescuelens.eval.metrics import Pred
from rescuelens.ingest.camera import CameraInfo

PERSON = Box(100, 100, 140, 160)


def test_found_is_looser_than_iou50() -> None:
    shifted = Box(115, 120, 165, 190)  # same person, sloppy box: IoU ~0.29
    assert match_quality("iou50", shifted, PERSON) == 0
    assert match_quality("found", shifted, PERSON) > 0
    tiny = Box(118, 128, 122, 132)  # centre inside the person, IoU far below 0.1
    assert match_quality("found", tiny, PERSON) > 0
    assert match_quality("found", Box(300, 300, 340, 360), PERSON) == 0


def test_duplicates_are_not_false_alarms_when_found() -> None:
    preds = [Pred(PERSON, 0.9), Pred(Box(102, 101, 141, 160), 0.8), Pred(Box(0, 0, 9, 9), 0.7)]
    labels = [lab for _, lab, _ in label_image(preds, [PERSON], "found")]
    assert labels == [TP, DUP, FP]
    lab = label_split({"a": preds}, {"a": [PERSON]}, "found")
    op = operating_point(lab, 0.5, n_boot=50)
    assert (op["tp"], op["duplicates"], op["fp"]) == (1, 1, 1)
    assert op["fp_per_image"] == 1.0  # only the box on nothing
    assert op["precision"] == pytest.approx(2 / 3)  # two of three boxes are on a person

    strict = operating_point(label_split({"a": preds}, {"a": [PERSON]}, "iou50"), 0.5, n_boot=50)
    assert strict["fp_per_image"] == 2.0  # VOC: the duplicate is a false positive
    assert strict["precision"] == pytest.approx(1 / 3)


def test_labels_do_not_depend_on_the_threshold() -> None:
    # greedy matching by score: dropping low scores never changes higher-scored labels
    gts = {"a": [PERSON, Box(500, 500, 540, 560)], "b": [], "c": [Box(10, 10, 50, 70)]}
    preds = {
        "a": [
            Pred(PERSON, 0.3),
            Pred(Box(101, 99, 139, 161), 0.6),
            Pred(Box(505, 505, 545, 565), 0.2),
        ],
        "b": [Pred(Box(0, 0, 30, 30), 0.95)],
        "c": [Pred(Box(12, 12, 52, 72), 0.5)],
    }
    lab = label_split(preds, gts, "iou50")
    for conf in (0.1, 0.25, 0.4, 0.55, 0.9):
        cut = {k: [p for p in v if p.score >= conf] for k, v in preds.items()}
        again = label_split(cut, gts, "iou50")
        assert np.array_equal(lab.counts(conf), again.counts(0.0))


def test_threshold_for_recall_and_curve() -> None:
    gts = {f"i{i}": [Box(0, 0, 10, 10)] for i in range(4)}
    preds = {f"i{i}": [Pred(Box(0, 0, 10, 10), s)] for i, s in enumerate([0.9, 0.7, 0.5, 0.3])}
    lab = label_split(preds, gts, "found")
    assert threshold_for_recall(lab, 0.5) == 0.7
    assert threshold_for_recall(lab, 0.75) == 0.5
    assert threshold_for_recall(lab, 1.0) == 0.3
    gts["i4"] = [Box(50, 50, 60, 60)]  # nobody detects this one
    assert threshold_for_recall(label_split(preds, gts, "found"), 0.9) is None
    curve = recall_fp_curve(lab, step=0.1)
    assert [r["recall"] for r in curve][:3] == [1.0, 0.75, 0.75]  # conf 0.3, 0.4, 0.5
    assert curve[-1]["recall"] == 0.25


def test_max_f1_threshold_skips_low_score_noise() -> None:
    gts = {"a": [Box(0, 0, 10, 10)], "b": []}
    preds = {"a": [Pred(Box(0, 0, 10, 10), 0.8)], "b": [Pred(Box(0, 0, 5, 5), 0.2)]}
    assert threshold_max_f1(label_split(preds, gts, "iou50")) == 0.8


def test_bootstrap_interval_brackets_the_point_and_is_reproducible() -> None:
    rng = np.random.default_rng(1)
    gts = {f"i{i}": [Box(0, 0, 10, 10)] for i in range(60)}
    preds = {k: [Pred(Box(0, 0, 10, 10), float(rng.random()))] for k in gts}
    lab = label_split(preds, gts, "found")
    a = operating_point(lab, 0.5, n_boot=500, seed=3)
    b = operating_point(lab, 0.5, n_boot=500, seed=3)
    assert a == b
    lo, hi = a["recall_ci"]
    assert lo < a["recall"] < hi
    assert lo >= 0.0 and hi <= 1.0


def test_fp_per_km2_uses_footprints() -> None:
    gts: dict[str, list[Box]] = {"a": [], "b": []}
    preds = {"a": [Pred(Box(0, 0, 9, 9), 0.9)], "b": []}
    lab = label_split(preds, gts, "found")
    op = operating_point(lab, 0.5, area_km2=np.array([0.004, 0.006]), n_boot=20)
    assert op["fp_per_km2"] == pytest.approx(100.0)  # 1 false alarm over 0.01 km²
    assert op["recall"] is None  # no people: recall is undefined, not 0


def test_protocol_picks_thresholds_on_val_only() -> None:
    val_gts = {f"v{i}": [Box(0, 0, 10, 10)] for i in range(10)}
    val = Split(
        "val", {k: [Pred(Box(0, 0, 10, 10), 0.1 * (i + 1))] for i, k in enumerate(val_gts)}, val_gts
    )
    test_gts = {"t": [Box(0, 0, 10, 10)]}
    test = Split("test", {"t": [Pred(Box(0, 0, 10, 10), 0.05)]}, test_gts)
    res = run_protocol(val, test, targets=[0.9], n_boot=20)
    rule = next(r for r in res["thresholds"] if r["rule"].startswith("recall >= 0.90"))
    assert rule["conf"] == pytest.approx(0.2)  # 9 of 10 val people at conf >= 0.2
    assert rule["test"]["found"]["recall"] == 0.0  # test is scored, never tuned on


def test_camera_footprint() -> None:
    cam = CameraInfo("FC300X", 4000, 3000, 20.0, f35_assumed=False)
    diag = cam.ground_diagonal_m(50)
    assert diag == pytest.approx(50 * math.hypot(36, 24) / 20)
    w, h = diag * 0.8, diag * 0.6
    assert cam.footprint_m2(50) == pytest.approx(w * h)
    assert cam.gsd_m(50) == pytest.approx(w / 4000)


def test_fixture_matches_expected_scores() -> None:
    from pathlib import Path

    from typer.testing import CliRunner

    from rescuelens.cli import app

    d = Path(__file__).parent / "fixtures" / "bench_small"
    args = ["bench", "score", "--preds", str(d / "preds.json"), "--gt", str(d / "gt.json")]
    result = CliRunner().invoke(app, [*args, "--check", str(d / "expected.json")])
    assert result.exit_code == 0, result.output


def test_threshold_for_fp_budget() -> None:
    from rescuelens.eval.bench import threshold_for_fp_budget

    gts = {"a": [PERSON], "b": []}
    preds = {
        "a": [Pred(PERSON, 0.9), Pred(Box(0, 0, 9, 9), 0.5), Pred(Box(300, 0, 309, 9), 0.5)],
        "b": [Pred(Box(0, 0, 9, 9), 0.7), Pred(Box(50, 50, 60, 60), 0.2)],
    }
    lab = label_split(preds, gts, "found")
    assert threshold_for_fp_budget(lab, 0.5) == 0.7  # 1 false alarm over 2 photos
    assert threshold_for_fp_budget(lab, 1.0) == 0.7  # the 0.5 tie would add two
    assert threshold_for_fp_budget(lab, 1.5) == 0.5
    assert threshold_for_fp_budget(lab, 0.0) == 0.9


def test_paired_difference_cancels_photo_difficulty() -> None:
    from rescuelens.eval.bench import paired_difference

    # b finds exactly one more person than a, in one of 40 photos
    gts = {f"i{i}": [Box(0, 0, 10, 10)] for i in range(40)}
    pa = {k: [Pred(Box(0, 0, 10, 10), 0.9)] for k in list(gts)[:20]}
    pb = {k: [Pred(Box(0, 0, 10, 10), 0.9)] for k in list(gts)[:21]}
    la, lb = label_split(pa, gts, "found"), label_split(pb, gts, "found")
    d = paired_difference(la, 0.5, lb, 0.5, n_boot=500)
    assert d["difference"] == pytest.approx(1 / 40)
    lo, hi = d["ci"]
    assert lo >= 0 and hi > 0  # never negative: b is never worse on any photo
