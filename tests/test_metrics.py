import math

import pytest

from rescuelens.data.voc import Box
from rescuelens.eval.metrics import Pred, average_precision, evaluate, iou, match_image, pr_curve

G1 = Box(0, 0, 10, 10)
G2 = Box(100, 100, 110, 110)


def test_iou() -> None:
    assert iou(G1, G1) == 1.0
    assert iou(G1, G2) == 0.0
    assert iou(Box(0, 0, 10, 10), Box(5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_match_image_each_gt_once_by_score() -> None:
    preds = [Pred(Box(0, 0, 10, 10), 0.6), Pred(Box(1, 0, 11, 10), 0.9)]
    # the higher-scoring prediction takes the ground truth; the other is a duplicate (FP)
    assert match_image(preds, [G1], 0.5) == [True, False]


def test_average_precision_perfect_and_empty() -> None:
    assert average_precision([(0.9, True), (0.8, True)], 2) == 1.0
    assert average_precision([], 2) == 0.0
    assert math.isnan(average_precision([], 0))


def test_average_precision_known_value() -> None:
    # TP, FP, TP with 2 GT: precision envelope 1.0 up to r=0.5, then 2/3 up to r=1.0
    assert average_precision([(0.9, True), (0.8, False), (0.7, True)], 2) == pytest.approx(
        0.5 * 1.0 + 0.5 * (2 / 3)
    )


def test_evaluate_counts() -> None:
    gts = {"a": [G1, G2], "b": []}
    preds = {
        "a": [Pred(G1, 0.9), Pred(Box(50, 50, 60, 60), 0.4)],  # TP + FP
        "b": [Pred(Box(0, 0, 5, 5), 0.2)],  # FP below conf
    }
    m = evaluate(preds, gts, conf=0.3)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5
    assert m["fp_per_image"] == 0.5
    assert m["ap50"] == pytest.approx(0.5)


def test_pr_curve_monotonic_recall() -> None:
    gts = {"a": [G1, G2]}
    preds = {"a": [Pred(G1, 0.9), Pred(G2, 0.5)]}
    assert pr_curve(preds, gts) == [(0.9, 1.0, 0.5), (0.5, 1.0, 1.0)]
