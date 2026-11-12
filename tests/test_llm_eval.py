import pytest

from rescuelens.data.voc import Box
from rescuelens.eval.llm_eval import Verdict, ablation, candidates, estimate_cost, summarize
from rescuelens.eval.metrics import Pred
from rescuelens.verify.cost import CostTracker

GT = {"img": [Box(0, 0, 10, 10), Box(100, 100, 110, 110)]}


def _v(box: list[float], score: float, person: bool | None, conf: float | None = 0.9) -> Verdict:
    status = "verified" if person is not None else "error"
    return Verdict("img", box, score, status, is_person=person, confidence=conf)


def test_claude_filter_removes_false_positive() -> None:
    verdicts = [
        _v([0, 0, 10, 10], 0.9, True),  # real person
        _v([50, 50, 60, 60], 0.8, False),  # rock, Claude rejects it
    ]
    m = ablation(verdicts, GT, conf=0.25)
    assert m["yolo"]["fp"] == 1 and m["yolo"]["tp"] == 1
    assert m["yolo+claude_filter"]["fp"] == 0 and m["yolo+claude_filter"]["tp"] == 1
    assert m["yolo+claude_filter"]["precision"] == 1.0


def test_claude_false_rejection_costs_recall() -> None:
    verdicts = [_v([0, 0, 10, 10], 0.9, False), _v([100, 100, 110, 110], 0.9, True)]
    m = ablation(verdicts, GT, conf=0.25)
    assert m["yolo"]["recall"] == 1.0
    assert m["yolo+claude_filter"]["recall"] == 0.5


def test_unverified_candidates_fail_open() -> None:
    m = ablation([_v([0, 0, 10, 10], 0.9, None, None)], GT, conf=0.25)
    assert m["yolo+claude_filter"]["tp"] == 1


def test_rescored_uses_claude_probability() -> None:
    # YOLO ranks the rock first; Claude's p(person) puts the real person first
    verdicts = [_v([50, 50, 60, 60], 0.95, False, 0.9), _v([0, 0, 10, 10], 0.3, True, 0.9)]
    m = ablation(verdicts, GT, conf=0.25)
    assert m["yolo+claude_rescored"]["ap50"] > m["yolo"]["ap50"]


def test_candidates_and_estimate() -> None:
    preds = {"a": [Pred(Box(0, 0, 1, 1), 0.1), Pred(Box(0, 0, 1, 1), 0.5)]}
    assert [p.score for p in candidates(preds, 0.25)["a"]] == [0.5]
    assert estimate_cost(10, "claude-opus-5-5") == pytest.approx(0.23)
    assert estimate_cost(10, "claude-haiku-5-5") < estimate_cost(10, "claude-opus-5-5")


def test_summary_reports_original_cost_for_cached_verdicts() -> None:
    fresh = Verdict(
        "img", [0, 0, 1, 1], 0.9, "verified", is_person=True, cost_usd=0.02, latency_s=4.0
    )
    cached = Verdict(
        "img", [5, 5, 6, 6], 0.8, "cached", is_person=False, cost_usd=0.02, latency_s=6.0
    )
    s = summarize([fresh, cached], CostTracker(max_cost_usd=1.0, spent_usd=0.02, calls=1))
    assert s["cost_usd"] == pytest.approx(0.04)  # what a cold run costs
    assert s["spent_this_run_usd"] == pytest.approx(0.02)
    assert s["said_person"] == 1 and s["verified"] == 2
