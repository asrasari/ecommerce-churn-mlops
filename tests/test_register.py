import pytest

from src.register import pick_best


class _FakeRun:
    def __init__(self, run_id, auc):
        self.info = type("I", (), {"run_id": run_id})()
        self.data = type("D", (), {"metrics": {"roc_auc": auc}})()


def test_pick_best_returns_highest_auc():
    runs = [_FakeRun("a", 0.81), _FakeRun("b", 0.94), _FakeRun("c", 0.88)]
    best = pick_best(runs, metric="roc_auc")
    assert best.info.run_id == "b"


def test_pick_best_empty_raises():
    with pytest.raises(ValueError):
        pick_best([], metric="roc_auc")
