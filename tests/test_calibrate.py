import numpy as np

from mailroom_triage.calibrate import choose_threshold


def test_threshold_meets_precision_target():
    probs = np.array([[0.99, 0.01]] * 80 + [[0.60, 0.40]] * 20)
    correct = np.array([1.0] * 80 + [0.0] * 20)
    chosen = choose_threshold(probs, correct, target_precision=0.90, min_coverage=0.20)
    assert chosen["met_target"] is True
    assert chosen["precision"] >= 0.90
    assert chosen["coverage"] >= 0.20


def test_threshold_falls_back_when_target_unreachable():
    probs = np.array([[0.90, 0.10]] * 10)
    correct = np.array([0.0] * 10)
    chosen = choose_threshold(probs, correct, target_precision=0.99, min_coverage=0.50)
    assert chosen["met_target"] is False
    assert "threshold" in chosen
