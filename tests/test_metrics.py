import numpy as np
import pytest

import config as C
from metrics import (accuracy, bayes_decide, bootstrap_ci, choose_tau, cus, macro_f1,
                     risk_coverage, summarize)

U3 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.9], [0.0, 0.9, 1.0]])


def test_cus_hand_computed():
    assert cus(np.array([0, 1, 2]), np.array([0, 2, 2]), U3) == pytest.approx((1 + 0.9 + 1) / 3)


def test_bayes_rule_prefers_expected_utility_over_argmax():
    proba = np.array([[0.40, 0.35, 0.25]])
    # EU(k) = sum_j p_j U[j,k] = [0.40, 0.575, 0.565]
    assert proba.argmax(1)[0] == 0
    assert bayes_decide(proba, U3)[0] == 1


def test_macro_f1_ignores_classes_without_support():
    per, m = macro_f1(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), 3)
    assert per[0] == pytest.approx(2 / 3) and per[1] == pytest.approx(0.8)
    assert m == pytest.approx((2 / 3 + 0.8) / 2)


def test_bootstrap_ci_on_perfect_predictions():
    y = np.arange(20) % 3
    ci = bootstrap_ci(accuracy, y, np.stack([y, y]), n_boot=50)
    assert ci == {"mean": 1.0, "lo": 1.0, "hi": 1.0}


def test_risk_coverage_and_tau():
    y = np.array([0, 1, 0, 1])
    proba = np.array([[0.95, 0.05], [0.9, 0.1], [0.6, 0.4], [0.2, 0.8]])
    rows = {r["tau"]: r for r in risk_coverage(y, proba)}
    assert rows[0.0]["coverage"] == 1.0 and rows[0.0]["accuracy"] == 0.75
    assert rows[0.95]["coverage"] == 0.25 and rows[0.95]["accuracy"] == 1.0
    assert choose_tau(list(rows.values()), target_acc=0.99, min_coverage=0.9) == 0.0


def test_summarize_keys():
    s = summarize(np.array([0, 1, 2]), np.array([0, 1, 1]), U3)
    assert set(s) == {"accuracy", "macro_f1", "cus", "per_class_f1", "confusion"}
    assert np.array(s["confusion"]).sum() == 3


def test_summarize_values_match_components():
    y, p = np.array([0, 1, 2]), np.array([0, 1, 1])
    s = summarize(y, p, U3)
    per, m = macro_f1(y, p, 3)
    assert s["accuracy"] == pytest.approx(2 / 3)
    assert s["macro_f1"] == pytest.approx(m)
    assert s["cus"] == pytest.approx(cus(y, p, U3))
    assert list(s["per_class_f1"]) == C.CLASSES[:3]
    assert list(s["per_class_f1"].values()) == pytest.approx(per.round(4).tolist())


# Asymmetric utility: a transposed U must change both CUS and the Bayes decision.
Ua = np.array([[1.0, 0.5], [0.0, 1.0]])


def test_cus_is_not_symmetric_in_U():
    assert cus(np.array([0]), np.array([1]), Ua) == pytest.approx(0.5)
    assert cus(np.array([1]), np.array([0]), Ua) == pytest.approx(0.0)


def test_bayes_uses_U_not_its_transpose():
    proba = np.array([[0.6, 0.4]])
    # proba @ Ua = [0.6, 0.7] -> 1 ; proba @ Ua.T = [0.8, 0.4] would give 0
    assert bayes_decide(proba, Ua)[0] == 1


def test_bootstrap_ci_on_imperfect_predictions():
    y = np.arange(30) % 3
    p = np.where(np.arange(30) < 10, (y + 1) % 3, y)  # 20/30 correct
    ci = bootstrap_ci(accuracy, y, p, n_boot=200)
    assert ci["lo"] < ci["mean"] < ci["hi"]
    assert 0 < ci["lo"]
    assert ci["mean"] == pytest.approx(2 / 3)


def test_bootstrap_ci_1d_matches_2d():
    y = np.arange(30) % 3
    p = np.where(np.arange(30) < 10, (y + 1) % 3, y)
    assert bootstrap_ci(accuracy, y, p, n_boot=50) == bootstrap_ci(accuracy, y, p[None], n_boot=50)


def test_risk_coverage_threshold_above_all_confidences():
    y = np.array([0, 1])
    proba = np.array([[0.9, 0.1], [0.2, 0.8]])
    (row,) = risk_coverage(y, proba, thresholds=[1.01])
    assert row["coverage"] == 0.0 and row["accuracy"] is None


def test_choose_tau_returns_zero_when_no_row_qualifies():
    rows = [{"tau": 0.3, "coverage": 1.0, "accuracy": 0.5},
            {"tau": 0.6, "coverage": 0.2, "accuracy": 0.99}]
    assert choose_tau(rows, target_acc=0.9, min_coverage=0.5) == 0.0
