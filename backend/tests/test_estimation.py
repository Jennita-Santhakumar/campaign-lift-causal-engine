import numpy as np

from app.internal.causal.estimation import CausalEstimation


def test_ate_recovers_known_effect():
    rng = np.random.default_rng(0)
    n = 5000
    treatment = rng.binomial(1, 0.5, n)
    outcomes = rng.normal(0, 1, n) + treatment * 2.0  # true effect = 2.0

    result = CausalEstimation.estimate_ate(outcomes, treatment)
    assert "ate" in result and "ci_lower" in result and "ci_upper" in result
    assert abs(result["ate"] - 2.0) < 0.2
    assert result["ci_lower"] < 2.0 < result["ci_upper"]


def test_did_estimate_panel_recovers_effect():
    rng = np.random.default_rng(1)
    n_units = 1000
    treatment = np.repeat(rng.binomial(1, 0.5, n_units), 2)
    period = np.tile([0, 1], n_units)
    baseline = np.repeat(rng.normal(10, 2, n_units), 2)
    true_did = 3.0
    outcomes = baseline + period * 1.0 + (treatment * period) * true_did + rng.normal(0, 1, n_units * 2)

    result = CausalEstimation.did_estimate_panel(outcomes, treatment, period)
    assert abs(result["ate"] - true_did) < 0.5


def test_cate_returns_segment_breakdown():
    rng = np.random.default_rng(2)
    n = 900
    X = rng.normal(size=(n, 2))
    treatment = rng.binomial(1, 0.5, n)
    outcomes = rng.normal(0, 1, n) + treatment * (1.0 + X[:, 0])
    segments = rng.integers(0, 3, n)

    result = CausalEstimation.estimate_cate(X, outcomes, treatment, segments, ["a", "b", "c"])
    assert set(result["by_segment"].keys()) <= {"a", "b", "c"}
    assert len(result["by_segment"]) > 0
