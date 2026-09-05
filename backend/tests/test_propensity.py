import numpy as np

from app.internal.causal.propensity import PropensityScoreAnalysis


def _fake_data(n=500, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    logit = 0.8 * X[:, 0] + 0.3 * X[:, 1]
    treatment = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return X, treatment


def test_propensity_scores_in_unit_interval():
    X, treatment = _fake_data()
    ps = PropensityScoreAnalysis.estimate_propensity_score(X, treatment)
    assert ps.min() >= 0.0
    assert ps.max() <= 1.0
    assert len(ps) == len(treatment)


def test_common_support_overlap():
    X, treatment = _fake_data()
    ps = PropensityScoreAnalysis.estimate_propensity_score(X, treatment)
    result = PropensityScoreAnalysis.check_common_support(ps[treatment == 1], ps[treatment == 0])
    assert result["common_min"] <= result["common_max"]
    assert result["treated_in_support"] <= result["treated_total"]
