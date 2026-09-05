import numpy as np

from app.internal.causal.balance import balance_report, standardized_mean_diff


def test_smd_zero_for_identical_groups():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert standardized_mean_diff(x, x) == 0.0


def test_smd_positive_when_treated_higher():
    treated = np.array([5.0, 6.0, 7.0])
    control = np.array([1.0, 2.0, 3.0])
    assert standardized_mean_diff(treated, control) > 0


def test_balance_report_improves_after_matching():
    rng = np.random.default_rng(0)
    n = 200
    covariates = rng.normal(size=(n, 2))
    treatment = np.zeros(n, dtype=int)
    treatment[:100] = 1
    covariates[:100, 0] += 2.0  # imbalanced covariate before matching

    matched_treated = np.arange(0, 50)
    matched_control = np.arange(100, 150)
    # force matched subset to be balanced
    covariates[matched_control, 0] = covariates[matched_treated, 0]

    report = balance_report(covariates, treatment, ["x0", "x1"], matched_treated, matched_control)
    assert report["max_abs_smd_before"] > report["max_abs_smd_after"]
