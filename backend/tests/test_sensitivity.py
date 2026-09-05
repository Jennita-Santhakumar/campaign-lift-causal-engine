import numpy as np

from app.internal.causal.sensitivity import SensitivityAnalysis


def test_rosenbaum_p_upper_increases_with_gamma():
    rng = np.random.default_rng(0)
    n = 100
    treated = rng.normal(5, 1, n)
    control = rng.normal(3, 1, n)  # consistent positive effect

    p_at_1 = SensitivityAnalysis.rosenbaum_bounds(treated, control, gamma=1.0)["p_upper"]
    p_at_3 = SensitivityAnalysis.rosenbaum_bounds(treated, control, gamma=3.0)["p_upper"]
    assert p_at_3 >= p_at_1  # more hypothesized hidden bias -> harder to stay significant


def test_sensitivity_curve_monotonic_and_has_breakpoint_field():
    rng = np.random.default_rng(1)
    n = 60
    treated = rng.normal(5, 1, n)
    control = rng.normal(4.9, 1, n)  # weak effect, should lose significance at low gamma
    curve = SensitivityAnalysis.sensitivity_curve(treated, control)
    p_values = [p["p_upper"] for p in curve["points"]]
    assert p_values == sorted(p_values)
    assert "bias_robustness" in curve
