"""Covariate balance assessment via standardized mean difference (SMD).

Not sketched in the source PRD (only named in its architecture diagram) — implemented
here using the standard Rosenbaum & Rubin (1985) pooled-variance SMD formula. A common
rule of thumb (used below) is that |SMD| < 0.1 indicates adequate balance.
"""

import numpy as np


def standardized_mean_diff(x_treated: np.ndarray, x_control: np.ndarray) -> float:
    mean_t, mean_c = np.mean(x_treated), np.mean(x_control)
    var_t, var_c = np.var(x_treated, ddof=1), np.var(x_control, ddof=1)
    pooled_sd = np.sqrt((var_t + var_c) / 2)
    if pooled_sd == 0:
        return 0.0
    return float((mean_t - mean_c) / pooled_sd)


def balance_report(
    covariates: np.ndarray,
    treatment: np.ndarray,
    covariate_names: list[str],
    matched_treated_idx: np.ndarray | None = None,
    matched_control_idx: np.ndarray | None = None,
) -> dict:
    """SMD per covariate before matching, and after matching if matched indices given."""
    treatment = np.asarray(treatment)
    before = {}
    for j, name in enumerate(covariate_names):
        before[name] = standardized_mean_diff(
            covariates[treatment == 1, j], covariates[treatment == 0, j]
        )

    report = {
        "before": before,
        "max_abs_smd_before": max(abs(v) for v in before.values()),
    }

    if matched_treated_idx is not None and matched_control_idx is not None and len(matched_treated_idx) > 0:
        after = {}
        for j, name in enumerate(covariate_names):
            after[name] = standardized_mean_diff(
                covariates[matched_treated_idx, j], covariates[matched_control_idx, j]
            )
        report["after"] = after
        report["max_abs_smd_after"] = max(abs(v) for v in after.values())
        report["balanced"] = report["max_abs_smd_after"] < 0.1

    return report
