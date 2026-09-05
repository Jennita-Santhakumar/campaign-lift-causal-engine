"""Treatment effect estimation: ATE, CATE, and panel-data DiD."""

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor


class CausalEstimation:
    @staticmethod
    def estimate_ate(
        outcomes: np.ndarray, treatment: np.ndarray, weights: np.ndarray | None = None
    ) -> dict:
        """Estimate Average Treatment Effect: ATE = E[Y(1)] - E[Y(0)].

        `weights` optionally applies IPW (already normalized to sum to 1 within each arm
        by matching.ipw_weights) instead of a plain difference of means.
        """
        outcomes = np.asarray(outcomes, dtype=float)
        treatment = np.asarray(treatment)
        treated_mask, control_mask = treatment == 1, treatment == 0

        if weights is None:
            treated, control = outcomes[treated_mask], outcomes[control_mask]
            mean_t, mean_c = np.mean(treated), np.mean(control)
            se = np.sqrt(np.var(treated, ddof=1) / len(treated) + np.var(control, ddof=1) / len(control))
        else:
            w = np.asarray(weights)
            mean_t = np.sum(outcomes[treated_mask] * w[treated_mask])
            mean_c = np.sum(outcomes[control_mask] * w[control_mask])
            # Weighted variance approximation (Hmisc-style) for the SE of a weighted mean.
            var_t = np.sum(w[treated_mask] * (outcomes[treated_mask] - mean_t) ** 2)
            var_c = np.sum(w[control_mask] * (outcomes[control_mask] - mean_c) ** 2)
            se = np.sqrt(var_t + var_c)

        ate = mean_t - mean_c
        ci_lower, ci_upper = ate - 1.96 * se, ate + 1.96 * se
        t_stat = ate / se if se > 0 else 0.0
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

        return {
            "ate": float(ate),
            "se": float(se),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "p_value": float(p_value),
            "sample_size": int(len(outcomes)),
        }

    @staticmethod
    def estimate_cate(
        covariates: np.ndarray,
        outcomes: np.ndarray,
        treatment: np.ndarray,
        segment_labels: np.ndarray,
        segment_names: list[str],
    ) -> dict:
        """Conditional ATE by subgroup, via a T-learner: fit one outcome regressor on the
        treated arm and one on the control arm, then for every unit predict both potential
        outcomes and average the individual treatment effect within each segment.

        T-learner (two separate sklearn regressors) is used instead of a dedicated causal-ML
        library (e.g. econml/causalml) — those pull in a heavier, more fragile dependency
        chain on Windows/Python 3.11 for a marginal accuracy gain on a synthetic demo dataset;
        a T-learner is a standard, well-understood CATE estimator and keeps the dependency
        footprint identical to the rest of this project (pure sklearn). Documented as a
        trade-off in the README.
        """
        covariates = np.asarray(covariates)
        outcomes = np.asarray(outcomes, dtype=float)
        treatment = np.asarray(treatment)

        model_t = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
        model_c = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
        model_t.fit(covariates[treatment == 1], outcomes[treatment == 1])
        model_c.fit(covariates[treatment == 0], outcomes[treatment == 0])

        mu1 = model_t.predict(covariates)
        mu0 = model_c.predict(covariates)
        ite = mu1 - mu0  # individual treatment effect estimate for every unit

        by_segment = {}
        for label, name in enumerate(segment_names):
            mask = segment_labels == label
            if mask.sum() == 0:
                continue
            by_segment[name] = {
                "cate": float(np.mean(ite[mask])),
                "n": int(mask.sum()),
            }

        return {"overall_ite_mean": float(np.mean(ite)), "by_segment": by_segment}

    @staticmethod
    def did_estimate_panel(
        outcomes: np.ndarray, treatment: np.ndarray, period: np.ndarray
    ) -> dict:
        """Difference-in-Differences over real panel data (unit observed pre and post).

        DiD = (Y1_post - Y1_pre) - (Y0_post - Y0_pre)

        The source PRD's own sketch takes flat outcome/treatment/time_period arrays with no
        notion of *which unit* each pre/post observation belongs to, so it can only ever
        compare group means at two points in time — it can't tell a real panel (same units
        followed over time) from two unrelated cross-sections. This version still operates on
        flat arrays but assumes they are unit-aligned panel rows (each unit contributes one
        period=0 row and one period=1 row), matching how the synthetic dataset and DB schema
        are actually structured.
        """
        outcomes = np.asarray(outcomes, dtype=float)
        treatment = np.asarray(treatment)
        period = np.asarray(period)

        t_pre = outcomes[(treatment == 1) & (period == 0)]
        t_post = outcomes[(treatment == 1) & (period == 1)]
        c_pre = outcomes[(treatment == 0) & (period == 0)]
        c_post = outcomes[(treatment == 0) & (period == 1)]

        treated_delta = t_post.mean() - t_pre.mean()
        control_delta = c_post.mean() - c_pre.mean()
        did = treated_delta - control_delta

        se = np.sqrt(
            np.var(t_post, ddof=1) / len(t_post) + np.var(t_pre, ddof=1) / len(t_pre)
            + np.var(c_post, ddof=1) / len(c_post) + np.var(c_pre, ddof=1) / len(c_pre)
        )
        ci_lower, ci_upper = did - 1.96 * se, did + 1.96 * se
        t_stat = did / se if se > 0 else 0.0
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

        return {
            "ate": float(did),
            "se": float(se),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "p_value": float(p_value),
            "sample_size": int(len(outcomes)),
            "treated_delta": float(treated_delta),
            "control_delta": float(control_delta),
        }
