"""Propensity score estimation: P(treatment=1 | covariates)."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class PropensityScoreAnalysis:
    @staticmethod
    def estimate_propensity_score(X: np.ndarray, treatment: np.ndarray) -> np.ndarray:
        """Fit logistic regression and return P(T=1|X) for every unit.

        Covariates are standardized first since logistic regression's L2 penalty
        is scale-sensitive and our covariates span very different ranges
        (tenure in days vs. engagement score in [0,1]).
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_scaled, treatment)

        propensity_scores = model.predict_proba(X_scaled)[:, 1]
        # Clip away from exact 0/1 so downstream IPW (1/ps, 1/(1-ps)) never divides by zero.
        return np.clip(propensity_scores, 1e-4, 1 - 1e-4)

    @staticmethod
    def check_common_support(ps_treated: np.ndarray, ps_control: np.ndarray) -> dict:
        """Verify overlap assumption: the region where both groups have support."""
        common_min = max(np.min(ps_treated), np.min(ps_control))
        common_max = min(np.max(ps_treated), np.max(ps_control))

        in_support_treated = np.sum((ps_treated >= common_min) & (ps_treated <= common_max))
        in_support_control = np.sum((ps_control >= common_min) & (ps_control <= common_max))

        return {
            "common_min": float(common_min),
            "common_max": float(common_max),
            "treated_in_support": int(in_support_treated),
            "treated_total": int(len(ps_treated)),
            "control_in_support": int(in_support_control),
            "control_total": int(len(ps_control)),
        }
