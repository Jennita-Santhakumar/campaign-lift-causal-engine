"""Matching and weighting: PSM (nearest-neighbor with caliper), IPW, stratification."""

import numpy as np


class PropensityScoreMatching:
    @staticmethod
    def match_neighbors(treated_ps: np.ndarray, control_ps: np.ndarray, caliper: float = 0.05) -> dict:
        """Nearest-neighbor matching with a caliper, one control per treated unit,
        without replacement (a matched control can't be reused for a second treated unit,
        which avoids a handful of "close" controls dominating the matched sample)."""
        matches: dict[int, int] = {}
        used_controls: set[int] = set()
        control_ps = np.asarray(control_ps)

        # Process treated units in order of how tightly they can be matched first,
        # so scarce close controls go to the treated units that need them most.
        order = np.argsort([np.min(np.abs(control_ps - t)) for t in treated_ps])

        for i in order:
            t_ps = treated_ps[i]
            diffs = np.abs(control_ps - t_ps)
            diffs_masked = diffs.copy()
            if used_controls:
                diffs_masked[list(used_controls)] = np.inf

            closest_idx = int(np.argmin(diffs_masked))
            if diffs_masked[closest_idx] <= caliper:
                matches[int(i)] = closest_idx
                used_controls.add(closest_idx)

        return matches

    @staticmethod
    def ipw_weights(propensity_scores: np.ndarray, treatment: np.ndarray) -> np.ndarray:
        """Inverse Probability Weighting, normalized to sum to 1 within each arm
        so the weighted average is still a proper mean (not scaled by sample size)."""
        propensity_scores = np.asarray(propensity_scores)
        treatment = np.asarray(treatment)
        weights = np.where(treatment == 1, 1.0 / propensity_scores, 1.0 / (1 - propensity_scores))

        for arm in (0, 1):
            mask = treatment == arm
            weights[mask] = weights[mask] / weights[mask].sum()

        return weights

    @staticmethod
    def stratify(propensity_scores: np.ndarray, n_strata: int = 5) -> np.ndarray:
        """Assign each unit to a propensity-score decile/quintile stratum (0-indexed)."""
        quantiles = np.linspace(0, 1, n_strata + 1)
        edges = np.quantile(propensity_scores, quantiles)
        edges[0], edges[-1] = -np.inf, np.inf
        return np.clip(np.digitize(propensity_scores, edges[1:-1]), 0, n_strata - 1)
