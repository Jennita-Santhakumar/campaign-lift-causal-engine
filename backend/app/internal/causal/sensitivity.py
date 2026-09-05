"""Sensitivity analysis: Rosenbaum bounds for robustness to unobserved (hidden) confounding."""

import numpy as np
from scipy import stats


class SensitivityAnalysis:
    @staticmethod
    def rosenbaum_bounds(
        treated_outcomes: np.ndarray, control_outcomes: np.ndarray, gamma: float = 1.25
    ) -> dict:
        """Rosenbaum sensitivity bounds on matched pairs.

        `treated_outcomes`/`control_outcomes` must be matched pairs (same length, index i of
        each array is one matched pair) — produced by PropensityScoreMatching.match_neighbors,
        not arbitrary unmatched arrays (the source PRD's sketch takes a dict but never shows
        where the pairs come from; here they're wired to real matches).

        gamma: hypothesized magnitude of hidden bias (an unmeasured confounder that changes
        the odds of treatment by a factor of gamma). gamma=1 means no hidden bias.
        """
        diffs = np.asarray(treated_outcomes) - np.asarray(control_outcomes)
        diffs = diffs[diffs != 0]  # ties carry no information for the sign test

        n_plus = int(np.sum(diffs > 0))
        n = len(diffs)

        if n == 0:
            return {"gamma": gamma, "p_upper": 1.0, "p_lower": 1.0, "n_pairs": 0}

        p_upper = 1 - stats.binom.cdf(n_plus - 1, n, gamma / (1 + gamma))
        p_lower = 1 - stats.binom.cdf(n_plus - 1, n, 1 / (1 + gamma))

        return {
            "gamma": float(gamma),
            "p_upper": float(p_upper),
            "p_lower": float(p_lower),
            "n_pairs": n,
        }

    @staticmethod
    def sensitivity_curve(
        treated_outcomes: np.ndarray, control_outcomes: np.ndarray, gammas: list[float] | None = None
    ) -> dict:
        """Bias-robustness curve: p_upper at each gamma, and the breakpoint gamma at which
        significance (p_upper > 0.05) would first be lost."""
        if gammas is None:
            gammas = [1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]

        points = []
        breakpoint_gamma = None
        for g in gammas:
            r = SensitivityAnalysis.rosenbaum_bounds(treated_outcomes, control_outcomes, g)
            points.append({"gamma": g, "p_upper": r["p_upper"]})
            if breakpoint_gamma is None and r["p_upper"] > 0.05:
                breakpoint_gamma = g

        return {
            "points": points,
            "breakpoint_gamma": breakpoint_gamma,
            "bias_robustness": breakpoint_gamma if breakpoint_gamma is not None else max(gammas),
        }
