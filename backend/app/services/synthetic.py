"""Synthetic marketing-campaign panel dataset with confounded treatment assignment
and a known ground-truth treatment effect, so estimated ATE can be checked against truth.
"""

import numpy as np
import pandas as pd

REGIONS = ["north", "south", "east", "west"]
COVARIATE_NAMES = [
    "purchase_frequency",
    "avg_order_value",
    "tenure_days",
    "engagement_score",
    "region_code",
]
SEGMENT_NAMES = ["low_engagement", "medium_engagement", "high_engagement"]


def generate_campaign_dataset(n_customers: int, true_ate: float, seed: int | None = None) -> pd.DataFrame:
    """Returns a long-format DataFrame with one row per (customer, period) — period 0 = pre-
    campaign, period 1 = post-campaign. Treatment (campaign exposure) is confounded with
    covariates: customers who purchase more often and are more engaged are more likely to
    have been targeted by the campaign, so a naive post-only or pre/post-only comparison is
    biased and matching/weighting is actually necessary to recover `true_ate`.
    """
    rng = np.random.default_rng(seed)

    purchase_frequency = rng.gamma(shape=3.0, scale=2.0, size=n_customers)  # purchases/month
    avg_order_value = rng.lognormal(mean=3.8, sigma=0.4, size=n_customers)  # ~$45 median
    tenure_days = rng.exponential(scale=400, size=n_customers)
    engagement_score = rng.beta(a=2.0, b=2.5, size=n_customers)  # in [0,1]
    region_code = rng.integers(0, len(REGIONS), size=n_customers)

    # Propensity: more frequent, more engaged, longer-tenured customers are more likely to be
    # targeted by the campaign — real confounding, not random assignment.
    logit = (
        -1.2
        + 0.35 * (purchase_frequency - purchase_frequency.mean()) / purchase_frequency.std()
        + 0.9 * (engagement_score - engagement_score.mean()) / engagement_score.std()
        + 0.25 * (tenure_days - tenure_days.mean()) / tenure_days.std()
    )
    prob_treated = 1 / (1 + np.exp(-logit))
    treatment = rng.binomial(1, prob_treated)

    # Baseline (pre-period) spend depends on covariates only — no treatment effect yet.
    baseline_spend = (
        20
        + 8.0 * purchase_frequency
        + 0.15 * avg_order_value
        + 0.01 * tenure_days
        + 25 * engagement_score
        + rng.normal(0, 8, size=n_customers)
    )
    baseline_spend = np.clip(baseline_spend, 5, None)

    # Post-period: organic time trend (affects everyone) + the true causal treatment effect,
    # baked in directly so we can compare the estimate against a known number.
    time_trend = rng.normal(3.0, 1.5, size=n_customers)  # small organic seasonal lift
    post_spend = baseline_spend + time_trend + treatment * true_ate + rng.normal(0, 8, size=n_customers)
    post_spend = np.clip(post_spend, 5, None)

    unit_ids = [f"cust_{i:06d}" for i in range(n_customers)]

    engagement_terciles = pd.qcut(engagement_score, 3, labels=SEGMENT_NAMES)

    pre_rows = pd.DataFrame(
        {
            "unit_id": unit_ids,
            "treatment": treatment.astype(bool),
            "period": 0,
            "outcome": baseline_spend,
            "purchase_frequency": purchase_frequency,
            "avg_order_value": avg_order_value,
            "tenure_days": tenure_days,
            "engagement_score": engagement_score,
            "region_code": region_code,
            "segment": engagement_terciles,
        }
    )
    post_rows = pre_rows.copy()
    post_rows["period"] = 1
    post_rows["outcome"] = post_spend

    return pd.concat([pre_rows, post_rows], ignore_index=True)
