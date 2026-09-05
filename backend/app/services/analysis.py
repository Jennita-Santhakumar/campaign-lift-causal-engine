"""Orchestrates the causal-inference pipeline: loads panel data for a campaign from the DB,
runs propensity/matching/balance/estimation/sensitivity, and returns/persists results.
"""

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.db.models import CausalEstimate, PropensityScore, SensitivityAnalysis, TreatmentData
from app.internal.causal.balance import balance_report
from app.internal.causal.estimation import CausalEstimation
from app.internal.causal.matching import PropensityScoreMatching
from app.internal.causal.propensity import PropensityScoreAnalysis
from app.internal.causal.sensitivity import SensitivityAnalysis as Sensitivity
from app.services.synthetic import COVARIATE_NAMES, SEGMENT_NAMES


def load_panel(db: Session, campaign_id) -> pd.DataFrame:
    rows = db.query(TreatmentData).filter(TreatmentData.campaign_id == campaign_id).all()
    if not rows:
        raise ValueError(f"No data found for campaign {campaign_id}")
    records = []
    for r in rows:
        rec = {"unit_id": r.unit_id, "treatment": r.treatment, "period": r.period, "outcome": r.outcome}
        rec.update(r.covariates)
        records.append(rec)
    return pd.DataFrame(records)


def _post_period_frame(panel: pd.DataFrame) -> pd.DataFrame:
    return panel[panel["period"] == 1].reset_index(drop=True)


def run_full_analysis(db: Session, campaign_id) -> dict:
    panel = load_panel(db, campaign_id)
    post = _post_period_frame(panel)

    X = post[COVARIATE_NAMES].to_numpy(dtype=float)
    treatment = post["treatment"].to_numpy().astype(int)
    outcome = post["outcome"].to_numpy(dtype=float)
    unit_ids = post["unit_id"].to_numpy()

    # --- Naive (unadjusted) comparison, to show the bias matching corrects ---
    naive = CausalEstimation.estimate_ate(outcome, treatment)

    # --- Propensity scores + common support ---
    ps = PropensityScoreAnalysis.estimate_propensity_score(X, treatment)
    common_support = PropensityScoreAnalysis.check_common_support(ps[treatment == 1], ps[treatment == 0])

    db.query(PropensityScore).filter(PropensityScore.campaign_id == campaign_id).delete()
    deciles = PropensityScoreMatching.stratify(ps, n_strata=10)
    for uid, score, decile in zip(unit_ids, ps, deciles):
        db.add(PropensityScore(campaign_id=campaign_id, unit_id=str(uid), propensity_score=float(score), decile=int(decile)))

    # --- PSM: nearest-neighbor matching with caliper ---
    treated_idx_all = np.where(treatment == 1)[0]
    control_idx_all = np.where(treatment == 0)[0]
    matches = PropensityScoreMatching.match_neighbors(ps[treated_idx_all], ps[control_idx_all], caliper=0.05)

    matched_treated_idx = np.array([treated_idx_all[i] for i in matches.keys()])
    matched_control_idx = np.array([control_idx_all[j] for j in matches.values()])

    psm_result = CausalEstimation.estimate_ate(
        np.concatenate([outcome[matched_treated_idx], outcome[matched_control_idx]]),
        np.concatenate([np.ones(len(matched_treated_idx)), np.zeros(len(matched_control_idx))]),
    )

    # --- IPW ---
    weights = PropensityScoreMatching.ipw_weights(ps, treatment)
    ipw_result = CausalEstimation.estimate_ate(outcome, treatment, weights=weights)

    # --- Balance assessment (before vs. after PSM) ---
    balance = balance_report(X, treatment, COVARIATE_NAMES, matched_treated_idx, matched_control_idx)

    # --- CATE by engagement segment ---
    segment_labels = post["segment"].map({name: i for i, name in enumerate(SEGMENT_NAMES)}).to_numpy()
    cate = CausalEstimation.estimate_cate(X, outcome, treatment, segment_labels, SEGMENT_NAMES)

    # --- DiD on full panel ---
    did_result = CausalEstimation.did_estimate_panel(
        panel["outcome"].to_numpy(dtype=float),
        panel["treatment"].to_numpy().astype(int),
        panel["period"].to_numpy().astype(int),
    )

    # --- Sensitivity analysis on the PSM matched pairs ---
    curve = Sensitivity.sensitivity_curve(outcome[matched_treated_idx], outcome[matched_control_idx])

    db.query(CausalEstimate).filter(CausalEstimate.campaign_id == campaign_id).delete()
    db.query(SensitivityAnalysis).filter(SensitivityAnalysis.campaign_id == campaign_id).delete()

    for method, result, meta in [
        ("naive", naive, None),
        ("psm", psm_result, {"balance": balance, "common_support": common_support}),
        ("ipw", ipw_result, {"common_support": common_support}),
        ("did", did_result, None),
    ]:
        db.add(
            CausalEstimate(
                campaign_id=campaign_id,
                method=method,
                ate=result["ate"],
                ate_ci_lower=result["ci_lower"],
                ate_ci_upper=result["ci_upper"],
                p_value=result["p_value"],
                sample_size=result["sample_size"],
                meta=meta,
            )
        )

    db.add(
        SensitivityAnalysis(
            campaign_id=campaign_id,
            method="rosenbaum",
            rosenbaum_gamma=curve["breakpoint_gamma"] or 0.0,
            bias_robustness=curve["bias_robustness"],
        )
    )
    db.commit()

    return {
        "naive": naive,
        "psm": psm_result,
        "ipw": ipw_result,
        "did": did_result,
        "balance": balance,
        "common_support": common_support,
        "cate": cate,
        "sensitivity_curve": curve,
        "n_matched_pairs": int(len(matched_treated_idx)),
    }
