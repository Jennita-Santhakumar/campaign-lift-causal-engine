"""Integration test: seed a real campaign into a real Postgres, run the full pipeline,
and check the PSM-adjusted ATE is closer to the known synthetic ground truth than the
naive (unadjusted) estimate — the actual point of doing propensity-score adjustment.
"""

from app.db.models import Campaign, TreatmentData
from app.services.analysis import run_full_analysis
from app.services.synthetic import COVARIATE_NAMES, generate_campaign_dataset


def test_full_pipeline_against_real_postgres(db_session):
    true_ate = 12.5
    campaign = Campaign(name="Integration Test Campaign", true_ate=true_ate)
    db_session.add(campaign)
    db_session.flush()

    df = generate_campaign_dataset(n_customers=1500, true_ate=true_ate, seed=7)
    for _, row in df.iterrows():
        db_session.add(
            TreatmentData(
                campaign_id=campaign.id,
                unit_id=row["unit_id"],
                treatment=bool(row["treatment"]),
                period=int(row["period"]),
                outcome=float(row["outcome"]),
                covariates={k: (float(row[k]) if k != "region_code" else int(row[k])) for k in COVARIATE_NAMES}
                | {"segment": str(row["segment"])},
            )
        )
    db_session.commit()

    results = run_full_analysis(db_session, campaign.id)

    naive_error = abs(results["naive"]["ate"] - true_ate)
    psm_error = abs(results["psm"]["ate"] - true_ate)
    ipw_error = abs(results["ipw"]["ate"] - true_ate)

    # The whole point of the demo: adjustment should get closer to truth than the naive estimate.
    assert psm_error < naive_error
    assert ipw_error < naive_error
    assert results["n_matched_pairs"] > 0
    assert "by_segment" in results["cate"]
    assert results["balance"]["max_abs_smd_after"] <= results["balance"]["max_abs_smd_before"]
