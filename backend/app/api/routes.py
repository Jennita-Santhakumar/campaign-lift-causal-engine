import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.config import settings
from app.db.base import get_db
from app.db.models import Campaign, TreatmentData
from app.api.schemas import CampaignOut, EstimateRequest, SeedRequest
from app.services.analysis import run_full_analysis
from app.services.synthetic import COVARIATE_NAMES, generate_campaign_dataset

router = APIRouter(prefix="/api/v1/causal", tags=["causal"])


@router.post("/seed", response_model=CampaignOut)
def seed_campaign(req: SeedRequest, db: Session = Depends(get_db)):
    n = req.n_customers or settings.seed_n_customers
    true_ate = req.true_ate if req.true_ate is not None else settings.true_ate

    campaign = Campaign(name=req.name, true_ate=true_ate)
    db.add(campaign)
    db.flush()

    df = generate_campaign_dataset(n, true_ate, seed=req.seed)
    for _, row in df.iterrows():
        db.add(
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
    db.commit()
    db.refresh(campaign)
    return campaign


@router.get("/campaigns", response_model=list[CampaignOut])
def list_campaigns(db: Session = Depends(get_db)):
    return db.query(Campaign).order_by(Campaign.created_at.desc()).all()


@router.post("/estimate")
def estimate(req: EstimateRequest, db: Session = Depends(get_db)):
    campaign = db.query(Campaign).filter(Campaign.id == req.campaign_id).first()
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    try:
        results = run_full_analysis(db, req.campaign_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    chosen = results.get(req.method)
    if chosen is None:
        raise HTTPException(status_code=400, detail=f"Unknown method '{req.method}'")

    return {
        "method": req.method,
        "true_ate": campaign.true_ate,
        "ate": chosen["ate"],
        "ci_lower": chosen["ci_lower"],
        "ci_upper": chosen["ci_upper"],
        "p_value": chosen["p_value"],
        "sample_size": chosen["sample_size"],
        "interpretation": (
            f"Treatment {'increases' if chosen['ate'] >= 0 else 'decreases'} outcome by "
            f"~{abs(chosen['ate']):.2f} units (method={req.method})"
        ),
        "all_methods": {
            m: {"ate": results[m]["ate"], "ci_lower": results[m]["ci_lower"], "ci_upper": results[m]["ci_upper"]}
            for m in ("naive", "psm", "ipw", "did")
        },
        "balance": results["balance"],
        "common_support": results["common_support"],
        "cate": results["cate"],
        "sensitivity_curve": results["sensitivity_curve"],
        "n_matched_pairs": results["n_matched_pairs"],
    }


@router.get("/campaigns/{campaign_id}/balance")
def get_balance(campaign_id: uuid.UUID, db: Session = Depends(get_db)):
    results = run_full_analysis(db, campaign_id)
    return results["balance"]


@router.get("/campaigns/{campaign_id}/sensitivity")
def get_sensitivity(campaign_id: uuid.UUID, db: Session = Depends(get_db)):
    results = run_full_analysis(db, campaign_id)
    return results["sensitivity_curve"]


@router.get("/campaigns/{campaign_id}/cate")
def get_cate(campaign_id: uuid.UUID, db: Session = Depends(get_db)):
    results = run_full_analysis(db, campaign_id)
    return results["cate"]
