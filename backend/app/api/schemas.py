import uuid

from pydantic import BaseModel


class SeedRequest(BaseModel):
    name: str = "Q3 Discount Campaign"
    n_customers: int | None = None
    true_ate: float | None = None
    seed: int | None = 42


class CampaignOut(BaseModel):
    id: uuid.UUID
    name: str
    true_ate: float | None

    class Config:
        from_attributes = True


class EstimateRequest(BaseModel):
    """Matches the PRD's POST /api/v1/causal/estimate shape, extended with a campaign_id
    so the estimate can be computed from data already persisted via /seed, instead of
    requiring the raw arrays to be re-sent on every call."""

    campaign_id: uuid.UUID
    method: str = "psm"  # naive | psm | ipw | did
