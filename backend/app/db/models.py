import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from app.db.base import Base


class Campaign(Base):
    __tablename__ = "campaigns"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    true_ate = Column(Float, nullable=True)  # known synthetic ground truth, null for real data
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)


class TreatmentData(Base):
    """One row per (customer, period). period=0 pre-campaign, period=1 post-campaign."""

    __tablename__ = "treatment_data"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    campaign_id = Column(UUID(as_uuid=True), ForeignKey("campaigns.id"), nullable=False, index=True)
    unit_id = Column(String(255), nullable=False, index=True)
    treatment = Column(Boolean, nullable=False)
    period = Column(Integer, nullable=False)  # 0=pre, 1=post
    outcome = Column(Float, nullable=False)
    covariates = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)


class PropensityScore(Base):
    __tablename__ = "propensity_scores"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    campaign_id = Column(UUID(as_uuid=True), ForeignKey("campaigns.id"), nullable=False, index=True)
    unit_id = Column(String(255), nullable=False, index=True)
    propensity_score = Column(Float, nullable=False)
    decile = Column(Integer, nullable=False)
    calculated_at = Column(DateTime(timezone=True), default=datetime.utcnow)


class CausalEstimate(Base):
    __tablename__ = "causal_estimates"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    campaign_id = Column(UUID(as_uuid=True), ForeignKey("campaigns.id"), nullable=False, index=True)
    method = Column(String(50), nullable=False)
    ate = Column(Float, nullable=False)
    ate_ci_lower = Column(Float, nullable=False)
    ate_ci_upper = Column(Float, nullable=False)
    p_value = Column(Float, nullable=False)
    sample_size = Column(Integer, nullable=False)
    meta = Column(JSONB, nullable=True)  # e.g. CATE-by-segment breakdown, balance report
    estimated_at = Column(DateTime(timezone=True), default=datetime.utcnow)


class SensitivityAnalysis(Base):
    __tablename__ = "sensitivity_analysis"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    campaign_id = Column(UUID(as_uuid=True), ForeignKey("campaigns.id"), nullable=False, index=True)
    method = Column(String(50), nullable=False)
    rosenbaum_gamma = Column(Float, nullable=False)
    bias_robustness = Column(Float, nullable=False)
    estimated_at = Column(DateTime(timezone=True), default=datetime.utcnow)
