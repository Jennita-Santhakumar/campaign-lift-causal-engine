# Causal Impact Analysis — Marketing Campaigns

A causal-inference engine for measuring the true effect of a marketing campaign on customer
spend, when campaign exposure was **not** randomly assigned. Implements propensity score
matching (PSM), inverse probability weighting (IPW), difference-in-differences (DiD) on panel
data, conditional average treatment effects (CATE) by segment, and Rosenbaum sensitivity bounds
for robustness to unobserved confounding.

## Why this exists

Most marketing campaigns target customers non-randomly — more engaged, higher-frequency
customers are more likely to receive a promo. A naive "treated vs. untreated" spend comparison
is therefore biased: some of the apparent lift is really just "engaged customers spend more
regardless." This project simulates exactly that scenario (with a known ground-truth effect
baked into the data generator) and shows how much of that bias propensity-score adjustment
actually removes.

## Architecture

```
┌─────────────┐      ┌──────────────────┐      ┌─────────────────────────────┐
│   React     │◄────►│  FastAPI backend  │◄────►│  Postgres                    │
│  dashboard  │ REST │                    │      │  campaigns, treatment_data,  │
│ (port 5230) │      │  internal/causal/  │      │  propensity_scores,          │
└─────────────┘      │  - propensity.py   │      │  causal_estimates,           │
                      │  - matching.py     │      │  sensitivity_analysis        │
                      │  - balance.py      │      └─────────────────────────────┘
                      │  - estimation.py   │
                      │  - sensitivity.py  │
                      └──────────────────┘
                          (port 8060)
```

- **Synthetic data generator** (`app/services/synthetic.py`): builds a panel of customers with
  covariates (purchase frequency, avg order value, tenure, engagement score, region), assigns
  treatment via a logistic function of those covariates (real confounding), and generates
  pre/post spend with a known treatment effect (`true_ate`) baked into the outcome.
- **Analysis pipeline** (`app/services/analysis.py`): loads a campaign's panel data, estimates
  propensity scores, matches treated/control units, computes ATE via naive/PSM/IPW/DiD, CATE by
  engagement segment, a covariate balance report, and a Rosenbaum sensitivity curve — then
  persists results to Postgres.

## Setup

```bash
docker compose up --build
```

- Backend: http://localhost:8060 (docs at `/docs`)
- Frontend: http://localhost:5230
- Postgres: localhost:5510

Seed a campaign and view results either via the dashboard ("Seed new campaign" → "Run analysis")
or directly:

```bash
curl -X POST http://localhost:8060/api/v1/causal/seed \
  -H "Content-Type: application/json" \
  -d '{"name": "Q3 Discount Campaign", "n_customers": 4000, "true_ate": 12.5, "seed": 42}'

curl -X POST http://localhost:8060/api/v1/causal/estimate \
  -H "Content-Type: application/json" \
  -d '{"campaign_id": "<id from seed response>", "method": "psm"}'
```

## API reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/v1/causal/seed` | POST | Generate and persist a synthetic campaign panel dataset |
| `/api/v1/causal/campaigns` | GET | List seeded campaigns |
| `/api/v1/causal/estimate` | POST | Run the full pipeline; returns ATE (chosen method) + all-methods comparison + balance + CATE + sensitivity curve |
| `/api/v1/causal/campaigns/{id}/balance` | GET | Covariate balance report (SMD before/after matching) |
| `/api/v1/causal/campaigns/{id}/cate` | GET | CATE broken down by engagement segment |
| `/api/v1/causal/campaigns/{id}/sensitivity` | GET | Rosenbaum sensitivity curve |

## Testing

```bash
cd backend
python -m venv venv && source venv/Scripts/activate   # or venv/bin/activate on Linux/Mac
pip install -r requirements-dev.txt
pytest -v
```

Unit tests cover propensity score bounds, SMD balance calculations, ATE/DiD/CATE recovery on
synthetic data with known effects, and Rosenbaum bound monotonicity. `test_integration_pipeline.py`
spins up a real Postgres via testcontainers, seeds a campaign, runs the full pipeline, and asserts
PSM/IPW adjustment gets closer to the true effect than the naive estimate.

## Results from a real run

Seeded 4,000 customers (8,000 panel rows), `true_ate = $12.50`:

| Method | Estimated ATE | 95% CI | Notes |
|---|---|---|---|
| **Naive** (unadjusted) | **$27.98** | [25.75, 30.20] | Badly biased — confounding roughly doubles the apparent effect |
| **PSM** | **$11.99** | [9.33, 14.65] | Within 4% of true ATE after nearest-neighbor caliper matching |
| **IPW** | **$12.71** | [-71.28, 96.71] | Point estimate close to truth, but CI is very wide — a handful of units near propensity 0/1 dominate the weights (a known IPW weakness, visible directly in these results rather than smoothed over) |
| **DiD** (panel) | **$12.31** | [9.22, 15.39] | Closest to truth — differencing out unit-level pre-period baseline cancels most of the confounding directly |

Covariate balance: max |SMD| dropped from **0.80** (engagement score, before matching) to **0.04**
(after matching) — well under the conventional 0.1 threshold.

CATE by engagement segment came out roughly flat (~$13-14 across low/medium/high engagement),
which is expected: the synthetic outcome model has a homogeneous treatment effect by design (it
doesn't interact `treatment` with any covariate). A flat CATE here is the pipeline correctly
reporting "no detected heterogeneity," not a limitation of the T-learner.

Rosenbaum sensitivity: the PSM result stays significant (p_upper < 0.05) up to **Γ ≈ 1.75** —
an unobserved confounder would need to change the odds of treatment by ~75% to overturn the
result.

## Design decisions & trade-offs

- **CATE via T-learner (two `RandomForestRegressor`s), not `econml`/`causalml`.** Those
  dedicated causal-ML libraries have a heavier, more fragile dependency chain on Windows/Python
  3.11 for a marginal accuracy gain on this synthetic dataset. A T-learner is a standard,
  well-understood CATE estimator and keeps the project's dependency footprint identical to the
  rest of the stack (pure scikit-learn).
- **DiD adapted to real panel data**, not the flat cross-sectional arrays the source PRD sketches.
  The PRD's own DiD sketch takes unit-agnostic flat arrays, so it can't distinguish a real panel
  (same customers observed pre/post) from two unrelated cross-sections. This implementation
  requires unit-aligned pre/post rows, matching how the synthetic generator and DB schema are
  actually structured, and it shows in the results above — DiD is the *most* accurate estimator
  here specifically because it differences out each unit's own baseline.
  Rosenbaum bounds computed here operate on **real matched pairs** from `matching.py`, not
  arbitrary unmatched arrays.
- **IPW's wide CI is reported honestly**, not hidden. It's a real, well-known failure mode of
  unstabilized IPW when a few units have propensity scores near 0 or 1 — visible directly in the
  95% CI above rather than smoothed over.

## Tech stack

FastAPI · SQLAlchemy · Postgres · scikit-learn · NumPy/SciPy/Pandas · React 18 · Vite · Recharts
· Docker Compose · GitHub Actions CI
