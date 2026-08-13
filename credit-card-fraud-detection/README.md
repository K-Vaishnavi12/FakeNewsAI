# SentinelPay AI

**Real-Time Hybrid Fraud Intelligence & Risk Decision Platform**

Submission for the Cognizant problem statement: *Credit Card Fraud Detection — Anomaly Detection*.

> Build an AI system that identifies whether a credit-card transaction is genuine or fraudulent based on transaction details such as amount, type, location and time.

SentinelPay does not stop at `fraud / genuine`. It answers four questions in under one second:

1. **Is this transaction suspicious?** → binary prediction
2. **How suspicious?** → risk score `0–100`
3. **Why?** → ranked, human-readable reason codes (SHAP + rule hits)
4. **What should the bank do?** → `APPROVE` / `STEP-UP AUTH` / `BLOCK`

---

## Table of Contents

- [1. Why this design](#1-why-this-design)
- [2. System architecture](#2-system-architecture)
- [3. The three intelligence layers](#3-the-three-intelligence-layers)
- [4. Datasets](#4-datasets)
- [5. On "live" data and streaming](#5-on-live-data-and-streaming)
- [6. On MCP servers](#6-on-mcp-servers)
- [7. Feature engineering](#7-feature-engineering)
- [8. Risk engine & decision policy](#8-risk-engine--decision-policy)
- [9. Explainability](#9-explainability)
- [10. API specification](#10-api-specification)
- [11. Dashboard](#11-dashboard)
- [12. Tech stack](#12-tech-stack)
- [13. Repository layout](#13-repository-layout)
- [14. Getting started](#14-getting-started)
- [15. Evaluation methodology](#15-evaluation-methodology)
- [16. Build plan](#16-build-plan)
- [17. Honest limitations](#17-honest-limitations)

---

## 1. Why this design

A single classifier trained on a labelled dataset has three fatal weaknesses in production fraud:

| Weakness | Consequence | Our mitigation |
|---|---|---|
| Extreme class imbalance (~0.17–0.6% fraud) | Accuracy of 99.8% is achieved by predicting "genuine" always | Optimise **PR-AUC / recall @ fixed FPR**, never accuracy |
| Cannot detect fraud patterns absent from training labels (zero-day) | New attack vectors pass through | **Unsupervised anomaly layer** (Isolation Forest) |
| Output is a black box; a bank cannot legally block a card without a reason | Regulatory + customer-trust failure | **SHAP + deterministic rule reason codes** |

Hence a **hybrid** engine: supervised model + anomaly detector + behavioural profile + deterministic rules, fused into one score.

---

## 2. System architecture

```
                       ┌────────────────────────────┐
                       │   Fraud Analyst / Client   │
                       └─────────────┬──────────────┘
                                     │
                       ┌─────────────▼──────────────┐
                       │   React + Tailwind SPA     │
                       │  Dashboard · Investigate   │
                       │  What-If Simulator         │
                       └─────────────┬──────────────┘
                                     │ HTTPS / JSON
                       ┌─────────────▼──────────────┐
                       │        FastAPI (ASGI)      │
                       │  Pydantic validation, auth │
                       └─────────────┬──────────────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        │                            │                            │
┌───────▼────────┐          ┌────────▼────────┐         ┌─────────▼────────┐
│  Preprocessing │          │ Feature Engine  │         │ Customer Profile │
│  & Validation  │─────────▶│ 40+ derived     │◀────────│ Engine (Redis)   │
└────────────────┘          │ features        │         │ rolling stats    │
                            └────────┬────────┘         └──────────────────┘
                                     │
                    ┌────────────────▼─────────────────┐
                    │     FRAUD INTELLIGENCE ENGINE    │
                    └────────────────┬─────────────────┘
             ┌───────────────────────┼───────────────────────┐
     ┌───────▼────────┐     ┌────────▼────────┐    ┌─────────▼────────┐
     │  L1 Supervised │     │  L2 Anomaly     │    │  L3 Rule Engine  │
     │  XGBoost       │     │  IsolationForest│    │  velocity, geo,  │
     │  P(fraud)      │     │  anomaly_score  │    │  device, testing │
     └───────┬────────┘     └────────┬────────┘    └─────────┬────────┘
             └───────────────────────┼───────────────────────┘
                    ┌────────────────▼─────────────────┐
                    │   HYBRID RISK FUSION  →  0..100  │
                    └────────────────┬─────────────────┘
                                     │
             ┌───────────────────────┼───────────────────────┐
        ┌────▼─────┐          ┌──────▼──────┐         ┌──────▼──────┐
        │ APPROVE  │          │  STEP-UP    │         │    BLOCK    │
        │  0–30    │          │   31–70     │         │   71–100    │
        └────┬─────┘          └──────┬──────┘         └──────┬──────┘
             └───────────────────────┼───────────────────────┘
                    ┌────────────────▼─────────────────┐
                    │  Explainability (SHAP + codes)   │
                    └────────────────┬─────────────────┘
                    ┌────────────────▼─────────────────┐
                    │  PostgreSQL — txns, scores,      │
                    │  alerts, profiles, audit log     │
                    └──────────────────────────────────┘
```

---

## 3. The three intelligence layers

### Layer 1 — Supervised model (XGBoost)

Trained on labelled historical transactions. Handles **known** fraud patterns with high precision.

- Algorithm: `XGBClassifier`, `scale_pos_weight = n_neg/n_pos`, `tree_method="hist"`
- Baseline comparison: Logistic Regression → Random Forest → XGBoost → LightGBM
- Output: `p_fraud ∈ [0,1]`
- Calibrated with `CalibratedClassifierCV` (isotonic) so the probability is meaningful, not just a ranking

### Layer 2 — Anomaly detection (Isolation Forest)

Trained on **genuine transactions only**. Detects transactions that are structurally unlike anything the customer base normally does — including attack types absent from the training labels.

- `IsolationForest(n_estimators=200, contamination=0.01)`
- Raw `score_samples` output is min-max normalised to `anomaly_score ∈ [0,1]`
- Optional upgrade: `LocalOutlierFactor(novelty=True)` or an autoencoder reconstruction-error model

### Layer 3 — Behavioural profile + rule engine

For every card we maintain a rolling profile:

| Profile attribute | Example |
|---|---|
| `avg_amount`, `std_amount`, `p95_amount` | ₹2,400 / ₹1,100 / ₹6,800 |
| `home_lat`, `home_lon`, `usual_city` | Hyderabad |
| `active_hours` histogram | 09:00–22:00 |
| `known_devices` set | `{iphone_a91f, web_chrome_x2}` |
| `merchant_category` distribution | grocery 41%, food 30%, fuel 12% |
| `txn_per_day` mean | 3.1 |

Deterministic rules fired against this profile:

| Rule ID | Condition | Weight |
|---|---|---|
| `R_AMT_SPIKE` | `amount > avg + 4·std` | +20 |
| `R_GEO_FOREIGN` | country ∉ customer's historical countries | +18 |
| `R_IMPOSSIBLE_TRAVEL` | implied speed between consecutive txns > 900 km/h | +30 |
| `R_ODD_HOUR` | hour ∈ [00:00, 05:00] and outside `active_hours` | +12 |
| `R_NEW_DEVICE` | `device_id ∉ known_devices` | +15 |
| `R_VELOCITY` | ≥ 5 txns in 120 s | +25 |
| `R_CARD_TESTING` | ≥ 4 txns < ₹50 within 10 min | +28 |
| `R_MERCHANT_DRIFT` | category probability in profile < 2% | +8 |

Rule weights are summed then squashed: `rule_score = min(1.0, Σweights / 100)`.

---

## 4. Datasets

### Direct answer: which dataset should we use?

The famous ULB `creditcard.csv` is **the wrong choice for this problem statement.** Its 30 columns are `Time, V1…V28, Amount, Class`, where `V1–V28` are PCA components of undisclosed original fields. You physically cannot build location logic, merchant logic, device logic, behavioural profiling, or a meaningful explanation ("this was flagged because V17 = -4.2" means nothing to a judge). Use it only as a sanity-check baseline, if at all.

### Recommended dataset stack

| # | Dataset | Rows / Fraud | Key fields | Role in SentinelPay |
|---|---|---|---|---|
| **1** | **Sparkov Credit Card Transactions** — Kaggle `kartik2112/fraud-detection` | 1.29M train / 7,506 fraud (0.58%) | `trans_date_trans_time, cc_num, merchant, category, amt, city, state, lat, long, city_pop, job, dob, merch_lat, merch_long, is_fraud` | **PRIMARY.** Supplies amount, time, geo-coords, merchant category and customer identity — everything the problem statement asks for. Enables real haversine distance features and true behavioural profiles per `cc_num`. |
| **2** | **PaySim** — Kaggle `ealaxi/paysim1` | 6.36M / 8,213 fraud | `step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, isFraud` | **SECONDARY.** Covers transaction **`type`** (`TRANSFER`, `CASH_OUT`, `PAYMENT`, `DEBIT`, `CASH_IN`) which the statement names explicitly. Great for a second demo tab and for balance-drain rules. |
| **3** | **IEEE-CIS Fraud Detection** — Kaggle `c/ieee-fraud-detection` | 590k / 3.5% fraud | `DeviceType, DeviceInfo, id_30 (OS), id_31 (browser), card1-6, addr1-2, dist1-2, P_emaildomain` | **OPTIONAL.** The only public set with genuine device fingerprints. Use to justify/calibrate the device-anomaly weight. 434 columns — heavy. |
| **4** | **BankSim** — Kaggle `ealaxi/banksim1` | 594k | `step, customer, age, gender, merchant, category, amount, fraud` | Fallback / cross-validation of merchant-category logic. |
| **5** | **Synthetic Attack Injector** *(we build this)* | ~2,000 crafted txns | device_id, country, ms-precision timestamps | **ESSENTIAL FOR DEMO.** No public dataset contains card-testing bursts, impossible travel across countries, or device takeover. We generate them. |

### Download

Requires a Kaggle account and `~/.kaggle/kaggle.json` API token.

```bash
pip install kaggle
kaggle datasets download -d kartik2112/fraud-detection      -p data/raw --unzip   # Sparkov  (primary)
kaggle datasets download -d ealaxi/paysim1                  -p data/raw --unzip   # PaySim
kaggle datasets download -d ealaxi/banksim1                 -p data/raw --unzip   # BankSim
kaggle competitions download -c ieee-fraud-detection        -p data/raw           # IEEE-CIS (accept rules first)
```

Or run `python scripts/download_data.py` which wraps the above and verifies checksums.

### The synthetic attack injector

`scripts/generate_attacks.py` takes real Sparkov customers and injects labelled attack scenarios on top of their genuine history:

```
card_testing      : 6 txns of ₹1–₹40 at the same merchant within 8 minutes
velocity_burst    : 11 txns in 74 seconds across 5 merchants
impossible_travel : Hyderabad 02:10 → London 02:34  (implied 7,900 km/h)
account_takeover  : new device_id + new country + 21× average amount at 02:13
merchant_drift    : grocery-only customer suddenly buys luxury electronics
micro_then_macro  : ₹5 probe, then ₹78,000 nine minutes later
```

Each injected row carries `attack_type` so the dashboard can display the ground-truth pattern name and you can report per-pattern recall. **Synthetic rows are disclosed in the presentation and reported separately from real-data metrics.** Never blend them into the headline PR-AUC.

---

## 5. On "live" data and streaming

**There is no public live credit-card transaction feed.** Real cardholder data is protected by PCI-DSS and cannot be published; every "real-time fraud API" on the market is a paid commercial product (Stripe Radar, Sift, Feedzai) and none expose training data.

So SentinelPay achieves *real-time* the correct way — **the engine is genuinely real-time, the source is a replay.**

```
data/processed/stream.parquet
        │  chronological order preserved
        ▼
scripts/stream_replay.py   ── configurable speed-up (1×, 60×, 3600×)
        │  POST /api/v1/transactions/analyze
        ▼
FastAPI  →  scored in <150 ms p95  →  PostgreSQL
        │
        ▼  WebSocket /ws/feed
React dashboard live tape
```

This is architecturally identical to production; only the producer differs. Swapping `stream_replay.py` for a Kafka consumer is a ~30-line change, and we say exactly that in the presentation. Claiming a live bank feed would be dishonest and any Cognizant judge would catch it immediately.

---

## 6. On MCP servers

**MCP servers are not part of this project's runtime and must not appear in the architecture diagram.**

Model Context Protocol is a standard for connecting *AI coding assistants* to external tools. It is a development-time convenience. A fraud-scoring engine has no use for it — inserting "MCP Server" into a bank-grade architecture slide is buzzword-stuffing and signals to a technical reviewer that the team does not understand what the component does.

Where MCP *is* legitimately useful — purely for the developers, never mentioned as a product feature:

| MCP server | Dev-time benefit |
|---|---|
| `postgres` | Let the coding assistant inspect the schema and run analytical queries while building |
| `filesystem` | Assistant reads/writes the repo |
| `fetch` | Pull dataset documentation and paper references |

Configure these in your local editor if you want. Keep them out of `docker-compose.yml`, out of the slides, and out of the demo.

---

## 7. Feature engineering

Derived in `src/features/`. Every feature is chosen because it is **explainable to a non-technical judge.**

**Amount**
- `amt`, `log_amt`
- `amt_over_cust_mean` — the "8.4× the customer's average" line in the demo
- `amt_zscore_vs_profile`
- `amt_pct_of_cust_p95`
- `is_micro_amount` (< ₹50, card-testing signal)

**Temporal**
- `hour`, `day_of_week`, `is_weekend`
- `is_night` (00:00–05:00)
- `hour_deviation_from_profile` — circular distance from the customer's modal hour
- `seconds_since_prev_txn`
- `txn_count_1min / 5min / 1h / 24h` (velocity)

**Geospatial**
- `haversine_km(cust_home, merchant)`
- `haversine_km(prev_txn, curr_txn)`
- `implied_speed_kmh` → drives `R_IMPOSSIBLE_TRAVEL`
- `is_foreign_country`, `distinct_countries_24h`

**Merchant / category**
- Target-encoded `category` (fit on train folds only, to avoid leakage)
- `category_freq_for_customer`
- `merchant_fraud_rate_historical` (smoothed, time-shifted to prevent leakage)

**Device / channel**
- `is_new_device`, `device_age_days`, `distinct_devices_7d`

**Demographic**
- `customer_age` from `dob`, `city_pop`, `job` (encoded)

> **Leakage discipline:** all customer-profile aggregates are computed with a **strictly causal expanding window** — a transaction's features use only transactions that occurred *before* it. Splits are **temporal**, never random. This is the single most common mistake in fraud-detection projects and calling it out explicitly earns credibility.

---

## 8. Risk engine & decision policy

### Fusion

```python
risk = 100 * (
      W_ML   * p_fraud            # XGBoost calibrated probability
    + W_ANOM * anomaly_score      # Isolation Forest, normalised
    + W_RULE * rule_score         # weighted rule hits, capped at 1.0
)
# defaults: W_ML = 0.55, W_ANOM = 0.25, W_RULE = 0.20   (sum = 1.0)
```

Plus **hard overrides** — non-negotiable in real banking:

```python
if "R_IMPOSSIBLE_TRAVEL" in fired_rules:  risk = max(risk, 90)
if "R_CARD_TESTING"      in fired_rules:  risk = max(risk, 85)
if card.is_reported_lost:                 risk = 100
```

Weights live in `config/risk_weights.yaml` and are tuned on a validation set to maximise recall at a fixed 1% false-positive rate. They are **not** hardcoded magic numbers — be ready to say how you tuned them.

### Decision bands

| Score | Level | Decision | Real-world action |
|---|---|---|---|
| 0–30 | LOW | `APPROVE` | Settle silently |
| 31–70 | MEDIUM | `STEP_UP_AUTH` | OTP / biometric challenge, or analyst queue |
| 71–100 | HIGH | `BLOCK` | Decline, freeze card, notify customer |

Band cut-offs are configurable, because a bank's risk appetite is a business decision, not a model decision. Show the trade-off curve: raising the block threshold from 71 → 80 cuts false blocks by *x*% but lets *y*% more fraud through.

---

## 9. Explainability

Two complementary sources, merged into one ranked list:

1. **SHAP** (`TreeExplainer` on the XGBoost model) → per-feature contribution to `p_fraud`
2. **Rule reason codes** → deterministic, plain-English, always available even when SHAP is ambiguous

Rendered response:

```
HIGH RISK TRANSACTION                              Risk Score 94/100
Decision: BLOCK                                    Confidence 0.96

Why was this flagged?
  Amount anomaly    ██████████████████  32%   ₹62,000 is 20.8× the customer's average
  Location anomaly  ███████████████     27%   United Kingdom; customer has only ever transacted in India
  Time anomaly      ████████████        21%   02:13 IST; customer is normally active 09:00–22:00
  Device anomaly    ███████             12%   Unrecognised device fingerprint
  Merchant anomaly  ████                 8%   Luxury Electronics; 0.4% of this customer's history
```

Each bar is `|shap_value| / Σ|shap_values|`, grouped from raw features into the five business-friendly buckets above via `config/feature_groups.yaml`.

---

## 10. API specification

Base URL `http://localhost:8000/api/v1` · interactive docs at `/docs` (Swagger) and `/redoc`.

### `POST /transactions/analyze`

Score a single transaction. Target latency: **p95 < 150 ms**.

**Request**
```json
{
  "transaction_id": "T10231",
  "customer_id": "C10452",
  "amount": 50000.00,
  "currency": "INR",
  "transaction_type": "PURCHASE",
  "merchant": "LuxeTronics London",
  "merchant_category": "electronics",
  "country": "GB",
  "city": "London",
  "latitude": 51.5074,
  "longitude": -0.1278,
  "timestamp": "2026-08-13T02:13:44Z",
  "device_id": "web_unknown_77b1",
  "channel": "ECOM"
}
```

**Response `200`**
```json
{
  "transaction_id": "T10231",
  "risk_score": 94,
  "risk_level": "HIGH",
  "prediction": "FRAUD",
  "decision": "BLOCK",
  "confidence": 0.96,
  "latency_ms": 87,
  "components": {
    "ml_probability": 0.87,
    "anomaly_score": 0.91,
    "rule_score": 0.83
  },
  "rules_fired": [
    "R_AMT_SPIKE", "R_GEO_FOREIGN", "R_ODD_HOUR", "R_NEW_DEVICE", "R_MERCHANT_DRIFT"
  ],
  "explanation": [
    { "factor": "amount_anomaly",   "contribution": 0.32, "detail": "₹50,000 is 20.8× customer average of ₹2,400" },
    { "factor": "location_anomaly", "contribution": 0.27, "detail": "GB not in customer's historical countries [IN]" },
    { "factor": "time_anomaly",     "contribution": 0.21, "detail": "02:13 outside active window 09:00–22:00" },
    { "factor": "device_anomaly",   "contribution": 0.12, "detail": "Unrecognised device web_unknown_77b1" },
    { "factor": "merchant_anomaly", "contribution": 0.08, "detail": "electronics = 0.4% of customer history" }
  ],
  "model_version": "xgb-1.4.0+iforest-1.1.0+rules-2026.08"
}
```

### Other endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/transactions/batch` | Score up to 1,000 transactions in one call |
| `POST` | `/simulate` | **What-If Simulator** — rescore a modified copy without persisting |
| `GET` | `/transactions/{id}` | Full investigation payload |
| `GET` | `/customers/{id}/profile` | Behavioural baseline for the investigation page |
| `GET` | `/analytics/summary` | KPI tiles: volume, fraud rate, blocked, amount saved |
| `GET` | `/analytics/timeseries?window=24h` | Fraud-trend chart |
| `GET` | `/alerts?status=open` | Analyst work queue |
| `PATCH` | `/alerts/{id}` | Analyst marks confirmed-fraud / false-positive → feedback loop |
| `GET` | `/health`, `/metrics` | Liveness + Prometheus metrics |
| `WS` | `/ws/feed` | Live transaction tape push |

---

## 11. Dashboard

Four screens. The last two are what differentiate this from a typical student ML demo.

**1. Command Centre**
KPI tiles (transactions scored, fraud rate, blocked count, ₹ prevented), live transaction tape via WebSocket with colour-coded risk, risk-distribution histogram, 24-hour fraud-trend line, geographic heat map.

**2. Alert Queue**
Sortable/filterable analyst worklist. Confirm or dismiss each alert — the decision writes back to `alerts` and feeds retraining.

**3. Transaction Investigation**
Deep dive on one transaction: risk gauge, decision badge, SHAP contribution bars, rules-fired chips, and a side-by-side **Customer Baseline vs This Transaction** table with deviation multipliers, plus a map showing home location → merchant location.

**4. What-If Simulator**
Sliders and dropdowns for amount, country, hour, device-known, merchant category. Hit **Recalculate** and watch the score move live:

```
Amount   ₹50,000 → ₹2,000
Country  GB      → IN
Time     02:00   → 14:00
Device   New     → Known

           94  HIGH RISK   ──────▶   18  LOW RISK
                BLOCK                    APPROVE
```

This proves to the judges that the model is genuinely reasoning over inputs rather than replaying a memorised result — and it turns the demo into something they can *touch*.

---

## 12. Tech stack

| Layer | Choice | Rationale |
|---|---|---|
| Frontend | React 18 + TypeScript + Tailwind + Recharts | Highest UI/UX ceiling; Recharts avoids fighting D3 |
| API | FastAPI + Pydantic v2 + Uvicorn | Async, auto-OpenAPI docs, native Python ML integration |
| Supervised ML | XGBoost (LightGBM as challenger) | Best-in-class on tabular imbalanced data |
| Anomaly | scikit-learn Isolation Forest | Fast, no labels required, well understood |
| Explainability | SHAP `TreeExplainer` | Exact, fast for tree models |
| Profile store | Redis | Sub-ms lookup of rolling customer stats on the hot path |
| Database | PostgreSQL 16 | Transactions, scores, alerts, audit trail |
| Data | pandas, NumPy, scikit-learn, PyArrow | Standard |
| Experiment tracking | MLflow | Model versioning + reproducible metrics for the report |
| Containers | Docker + docker-compose | One-command demo, no "works on my machine" |
| Testing | pytest, httpx, Playwright | Unit, API contract, and E2E |

**Fallback:** if the team is short on time, swap React for **Streamlit** and Redis for an in-memory dict. You will lose UI/UX points but keep the full engine. Do not cut the hybrid engine or the explainability — those are the scoring differentiators.

---

## 13. Repository layout

```
credit-card-fraud-detection/
├── README.md
├── docker-compose.yml
├── .env.example
├── config/
│   ├── risk_weights.yaml
│   ├── rules.yaml
│   └── feature_groups.yaml
├── data/
│   ├── raw/                 # gitignored — Kaggle downloads
│   ├── processed/           # gitignored — parquet
│   └── synthetic/           # injected attack scenarios
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_supervised_model.ipynb
│   ├── 04_anomaly_model.ipynb
│   └── 05_threshold_tuning.ipynb
├── scripts/
│   ├── download_data.py
│   ├── generate_attacks.py
│   ├── train.py
│   └── stream_replay.py
├── src/
│   ├── api/                 # FastAPI routers, schemas, deps
│   ├── features/            # feature builders (causal windows)
│   ├── models/              # xgb wrapper, iforest wrapper, loaders
│   ├── rules/               # rule engine
│   ├── risk/                # fusion + decision policy
│   ├── explain/             # SHAP grouping → reason codes
│   ├── profiles/            # customer profile engine (Redis)
│   └── db/                  # SQLAlchemy models, migrations
├── frontend/                # React app
├── models/                  # gitignored — serialised artefacts
└── tests/
```

---

## 14. Getting started

### Prerequisites
Python 3.11+, Node 20+, Docker Desktop, a Kaggle API token.

### Quick start

```powershell
git clone <repo-url>
cd "credit card fraud detection"

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Copy-Item .env.example .env        # then edit credentials

python scripts/download_data.py    # Kaggle pulls, ~1.2 GB
python scripts/generate_attacks.py # synthetic demo scenarios
python scripts/train.py            # trains XGBoost + IsolationForest → models/

docker compose up -d               # postgres + redis + api + frontend
python scripts/stream_replay.py --speed 60
```

Dashboard → http://localhost:5173 · API docs → http://localhost:8000/docs

---

## 15. Evaluation methodology

**Never report accuracy.** With 0.58% fraud, predicting "genuine" for everything scores 99.42%.

| Metric | Why it matters | Target |
|---|---|---|
| **PR-AUC** (average precision) | The correct headline metric under extreme imbalance | > 0.80 (Sparkov) |
| **Recall @ 1% FPR** | "How much fraud do we catch while annoying only 1% of good customers?" | > 0.75 |
| **Precision @ chosen threshold** | Analyst workload / false-block rate | > 0.60 |
| ROC-AUC | Reported for comparability only; inflated on imbalanced data | > 0.97 |
| **Amount-weighted recall** | ₹ of fraud prevented — the number a bank actually cares about | maximise |
| **p95 latency** | "Real-time decisions" requirement | < 150 ms |

**Validation protocol**
- **Temporal split**, not random: train on the earliest 70%, validate on the next 15%, test on the final 15%. Random splits leak future information through customer aggregates and inflate scores by 10–20 points.
- Threshold selection on validation only; the test set is touched once.
- Per-attack-type recall reported separately for the synthetic set.
- Baseline table: Logistic Regression → Random Forest → XGBoost → XGBoost + Anomaly + Rules, so the hybrid gain is quantified rather than asserted.

---

## 16. Build plan

| Phase | Deliverable | Definition of done |
|---|---|---|
| 1 | Data + EDA | Sparkov & PaySim loaded, class imbalance quantified, fraud patterns charted |
| 2 | Feature engine | 40+ causal features, leakage test passes |
| 3 | Supervised model | XGBoost beats LR/RF baselines on PR-AUC, tracked in MLflow |
| 4 | Anomaly model | Isolation Forest trained on genuine-only, scores normalised |
| 5 | Rule engine + profiles | All 8 rules unit-tested; Redis profile lookups < 5 ms |
| 6 | Risk fusion + API | `/analyze` returns full contract, p95 < 150 ms under load |
| 7 | Explainability | SHAP grouped into 5 business buckets, reason codes rendered |
| 8 | Dashboard | 4 screens, live WebSocket feed |
| 9 | Simulator + stream replay | Judge can change inputs and watch the score move |
| 10 | Docker, tests, docs, deck | `docker compose up` reproduces the demo from a clean machine |

---

## 17. Honest limitations

State these in the presentation. Acknowledging them reads as engineering maturity, not weakness — and it pre-empts exactly the questions judges will ask.

1. **No live bank feed exists publicly.** We replay historical transactions through a genuinely real-time engine. Swapping the producer for Kafka is a small, well-defined change.
2. **Device and multi-country attack data is synthetic.** No public dataset carries device fingerprints alongside geo and merchant fields. Synthetic rows are labelled `attack_type` and their metrics are reported separately from real-data metrics.
3. **Sparkov is itself simulator-generated.** Its fraud patterns are cleaner than reality, so absolute scores are optimistic. Relative model comparisons remain valid.
4. **Concept drift is unaddressed at runtime.** Fraud tactics change weekly; a production system needs scheduled retraining and drift monitoring. We ship the analyst feedback loop (`PATCH /alerts/{id}`) as the foundation but do not automate retraining.
5. **Fairness is not audited.** Features such as `job`, `city_pop` and derived age could encode socio-economic bias. A production deployment would require disparate-impact testing across protected attributes before go-live.
6. **Rule weights are tuned on one dataset.** They would need per-institution recalibration; that is why they live in a YAML file rather than in code.
