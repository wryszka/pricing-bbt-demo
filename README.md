# Pricing Workbench — Databricks Accelerator

End-to-end commercial P&C pricing on Databricks, laid out the way a real pricing
team actually operates — not abstracted into a "data + model" black box.

## The flow, literally

```
External data ─ enrichment ─┐
  (ONSPD + IMD + market +   │
   geo + credit bureau)     ├─→ Quote request ─→ Pricing model ─→ Quote response
                            │     (Jane)          (freq × severity)
                            │         │
                            │         └─ if bound ─→ Policy ─ accrues ─→ Claims
                            │                           │
                            │                           └─→ Training feature store
                            └───────────────────────────────┘        │
                                                                    retrain
```

- **Training feature store** = policy-keyed Delta table, 50K rows with features at policy inception + observed outcomes. What the GLMs and GBMs learn from. Backed by a promotable online store (Lakebase) for sub-10ms lookups at serving time.
- **Quote stream** = the serving-time feature shape. Each quote is captured as three JSON payloads in Unity Catalog — sales request, rating-engine call, rating-engine response. Same rows train the Demand GBM.
- **External data** = joined at both quote and policy time. Includes the real 1.5M English postcode enrichment (ONSPD + IMD 2019 + ONS RUC + coastal flags) so the feature catalog has real lineage, not synthetic stubs.
- **Feature catalog** = one row per feature in the UPT, with source tables, transformation, owner, regulatory/PII flags. Foundation for feature-level lineage and audit bolt-ons.

## What's in the app

- **External Data** — 4 datasets visible, including the real UK postcode enrichment. HITL approval flow for the synthetic ones.
- **Quote Review** — transaction lookup, JSON payload view, simulated replay, Claude-backed AI Analyst (placeholder).
- **Feature Store** — offline Delta + online Lakebase status, promote / pause buttons, **feature catalog** with per-feature provenance.
- **Model Development** — notebook inventory + challenger panel showing Gini lift per real-UK factor.
- **Model Factory** — 50-spec GLM factory, leaderboard, governance PDF per model.
- **Model Deployment** — two scoring paths: new-business (feature vector direct) and renewal (FeatureLookup via online store).
- **Quote Review Analytics + Genie** — broader pattern analysis across the quote stream.
- **Monitoring, Governance** — data freshness, DQ, immutable audit log, regulatory export.

## Notebook track for data scientists / actuaries

`src/new_data_impact/` — six standalone notebooks that answer *"does adding real external data actually make pricing models better?"* Standard vs enriched freq+sev GLMs on a 200K portfolio, Claude review agent, governance PDF. Hero numbers: Gini 0.11 → 0.25, Deviance Explained 1.0% → 5.3%.

## Quick Start

```bash
# 1. Clone
git clone https://github.com/wryszka/pricing-workbench.git
cd pricing-workbench

# 2. Configure (edit databricks.yml with your workspace)
#    Change catalog_name and workspace host

# 3. Deploy
databricks bundle deploy

# 4. Run setup (creates tables + test data)
databricks bundle run setup_demo

# 5. Build the real UK postcode enrichment (~2-5 min — ONSPD + IMD download)
databricks bundle run build_postcode_enrichment

# 6. Run pipeline
databricks bundle run ingest_external_data    # bronze → silver
databricks bundle run build_upt                # derive_factors → UPT → feature_catalog
databricks bundle run train_pricing_models     # GLMs + GBMs + challenger comparison

# 7. Sync the notebook track to your Workspace Home folder
./scripts/sync_notebooks.sh
#   → /Workspace/Users/<you>/pricing-workbench/new_data_impact/

# 8. Open the app (URL in Databricks Serving UI)
#    Promote the Feature Store to the online store (Lakebase) from the
#    "Feature Store" tab when you want to demo sub-10ms renewal scoring.
```

## Two tracks

| Track | For | Entry point |
|---|---|---|
| **Pricing Workbench app** | Execs, underwriters, operators, actuaries | React app — sidebar: Data Ingestion, Model Factory, Quote Review, Governance, etc. |
| **New Data Impact study** (`src/new_data_impact/`) | Data scientists, actuaries, governance | 6 notebooks — build enrichment → train standard vs enriched models → governance PDF → AI agent |

Both tracks share the same Unity Catalog schema (`pricing_upt`). The study's derivative tables are prefixed `impact_*` so they group together in Catalog Explorer; the reusable `postcode_enrichment` reference is used by both tracks.

## Architecture

```
External Data → Volume → Bronze → DLT (expectations) → Silver
                                                          ↓
Internal Data (policies, claims, quotes) ───────→ Unified Pricing Table (Gold)
                                                          ↓
              Feature Lookup → Train 6 Models → MLflow → UC Registry
                                                          ↓
              Online Store (Lakebase) → Model Serving → REST API
                                                          ↓
              GOVERNANCE: UC Lineage │ Audit Log │ Time Travel │ DQ Monitoring
```

## Prerequisites

- Databricks workspace with **serverless compute**
- Unity Catalog enabled
- Databricks CLI v0.200+

## Repository Structure

```
├── databricks.yml              # DABs configuration
├── resources/                  # Job and pipeline definitions
├── src/
│   ├── 00_setup/               # Data generation + overview
│   ├── 01_ingestion/           # CSV → Bronze
│   ├── 02_silver/              # DLT expectations + cleansing
│   ├── 03_gold/                # Unified Pricing Table build
│   ├── 04_models/              # 6 model training notebooks + AI agent
│   ├── 05_use_cases/           # Shadow pricing, PIT, enriched pricing
│   ├── 06_model_factory/       # Automated training + evaluation
│   ├── 07_serving/             # Online store + model endpoints
│   ├── 08_governance/          # Dashboard + regulatory export
│   ├── app/                    # FastAPI + React HITL application
│   └── utils/                  # Shared audit + diagram utilities
└── docs/
    ├── talk_track.md           # Executive (30 min) + Technical (60 min)
    ├── data_dictionary.md      # Every table and column documented
    └── about_demo.md           # Deployment guide + feature list
```

## Documentation

- **[Demo Runbook (Google Doc)](https://docs.google.com/document/d/1VHVMrbwo1D2Gfl2NKnKJzosBlS-hltcFZ9guvBejUkM/edit)** — The canonical presenter guide: pre-demo checklist, 12-step demo flow, Q&A cheat sheet (internal)
- **[Talk Track](docs/talk_track.md)** — Executive and technical demo scripts
- **[Data Dictionary](docs/data_dictionary.md)** — Complete table and column reference
- **[About This Demo](docs/about_demo.md)** — Deployment guide, features, disclaimer

## Disclaimer

This is a synthetic demonstration. All company names, policy data, and financial
figures are entirely fictional. No real customer data is used.
