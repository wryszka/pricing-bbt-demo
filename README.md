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

## Deploy to a fresh workspace

Everything is serverless and scale-to-zero. Three commands after a one-time
config edit. No notebook-by-notebook running — the **Full Build** orchestrator
chains the whole populate in dependency order and is idempotent.

**Prerequisites:** Databricks CLI authenticated to the target workspace
(`databricks auth login --host <url> --profile <PROFILE>`), a serverless SQL
warehouse, and Unity Catalog. That's it — no classic compute needed.

```bash
# 1. Point the target at your workspace
#    Edit databricks.yml → targets.v2: host + profile, and its variables:
#    catalog_name, schema_name (default pricing_workbench), warehouse_id.

# 2. Create the app first so it mints its service principal, then bind + wire it.
#    (The app SP is granted CAN_MANAGE_RUN on jobs, so the bundle needs its id;
#     and the bundle can't adopt an app it didn't create — hence create+bind.)
databricks apps create pricing-workbench --profile <PROFILE>
#    → copy the printed service_principal_client_id into
#      databricks.yml → targets.v2.variables.app_service_principal_id
databricks bundle deployment bind pricing_workbench pricing-workbench \
    --target v2 --profile <PROFILE> --auto-approve

# 3. Deploy the bundle (all jobs + the app resource)
./deploy.sh v2          # builds frontend, syncs app.yaml, deploys bundle + app

# 4. Populate everything — ONE job, ~30-40 min, all serverless
databricks bundle run full_build --target v2 --profile <PROFILE>
#    data → UPT (incl. real 1.5M-row UK postcode enrichment) → 4 champions →
#    aliases → rating config + release rate-book → inference/bias backfills →
#    shadow-pricing → commercial rating-engine endpoint → governance + chat
#    agents (self-mint their SQL token) → app-SP grants → metadata/tags.
```

Open the app (URL from `databricks apps get pricing-workbench`). The commercial
rating engine prices live (scale-to-zero); the agents answer from real tables.

**Notes**
- The agents self-provision a 90-day PAT for their SQL tools on deploy. If your
  workspace disables PATs, inject `AGENT_TOKEN`/`AGENT_HOST` on the two agent
  endpoints instead.
- **Genie spaces + the Modelling-Mart dashboard** are workspace assets you
  create once and wire via `app.v2.yaml` (`GENIE_SPACE_ID`,
  `GENIE_QUOTE_SPACE_ID`, `MART_DASHBOARD_ID`); until set, those tabs stay
  hidden.
- The optional **live-serving tier** (Lakebase online store + route-optimized
  scorer + QPS load tester) is *not* part of Full Build — it costs money while
  up. Arm it on demand with `databricks bundle run live_pricing_provision` and
  tear it down with `live_pricing_teardown`.

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
