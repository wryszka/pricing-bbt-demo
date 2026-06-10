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

> **Order matters.** Deploy the bundle, run the data pipeline, *then* deploy the
> app + grants. The post-deploy grant job needs the schema and trained models to
> already exist (it grants the app's service principal and sets the initial
> champions), so it must run **after** the pipeline.

### 0. Prerequisites

- A Databricks workspace with **serverless compute** + **Unity Catalog**
- A **SQL warehouse** in that workspace (note its id)
- Databricks CLI v0.230+ authenticated to the workspace as a named profile:
  ```bash
  databricks auth login --host https://<your-workspace-host> --profile <your-profile>
  ```
- Node 18+ (the app frontend is built locally by `deploy.sh`)

### 1. Clone

```bash
git clone https://github.com/wryszka/pricing-workbench.git
cd pricing-workbench
```

### 2. Add a deploy target

`databricks.yml` ships `prod`/`dev` (maintainer targets - do not deploy to them).
Add your own target alongside them (copy the `pinchu` target as a template):

```yaml
  mytarget:
    mode: development
    workspace:
      host: https://<your-workspace-host>
      profile: <your-profile>
      root_path: /Workspace/Users/${workspace.current_user.userName}/.bundle/${bundle.name}/${bundle.target}
    variables:
      catalog_name: <your_catalog>      # created by setup_demo if it does not exist
      warehouse_id: "<your-warehouse-id>"
      # leave app SP / Genie / dashboard blank - they are filled in later
```

Then create `src/app/app.mytarget.yaml` (copy `app.pinchu.yaml`) and set
`CATALOG_NAME`, `WAREHOUSE_ID`, and `BUNDLE_NOTEBOOKS_BASE`
(`/Workspace/Users/<you>/.bundle/pricing-upt-demo/mytarget/files/src/04_models`).
Leave the Genie / dashboard ids blank - the UI hides those panels until they are set.

### 3. Deploy the bundle (creates the jobs)

```bash
databricks bundle deploy --target mytarget --profile <your-profile>
```

### 4. Run the data pipeline (in order)

```bash
T=mytarget; P=<your-profile>
databricks bundle run setup_demo                 --target $T --profile $P   # schema + tables + test data
databricks bundle run build_postcode_enrichment  --target $T --profile $P   # ~2-5 min - ONSPD + IMD download
databricks bundle run ingest_external_data        --target $T --profile $P   # bronze → silver
databricks bundle run build_upt                   --target $T --profile $P   # derive_factors → UPT → feature_catalog
databricks bundle run production_training         --target $T --profile $P   # GLMs + GBMs + challenger comparison
```

### 5. Deploy the app + grant its service principal

```bash
./deploy.sh mytarget <your-profile>
```

This builds the frontend, deploys the bundle + app, then runs
`grant_app_permissions` - which grants the app's auto-minted service principal
the Unity Catalog / experiment access it needs and sets the initial `champion`
alias per model family. **Without this step the app's data tabs return errors**
(the SP starts with no grants). The job is idempotent - safe to re-run.

> Pass your CLI profile as the second argument for any custom target. The
> built-in `dev` / `prod` / `pinchu` targets have default profiles and need no
> second argument.

### 6. Open the app

The app URL is printed by `deploy.sh` (and in the Databricks **Apps** UI).
To demo sub-10ms renewal scoring, promote the Feature Store to the online store
(Lakebase) from the **Modelling Mart** tab.

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

- **[Talk Track](docs/talk_track.md)** — Executive and technical demo scripts
- **[Data Dictionary](docs/data_dictionary.md)** — Complete table and column reference
- **[About This Demo](docs/about_demo.md)** — Deployment guide, features, disclaimer

## Disclaimer

This is a synthetic demonstration. All company names, policy data, and financial
figures are entirely fictional. No real customer data is used.
