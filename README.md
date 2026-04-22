# Pricing Workbench — Databricks Accelerator

End-to-end commercial P&C pricing on Databricks, from raw vendor data to live
pricing decisions with full governance and investigation.

## What this is

A reusable accelerator showing the complete pricing lifecycle on Databricks:

- **Medallion architecture:** External data → Bronze → Silver (DLT) → Gold (UPT)
- **6 pricing models:** GLM frequency/severity, GBM demand/uplift/fraud/retention
- **Derived factors:** urban_score, neighbourhood claim frequency — with per-factor lift attribution
- **Quote Review:** investigation workflow for "why was I charged so much?" — looks up any transaction, shows the three JSON payloads captured from the sales channel and rating engine, replays against today's model, and includes an AI-analyst placeholder for Claude-powered root-cause analysis
- **HITL app:** React + FastAPI for actuarial review and approval
- **Real-time serving:** Online Feature Store + Model Serving with auto feature lookup
- **Full governance:** Audit trail, regulatory PDF export, UC lineage
- **Optional AI agents:** LLM-assisted model selection, DQ monitor, explainability (off by default)

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

# 5. Run pipeline
databricks bundle run ingest_external_data
databricks bundle run build_upt
databricks bundle run train_pricing_models

# 6. Open the app (URL in Databricks Serving UI)
```

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

- **[Talk Track](docs/talk_track.md)** — Executive and technical demo scripts
- **[Data Dictionary](docs/data_dictionary.md)** — Complete table and column reference
- **[About This Demo](docs/about_demo.md)** — Deployment guide, features, disclaimer

## Disclaimer

This is a synthetic demonstration. All company names, policy data, and financial
figures are entirely fictional. No real customer data is used.
