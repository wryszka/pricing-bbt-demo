# About This Demo

## Disclaimer

This is a **synthetic demonstration** built by the Databricks Field Engineering
team. All company names (Bricksurance SE), policy data, financial figures,
and model outputs are entirely fictional and generated for illustrative purposes
only. **No real customer data is used.**

The models are trained on synthetic data and should not be used for actual
pricing decisions. They are designed to demonstrate Databricks platform
capabilities, not actuarial best practice.

---

## Databricks Features Demonstrated

| Feature | Where it's used |
|---------|----------------|
| **Unity Catalog** | Schema governance, table tags, column comments, model registry |
| **Delta Lake** | Medallion architecture (bronze/silver/gold), Time Travel, MERGE |
| **Delta Live Tables (DLT)** | Data quality expectations, silver layer cleansing |
| **Volumes** | External vendor file landing zone |
| **MLflow** | Experiment tracking, model logging, metrics, artifacts |
| **Feature Engineering Client** | FeatureLookup, training sets, fe.log_model() |
| **Online Feature Store** | Lakebase-powered sub-10ms key-value lookups |
| **Mosaic AI Model Serving** | Serverless endpoints with auto feature lookup |
| **Foundation Model API** | AI agent for model selection (Claude via FMAPI) |
| **Databricks Apps** | FastAPI + React HITL application |
| **Genie Space** | Natural language data exploration |
| **Serverless Compute** | All jobs run on serverless (environment v5) |
| **Databricks Asset Bundles** | Infrastructure-as-code deployment |

---

## Architecture

```
External Data → Volume → Bronze → DLT → Silver → HITL Review → Gold (UPT)
                                                                    ↓
                                                            Model Training (6 models)
                                                                    ↓
                                                    MLflow → UC Registry → Actuary Review
                                                                    ↓
                                            Online Store (Lakebase) → Model Serving → REST API
```

Governance layer: Unity Catalog lineage + audit_log + Delta Time Travel

---

## How to Deploy on Your Own Workspace

### Prerequisites

- Databricks workspace with **serverless compute** enabled
- Unity Catalog with **CREATE CATALOG** or an existing catalog
- **Databricks CLI** v0.200+ installed locally
- **Databricks Asset Bundles** (included in CLI)
- GitHub access to clone the repo

### Step-by-step

1. **Clone the repo:**
   ```bash
   git clone https://github.com/wryszka/pricing-workbench.git
   cd pricing-workbench
   ```

2. **Configure your workspace:**
   Edit `databricks.yml`:
   ```yaml
   variables:
     catalog_name:
       default: your_catalog_name    # Change this
     schema_name:
       default: pricing_upt          # Or your preferred schema

   workspace:
     host: https://your-workspace.cloud.databricks.com
     profile: DEFAULT                 # Or your CLI profile
   ```

3. **Deploy:**
   ```bash
   databricks bundle deploy
   ```

4. **Run setup:**
   ```bash
   databricks bundle run setup_demo
   ```

5. **Run ingestion + gold build:**
   ```bash
   databricks bundle run ingest_external_data
   databricks bundle run build_upt
   ```

6. **Train models:**
   ```bash
   databricks bundle run train_pricing_models
   databricks bundle run train_supplementary_models
   ```

7. **Start the app:**
   The Databricks App deploys automatically with `bundle deploy`. Check the
   Serving UI for the app URL.

8. **Optional — Online store + serving:**
   ```bash
   databricks bundle run setup_online_store
   databricks bundle run deploy_model_endpoint
   ```

### Portability

All catalog/schema references come from DAB variables. To deploy on a different
workspace with a different catalog:

```bash
# One-liner to change catalog everywhere
sed -i 's/lr_serverless_aws_us_catalog/your_catalog/g' databricks.yml
```

---

## How to Extend Into Production

This accelerator demonstrates concepts. To move to production:

1. **Replace synthetic data** with real policy/claims/quote extracts
2. **Connect real vendors** via Lakeflow Connect (SFTP, APIs, Marketplace)
3. **Tune DLT expectations** to match your actual data contracts
4. **Replace proxy pricing formula** with your real GLM/GBM models
5. **Configure ABAC** for multi-business-unit access control
6. **Add CI/CD** via GitHub Actions + DABs for deployment pipeline
7. **Enable monitoring** with Lakehouse Monitoring on the UPT
8. **Scale up** online store capacity (CU_1 → CU_4/CU_8) for production traffic
9. **Integrate with existing tools** (Radar, Earnix, Guidewire) via REST APIs
