# Pricing Workbench - AXA edition: manual deployment guide

This is a slimmed build of the Pricing Workbench accelerator scoped to **data
preparation, the modelling mart, and model development**. Live serving, model
deployment, the pricing engine, model governance, pricing AI and the add-ons are
intentionally excluded.

It deploys with the standard Databricks CLI and UI - **no Claude Code or any AI
tooling is required to deploy it.** Everything below is copy-paste shell + a few
UI clicks.

## What this build includes / excludes

| Included | Excluded |
|---|---|
| Data Ingestion (internal book, vendor feeds, public reference data, DQ + approval gate) | Live serving / online store / Lakebase |
| Modelling Mart (feature table, factor catalog, AI/BI Genie) | Model Deployment, Pricing Engine |
| Model Development (reference notebooks, model library) | Model Factory, Model Governance, Pricing AI, Add-ons |
| | The motor dataset (existed only for live-pricing) |

## 0. Prerequisites (in the AXA workspace)

- A Databricks workspace with **serverless compute** and **Unity Catalog** enabled.
- A **SQL warehouse** (note its id - Compute -> SQL warehouses -> the warehouse -> copy the id from the URL or the connection details).
- A **Unity Catalog catalog** the deployer can create schemas in (e.g. `pricing_workbench`), plus `CREATE SCHEMA` / `USE CATALOG` on it.
- **Databricks CLI v0.230+** on the deployer's machine, authenticated to the workspace:
  ```bash
  databricks auth login --host https://<axa-workspace-host> --profile axa
  ```
- **Node 18+** (the app frontend is built locally before deploy).
- Optional: AI/BI Genie enabled if you want the Modelling Mart Genie panel (it is hidden automatically if unavailable).

## 1. Get the code

```bash
git clone -b axa/data-prep-mart https://github.com/wryszka/pricing-workbench.git
cd pricing-workbench
```

## 2. Configure a deploy target

Add a target to `databricks.yml` (copy the shape of the `pinchu` target). Example:

```yaml
  axa:
    mode: development
    workspace:
      host: https://<axa-workspace-host>
      profile: axa
      root_path: /Workspace/Users/${workspace.current_user.userName}/.bundle/${bundle.name}/${bundle.target}
    variables:
      catalog_name: pricing_workbench      # created by setup_demo if it does not exist
      warehouse_id: "<axa-warehouse-id>"
      # leave app SP / Genie ids blank - filled in by the post-deploy job
```

Create `src/app/app.axa.yaml` (copy `src/app/app.pinchu.yaml`) and set:
- `CATALOG_NAME` = your catalog
- `WAREHOUSE_ID` = your warehouse id
- `BUNDLE_NOTEBOOKS_BASE` = `/Workspace/Users/<deployer>/.bundle/pricing-upt-demo/axa/files/src/04_models`
- leave `GENIE_SPACE_ID` / `GENIE_QUOTE_SPACE_ID` blank (the post-deploy job fills them; blank simply hides the Genie panel).

## 3. Deploy the bundle (creates the jobs)

```bash
databricks bundle deploy --target axa --profile axa
```

## 4. Run the data pipeline, in this order

The order matters: the mart depends on ingestion, models depend on the mart, and
the post-deploy grant job (step 6) needs the schema and models to already exist.

```bash
T=axa; P=axa
databricks bundle run setup_demo                --target $T --profile $P   # schema + tables + synthetic data
databricks bundle run build_postcode_enrichment --target $T --profile $P   # ~2-5 min, downloads UK public data
databricks bundle run ingest_external_data       --target $T --profile $P   # bronze -> silver
databricks bundle run build_upt                  --target $T --profile $P   # derive factors -> mart -> factor catalog
databricks bundle run production_training         --target $T --profile $P   # GLMs + GBMs registered to Unity Catalog
```

## 5. Deploy the app

```bash
databricks apps deploy pricing-workbench \
  --source-code-path /Workspace/Users/<deployer>/.bundle/pricing-upt-demo/axa/files/src/app \
  --profile axa
```

(If the app reports it is not running, start it first: `databricks apps start pricing-workbench --profile axa`, then re-run the deploy. The frontend is built and the bundle redeployed for you if you use `./deploy.sh axa axa` instead of steps 3 and 5 - it does the npm build, bundle deploy, app start, app deploy, and the post-deploy jobs in one go.)

## 6. Run the post-deploy bootstrap (grants + Genie)

The app runs as its own auto-minted service principal that starts with no Unity
Catalog access, so this step is required - without it the data tabs error.

```bash
databricks bundle run grant_app_permissions --target axa --profile axa   # grants the app SP UC access
databricks bundle run create_genie_spaces   --target axa --profile axa   # optional: creates the mart Genie space
```

Both are idempotent and safe to re-run. `create_genie_spaces` is optional - skip
it if Genie is not available in the AXA workspace; the panel simply stays hidden.

## 7. Open the app

The app URL is shown in the Databricks **Apps** UI (and printed by `deploy.sh`).
You should see four tabs: **Home, Data Ingestion, Modelling Mart, Model
Development**.

## Shortcut

Steps 3, 5 and 6 are wrapped by the deploy helper:

```bash
./deploy.sh axa axa
```

Run the data pipeline (step 4) once before the first `deploy.sh`, since the
post-deploy grant job needs the schema and models to exist.

## Notes for the AXA environment

- **No Claude Code needed.** Every step here is the Databricks CLI or UI.
- **AI features are optional.** The Modelling Mart Genie panel and the
  explainability assistant call AI services; if those are not enabled in the AXA
  sandbox, the rest of the app still works and those panels degrade quietly.
- **No live serving / Lakebase / model endpoints** are created by this build, so
  there is nothing to tear down and no serving cost.
