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

## 4. Run the data pipeline (one command)

`setup_all` chains setup -> enrichment + ingestion -> mart -> training in the
correct order (~20 min). If a step fails, use "Repair run" on the run page to
retry from that task.

```bash
databricks bundle run setup_all --target axa --profile axa
```

## 5. Deploy the app + grant its service principal

```bash
./deploy.sh axa axa
```

This builds the frontend, deploys the app, starts it, then runs
`grant_app_permissions` (grants the auto-minted app SP its Unity Catalog,
experiment, job-run and Genie access - **without this the data tabs error**) and
`create_genie_spaces` (creates the Modelling Mart Genie space; skipped quietly if
Genie is unavailable). Both are idempotent.

## 6. Open the app

The app URL is printed by `deploy.sh` (and shown in the Databricks **Apps** UI).
You should see four tabs: **Home, Data Ingestion, Modelling Mart, Model
Development**.

## Flow summary

```
databricks bundle deploy --target axa --profile axa   # 3. create jobs
databricks bundle run setup_all --target axa --profile axa   # 4. build data (~20 min)
./deploy.sh axa axa                                    # 5. app + grants + Genie
```

## Notes for the AXA environment

- **No Claude Code needed.** Every step here is the Databricks CLI or UI.
- **AI features are optional.** The Modelling Mart Genie panel and the
  explainability assistant call AI services; if those are not enabled in the AXA
  sandbox, the rest of the app still works and those panels degrade quietly.
- **No live serving / Lakebase / model endpoints** are created by this build, so
  there is nothing to tear down and no serving cost.
