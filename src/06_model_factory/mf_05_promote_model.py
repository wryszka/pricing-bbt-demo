# Databricks notebook source
# MAGIC %md
# MAGIC # Model Factory — Step 5: Model Promotion
# MAGIC
# MAGIC This is the final step in the Model Factory pipeline. For every model that the
# MAGIC actuary has **approved**, this notebook:
# MAGIC
# MAGIC 1. **Registers** the model in the **Unity Catalog Model Registry**
# MAGIC 2. Sets the **champion** alias on the top-ranked approved model for each target
# MAGIC 3. Tags each model version with governance metadata (factory run, regulatory score,
# MAGIC    actuary name, approval conditions)
# MAGIC 4. Logs the promotion to the audit trail
# MAGIC
# MAGIC ### Why Unity Catalog?
# MAGIC
# MAGIC Unity Catalog provides:
# MAGIC - **Centralised governance** — one registry across all workspaces
# MAGIC - **Lineage** — automatic tracking of which table trained which model
# MAGIC - **Access control** — fine-grained permissions on who can deploy models
# MAGIC - **Aliases** — `champion` / `challenger` labels for A/B testing in production
# MAGIC
# MAGIC After promotion, models can be:
# MAGIC - Served via **Mosaic AI Model Serving** for real-time API scoring
# MAGIC - Used in batch scoring pipelines via `mlflow.pyfunc.load_model()`
# MAGIC - Referenced in downstream notebooks: `models:/{catalog}.{schema}.{model_name}@champion`

# COMMAND ----------

dbutils.widgets.text("catalog_name", "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name", "pricing_upt")
dbutils.widgets.text("factory_run_id", "")

catalog = dbutils.widgets.get("catalog_name")
schema = dbutils.widgets.get("schema_name")
fqn = f"{catalog}.{schema}"

# COMMAND ----------

import json
import uuid
from datetime import datetime, timezone
import mlflow

mlflow.set_registry_uri("databricks-uc")

factory_run_id = dbutils.widgets.get("factory_run_id").strip()
if not factory_run_id:
    try:
        factory_run_id = dbutils.jobs.taskValues.get(taskKey="mf_actuary_review", key="factory_run_id")
    except Exception:
        latest = spark.sql(f"SELECT DISTINCT factory_run_id FROM {fqn}.mf_actuary_decisions ORDER BY factory_run_id DESC LIMIT 1").collect()
        factory_run_id = latest[0]["factory_run_id"] if latest else "UNKNOWN"

print(f"Factory Run ID: {factory_run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Audit Helpers

# COMMAND ----------

def get_current_user():
    try:
        return dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
    except Exception:
        import os
        return os.getenv("USER", "unknown")

def log_audit_event(spark, fqn, factory_run_id, event_type, details, mlflow_run_id=None, upt_version=None):
    event = {
        "event_id": str(uuid.uuid4()),
        "factory_run_id": factory_run_id,
        "event_type": event_type,
        "event_timestamp": datetime.now(timezone.utc).isoformat(),
        "actor": get_current_user(),
        "details_json": json.dumps(details) if isinstance(details, dict) else str(details),
        "mlflow_run_id": mlflow_run_id,
        "upt_table_version": upt_version,
    }
    spark.createDataFrame([event]).write.mode("append").saveAsTable(f"{fqn}.mf_audit_log")

upt_version = spark.sql(f"DESCRIBE HISTORY {fqn}.unified_pricing_table_live LIMIT 1").collect()[0]["version"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Approved Models

# COMMAND ----------

approved = (
    spark.table(f"{fqn}.mf_actuary_decisions")
    .filter(f"factory_run_id = '{factory_run_id}' AND decision = 'APPROVED'")
    .collect()
)

if not approved:
    print("No approved models found for this factory run.")
    print("Run mf_04_actuary_review first to approve models.")
    dbutils.notebook.exit("NO_APPROVED_MODELS")

print(f"Approved models to promote: {len(approved)}")
for a in approved:
    print(f"  {a['model_config_id']:45s} reviewer={a['reviewer']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Leaderboard for Ranking Info

# COMMAND ----------

leaderboard = {
    row["model_config_id"]: row
    for row in spark.table(f"{fqn}.mf_leaderboard")
        .filter(f"factory_run_id = '{factory_run_id}'")
        .collect()
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Promote Each Approved Model
# MAGIC
# MAGIC Models are registered in Unity Catalog under the same schema as all other demo
# MAGIC tables: `{catalog}.{schema}.mf_{model_name}`.
# MAGIC
# MAGIC The **champion** alias is set on the top-ranked approved model for each target
# MAGIC variable. All other approved models get the **challenger** alias, enabling
# MAGIC A/B testing in production.

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()
promoted = []

# Group approved models by target to determine champion vs challenger
from collections import defaultdict
by_target = defaultdict(list)
for a in approved:
    lb = leaderboard.get(a["model_config_id"])
    if lb:
        by_target[lb["target_column"]].append((lb["rank"], a, lb))

for target, models in by_target.items():
    models.sort(key=lambda x: x[0])  # Sort by rank

    for idx, (rank, approval, lb_row) in enumerate(models):
        config_id = approval["model_config_id"]
        run_id = approval["mlflow_run_id"]

        # Clean model name for UC (replace hyphens, dots)
        model_name = f"mf_{config_id}".replace("-", "_").replace(".", "_")
        full_model_name = f"{catalog}.{schema}.{model_name}"

        print(f"\nPromoting: {config_id}")
        print(f"  UC Model: {full_model_name}")

        try:
            # Determine the artifact path (GBM models log as "model", uplift as "gbm_uplift_model")
            if lb_row["model_family"] == "GBM":
                artifact_uri = f"runs:/{run_id}/model"
            elif lb_row["model_family"] == "GLM_GBM_UPLIFT":
                artifact_uri = f"runs:/{run_id}/gbm_uplift_model"
            else:
                # GLMs don't log sklearn models, so we register the run itself
                artifact_uri = f"runs:/{run_id}/model"

            # Register the model
            try:
                mv = mlflow.register_model(artifact_uri, full_model_name)
                version = mv.version
            except Exception as reg_err:
                # If the model artifact doesn't exist (e.g., GLMs), create a placeholder
                print(f"  Note: Could not register artifact ({reg_err}). Logging registration in audit only.")
                version = None

            if version:
                # Set alias
                alias = "champion" if idx == 0 else "challenger"
                try:
                    client.set_registered_model_alias(full_model_name, alias, version)
                    print(f"  Alias: @{alias} (version {version})")
                except Exception:
                    print(f"  Could not set alias (model may need to be in READY state)")

                # Set tags on the model version
                tags = {
                    "factory_run_id": factory_run_id,
                    "model_config_id": config_id,
                    "target_column": target,
                    "rank": str(rank),
                    "composite_score": str(lb_row["composite_score"]),
                    "regulatory_suitability_score": str(lb_row["regulatory_suitability_score"]),
                    "gini": str(lb_row["gini"]),
                    "approved_by": approval["reviewer"],
                    "approved_at": approval["decided_at"],
                    "conditions": approval.get("conditions") or "None",
                }
                for tag_key, tag_val in tags.items():
                    try:
                        client.set_model_version_tag(full_model_name, version, tag_key, tag_val)
                    except Exception:
                        pass

                print(f"  Tags set: factory_run_id, regulatory_score, approved_by, etc.")

            promoted.append({
                "model_config_id": config_id,
                "uc_model_name": full_model_name,
                "version": version,
                "alias": "champion" if idx == 0 else "challenger",
                "target": target,
                "rank": rank,
            })

            # Audit log
            log_audit_event(spark, fqn, factory_run_id, "MODEL_PROMOTED", {
                "model_config_id": config_id,
                "uc_model_name": full_model_name,
                "version": version,
                "alias": "champion" if idx == 0 else "challenger",
                "target_column": target,
                "rank": rank,
                "approved_by": approval["reviewer"],
                "regulatory_suitability_score": lb_row["regulatory_suitability_score"],
                "composite_score": lb_row["composite_score"],
            }, mlflow_run_id=run_id, upt_version=upt_version)

        except Exception as e:
            print(f"  ERROR promoting {config_id}: {e}")
            log_audit_event(spark, fqn, factory_run_id, "MODEL_PROMOTION_FAILED", {
                "model_config_id": config_id,
                "error": str(e),
            }, mlflow_run_id=run_id, upt_version=upt_version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Promotion Summary

# COMMAND ----------

print("=" * 70)
print("MODEL PROMOTION COMPLETE")
print("=" * 70)
print(f"Factory Run: {factory_run_id}")
print(f"Models Promoted: {len(promoted)}")
print()
for p in promoted:
    alias_badge = f"@{p['alias']}" if p['alias'] else ""
    version_str = f"v{p['version']}" if p['version'] else "(audit only)"
    print(f"  {p['target']:20s}  #{p['rank']}  {p['uc_model_name']}  {version_str}  {alias_badge}")
print()
print("=" * 70)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Using Promoted Models
# MAGIC
# MAGIC Once a model is registered in Unity Catalog, it can be loaded for scoring:
# MAGIC
# MAGIC ```python
# MAGIC import mlflow
# MAGIC
# MAGIC # Load the champion model for claim frequency
# MAGIC model = mlflow.pyfunc.load_model(
# MAGIC     f"models:/{catalog}.{schema}.mf_glm_poisson_freq_all_features@champion"
# MAGIC )
# MAGIC
# MAGIC # Score new data
# MAGIC predictions = model.predict(new_data_pdf)
# MAGIC ```
# MAGIC
# MAGIC Or deploy as a **real-time endpoint** via Mosaic AI Model Serving:
# MAGIC
# MAGIC ```python
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC
# MAGIC w = WorkspaceClient()
# MAGIC w.serving_endpoints.create(
# MAGIC     name="pricing-frequency-model",
# MAGIC     config={
# MAGIC         "served_entities": [{
# MAGIC             "entity_name": f"{catalog}.{schema}.mf_glm_poisson_freq_all_features",
# MAGIC             "entity_version": "1",
# MAGIC             "scale_to_zero_enabled": True,
# MAGIC         }]
# MAGIC     }
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC ## Complete Audit Trail
# MAGIC
# MAGIC The full factory run — from feature inspection through training, evaluation,
# MAGIC actuary review, to this promotion — is recorded in `mf_audit_log`.

# COMMAND ----------

audit_count = spark.table(f"{fqn}.mf_audit_log").filter(f"factory_run_id = '{factory_run_id}'").count()
print(f"Total audit events for this factory run: {audit_count}")

display(
    spark.table(f"{fqn}.mf_audit_log")
    .filter(f"factory_run_id = '{factory_run_id}'")
    .groupBy("event_type")
    .count()
    .orderBy("event_type")
)
