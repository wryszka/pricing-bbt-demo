# Databricks notebook source
# MAGIC %md
# MAGIC # Model Factory — Step 4: Actuary Review
# MAGIC
# MAGIC This notebook is the **human-in-the-loop gate** for the Model Factory.
# MAGIC The actuary reviews the leaderboard, inspects individual model cards,
# MAGIC and records an approval or rejection decision for each model.
# MAGIC
# MAGIC ### What the Actuary Sees
# MAGIC 1. **Leaderboard** — All models ranked by composite score with recommended actions
# MAGIC 2. **Model Cards** — For the top 5 models: coefficients (GLM) or feature importances (GBM), metrics, prediction distributions
# MAGIC 3. **Decision Form** — Approve, reject, or defer each model with notes and conditions
# MAGIC
# MAGIC ### Regulatory Compliance
# MAGIC - Every decision is recorded with: **who** (reviewer), **when** (timestamp),
# MAGIC   **what** (model ID + decision), **why** (notes), and **conditions** (e.g.,
# MAGIC   "approved subject to 6-month monitoring period")
# MAGIC - Decisions are immutable — stored in an append-only table alongside the audit log
# MAGIC - The actuary can query past decisions across factory runs for governance reporting

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

factory_run_id = dbutils.widgets.get("factory_run_id").strip()
if not factory_run_id:
    try:
        factory_run_id = dbutils.jobs.taskValues.get(taskKey="mf_evaluation", key="factory_run_id")
    except Exception:
        latest = spark.sql(f"SELECT DISTINCT factory_run_id FROM {fqn}.mf_leaderboard ORDER BY factory_run_id DESC LIMIT 1").collect()
        factory_run_id = latest[0]["factory_run_id"] if latest else "UNKNOWN"

try:
    dbutils.jobs.taskValues.set(key="factory_run_id", value=factory_run_id)
except Exception:
    pass

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
    from pyspark.sql.types import StructType, StructField, StringType
    schema = StructType([
        StructField("event_id", StringType()),
        StructField("factory_run_id", StringType()),
        StructField("event_type", StringType()),
        StructField("event_timestamp", StringType()),
        StructField("actor", StringType()),
        StructField("details_json", StringType()),
        StructField("mlflow_run_id", StringType()),
        StructField("upt_table_version", StringType()),
    ])
    event = (
        str(uuid.uuid4()),
        factory_run_id,
        event_type,
        datetime.now(timezone.utc).isoformat(),
        get_current_user(),
        json.dumps(details) if isinstance(details, dict) else str(details),
        mlflow_run_id or "",
        str(upt_version) if upt_version is not None else "",
    )
    spark.createDataFrame([event], schema=schema).write.mode("append").saveAsTable(f"{fqn}.mf_audit_log")

upt_version = spark.sql(f"DESCRIBE HISTORY {fqn}.unified_pricing_table_live LIMIT 1").collect()[0]["version"]
reviewer = get_current_user()
print(f"Reviewer: {reviewer}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ensure Decisions Table Exists

# COMMAND ----------

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {fqn}.mf_actuary_decisions (
        decision_id STRING,
        factory_run_id STRING,
        model_config_id STRING,
        mlflow_run_id STRING,
        decision STRING,
        reviewer STRING,
        reviewer_notes STRING,
        decided_at STRING,
        regulatory_sign_off BOOLEAN,
        conditions STRING
    )
""")

print("Actuary decisions table ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Full Leaderboard
# MAGIC
# MAGIC Review all models sorted by rank. The `recommended_action` column indicates the
# MAGIC factory's automated suggestion, but the **actuary has final authority**.

# COMMAND ----------

display(
    spark.table(f"{fqn}.mf_leaderboard")
    .filter(f"factory_run_id = '{factory_run_id}'")
    .select("rank", "model_config_id", "model_family", "model_type",
            "target_column", "feature_count",
            "rmse", "gini", "lift_decile1", "psi",
            "regulatory_suitability_score", "composite_score",
            "recommended_action")
    .orderBy("target_column", "rank")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Cards — Top 5 Models
# MAGIC
# MAGIC Detailed view of the best-performing models. For GLMs, you'll see the
# MAGIC coefficient table with relativities (multiplicative rating factors). For GBMs,
# MAGIC you'll see feature importance rankings.

# COMMAND ----------

import mlflow
import numpy as np
import pandas as pd

mlflow.set_registry_uri("databricks-uc")
client = mlflow.tracking.MlflowClient()

# Get top models
top_models = (
    spark.table(f"{fqn}.mf_leaderboard")
    .filter(f"factory_run_id = '{factory_run_id}'")
    .orderBy("target_column", "rank")
    .limit(15)  # Top 5 per target
    .collect()
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Individual Model Cards

# COMMAND ----------

for model in top_models:
    if model["rank"] > 5:
        continue

    config_id = model["model_config_id"]
    run_id = model["mlflow_run_id"]

    print(f"\n{'='*70}")
    print(f"MODEL CARD: {config_id}")
    print(f"{'='*70}")
    print(f"  Rank:                  #{model['rank']} for {model['target_column']}")
    print(f"  Family:                {model['model_family']} ({model['model_type']})")
    print(f"  Target:                {model['target_column']}")
    print(f"  Features:              {model['feature_count']}")
    print(f"  Recommended Action:    {model['recommended_action']}")
    print(f"  ")
    print(f"  --- Metrics ---")
    if model["rmse"] is not None:
        print(f"  RMSE:                  {model['rmse']:.4f}")
    if model["mae"] is not None:
        print(f"  MAE:                   {model['mae']:.4f}")
    if model["r2"] is not None:
        print(f"  R²:                    {model['r2']:.4f}")
    if model["aic"] is not None:
        print(f"  AIC:                   {model['aic']:.1f}")
    if model["roc_auc"] is not None:
        print(f"  ROC AUC:               {model['roc_auc']:.4f}")
    print(f"  Gini:                  {model['gini']:.4f}")
    print(f"  Lift @ Decile 1:       {model['lift_decile1']:.2f}x")
    print(f"  PSI:                   {model['psi']:.4f} ({'STABLE' if model['psi'] < 0.1 else 'CAUTION' if model['psi'] < 0.25 else 'UNSTABLE'})")
    print(f"  Regulatory Score:      {model['regulatory_suitability_score']:.0f}/100")
    print(f"  Composite Score:       {model['composite_score']:.4f}")
    print(f"  MLflow Run:            {run_id}")

    # Try to load and display artifacts
    try:
        # GLM relativities
        artifact_path = client.download_artifacts(run_id, "glm_relativities.json")
        with open(artifact_path, "r") as f:
            relativities = json.load(f)
        print(f"\n  --- GLM Relativities (Rating Factors) ---")
        print(f"  {'Feature':<40s} {'Coeff':>10s} {'Relativity':>12s} {'p-value':>10s} {'Sig?':>5s}")
        print(f"  {'-'*77}")
        for rel in relativities[:20]:
            sig = "***" if rel.get("significant") else ""
            print(f"  {rel['feature']:<40s} {rel['coefficient']:>10.4f} {rel['relativity']:>12.4f} {rel['p_value']:>10.4f} {sig:>5s}")
    except Exception:
        pass

    try:
        # GBM feature importances
        artifact_path = client.download_artifacts(run_id, "feature_importances.json")
        with open(artifact_path, "r") as f:
            importances = json.load(f)
        print(f"\n  --- Feature Importances (Top 15) ---")
        print(f"  {'Feature':<40s} {'Importance':>12s}")
        print(f"  {'-'*52}")
        for imp in importances[:15]:
            bar = "█" * min(40, imp["importance"] // max(1, importances[0]["importance"] // 40))
            print(f"  {imp['feature']:<40s} {imp['importance']:>8d}  {bar}")
    except Exception:
        pass

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prediction Distributions — Top 3 Models

# COMMAND ----------

for model in top_models[:3]:
    run_id = model["mlflow_run_id"]
    config_id = model["model_config_id"]

    try:
        artifact_path = client.download_artifacts(run_id, "predictions.json")
        with open(artifact_path, "r") as f:
            preds = json.load(f)

        pred_pdf = pd.DataFrame({
            "actual": preds["y_test"][:500],
            "predicted": preds["y_pred"][:500],
        })
        pred_pdf["model"] = config_id

        display(
            spark.createDataFrame(pred_pdf)
        )
    except Exception:
        print(f"Could not load predictions for {config_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Record Actuary Decision
# MAGIC
# MAGIC Use the widgets below to record your decision for a specific model.
# MAGIC Each decision is **immutable** — once recorded, it becomes part of the
# MAGIC permanent audit trail.
# MAGIC
# MAGIC **Decision options:**
# MAGIC - `APPROVED` — Model is cleared for promotion to Unity Catalog registry
# MAGIC - `REJECTED` — Model does not meet standards; document the reason
# MAGIC - `DEFERRED` — Need more information; specify what's needed in conditions

# COMMAND ----------

dbutils.widgets.text("decision_model_config_id", "")
dbutils.widgets.dropdown("decision", "APPROVED", ["APPROVED", "REJECTED", "DEFERRED"])
dbutils.widgets.text("decision_notes", "")
dbutils.widgets.text("decision_conditions", "")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Submit Decision
# MAGIC
# MAGIC Fill in the widgets above and then **run this cell** to record the decision.

# COMMAND ----------

decision_config_id = dbutils.widgets.get("decision_model_config_id").strip()
decision = dbutils.widgets.get("decision")
notes = dbutils.widgets.get("decision_notes").strip()
conditions = dbutils.widgets.get("decision_conditions").strip()

if decision_config_id:
    # Look up the model in the leaderboard
    model_row = (
        spark.table(f"{fqn}.mf_leaderboard")
        .filter(f"factory_run_id = '{factory_run_id}' AND model_config_id = '{decision_config_id}'")
        .collect()
    )

    if model_row:
        model_info = model_row[0]
        decision_id = f"DEC-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{str(uuid.uuid4())[:8]}"

        decision_record = {
            "decision_id": decision_id,
            "factory_run_id": factory_run_id,
            "model_config_id": decision_config_id,
            "mlflow_run_id": model_info["mlflow_run_id"],
            "decision": decision,
            "reviewer": reviewer,
            "reviewer_notes": notes if notes else f"Actuary review of {decision_config_id}",
            "decided_at": datetime.now(timezone.utc).isoformat(),
            "regulatory_sign_off": decision == "APPROVED",
            "conditions": conditions if conditions else None,
        }

        spark.createDataFrame([decision_record]).write.mode("append").saveAsTable(f"{fqn}.mf_actuary_decisions")

        # Audit log
        log_audit_event(spark, fqn, factory_run_id,
            f"ACTUARY_{decision}",
            {
                "decision_id": decision_id,
                "model_config_id": decision_config_id,
                "decision": decision,
                "reviewer": reviewer,
                "notes": notes,
                "conditions": conditions,
                "model_composite_score": model_info["composite_score"],
                "model_regulatory_score": model_info["regulatory_suitability_score"],
            },
            mlflow_run_id=model_info["mlflow_run_id"],
            upt_version=upt_version,
        )

        print(f"Decision recorded: {decision} for {decision_config_id}")
        print(f"Decision ID: {decision_id}")
        print(f"Reviewer: {reviewer}")
        if conditions:
            print(f"Conditions: {conditions}")
    else:
        print(f"Model {decision_config_id} not found in leaderboard for this factory run")
else:
    print("Enter a model_config_id in the widget above to record a decision.")
    print("\nAvailable models:")
    for m in top_models:
        print(f"  {m['model_config_id']:45s} rank=#{m['rank']}  action={m['recommended_action']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Decision History
# MAGIC
# MAGIC All decisions recorded for this factory run (and across all runs if the actuary
# MAGIC wants to compare historical governance).

# COMMAND ----------

display(
    spark.table(f"{fqn}.mf_actuary_decisions")
    .filter(f"factory_run_id = '{factory_run_id}'")
    .orderBy("decided_at")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Full Audit Trail
# MAGIC
# MAGIC Every event from this factory run — from feature profiling through training
# MAGIC to this review — is recorded in the audit log. This table is ready to be
# MAGIC shared with internal model governance, external auditors, or the regulator.

# COMMAND ----------

display(
    spark.table(f"{fqn}.mf_audit_log")
    .filter(f"factory_run_id = '{factory_run_id}'")
    .orderBy("event_timestamp")
    .select("event_timestamp", "event_type", "actor", "mlflow_run_id", "details_json")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC Once you have approved one or more models, run `mf_05_promote_model` to:
# MAGIC 1. Register approved models in **Unity Catalog Model Registry**
# MAGIC 2. Set the **champion** alias for the top-ranked approved model per target
# MAGIC 3. Tag model versions with factory run ID, regulatory score, and actuary info
# MAGIC
# MAGIC The promoted models are then available for serving via **Mosaic AI Model Serving**
# MAGIC or for batch scoring in downstream pipelines.
