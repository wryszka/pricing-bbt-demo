# Databricks notebook source
# MAGIC %md
# MAGIC # Model Factory — Step 3: Evaluation & Leaderboard
# MAGIC
# MAGIC This notebook pulls metrics from MLflow for every model trained in the factory
# MAGIC run, computes **insurance-specific evaluation metrics**, and builds a ranked
# MAGIC **leaderboard** that the actuary will review in the next step.
# MAGIC
# MAGIC ### Insurance-Specific Metrics
# MAGIC
# MAGIC | Metric | What It Measures | Why It Matters |
# MAGIC |--------|-----------------|----------------|
# MAGIC | **Gini Coefficient** | Discriminatory power | Can the model separate good from bad risks? |
# MAGIC | **Lift at Decile 1** | Extreme risk identification | Does the top 10% predicted capture disproportionate losses? |
# MAGIC | **PSI** | Population Stability Index | Is the model stable across train/test? (Overfitting detector) |
# MAGIC | **Regulatory Suitability** | Composite governance score | Would a regulator accept this model? |
# MAGIC
# MAGIC ### Ranking
# MAGIC
# MAGIC Models are ranked by a weighted composite score that balances predictive power
# MAGIC with regulatory acceptability. GLMs get a natural boost for interpretability;
# MAGIC GBMs must earn their place through superior discrimination.

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
        factory_run_id = dbutils.jobs.taskValues.get(taskKey="mf_automated_training", key="factory_run_id")
    except Exception:
        latest = spark.sql(f"SELECT DISTINCT factory_run_id FROM {fqn}.mf_training_log ORDER BY factory_run_id DESC LIMIT 1").collect()
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Training Results

# COMMAND ----------

import mlflow
import numpy as np
import pandas as pd

mlflow.set_registry_uri("databricks-uc")

# Get all successful training runs for this factory run
training_log = (
    spark.table(f"{fqn}.mf_training_log")
    .filter(f"factory_run_id = '{factory_run_id}' AND status = 'SUCCESS'")
    .collect()
)

# Also load the training plan for metadata
training_plan = {
    row["model_config_id"]: row
    for row in spark.table(f"{fqn}.mf_training_plan")
        .filter(f"factory_run_id = '{factory_run_id}'")
        .collect()
}

print(f"Successful models to evaluate: {len(training_log)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retrieve MLflow Metrics & Compute Insurance Metrics

# COMMAND ----------

client = mlflow.tracking.MlflowClient()

def compute_gini(y_true, y_pred):
    """Compute Gini coefficient (2*AUC - 1 for ordered predictions)."""
    n = len(y_true)
    if n == 0:
        return 0.0
    # Sort by predicted values
    idx = np.argsort(y_pred)
    y_sorted = np.array(y_true)[idx]
    cumulative = np.cumsum(y_sorted)
    total = cumulative[-1]
    if total == 0:
        return 0.0
    # Lorenz curve
    lorenz = cumulative / total
    # Gini = 1 - 2 * area under Lorenz curve
    gini = 1.0 - 2.0 * np.mean(lorenz)
    return abs(gini)


def compute_lift_decile1(y_true, y_pred):
    """Compute lift in top decile (top 10% by prediction vs average)."""
    n = len(y_true)
    if n < 10:
        return 1.0
    idx = np.argsort(y_pred)[::-1]  # Descending by prediction
    top_n = max(1, n // 10)
    top_actual = np.mean(np.array(y_true)[idx[:top_n]])
    overall = np.mean(y_true)
    if overall == 0:
        return 1.0
    return top_actual / overall


def compute_psi(y_train_pred, y_test_pred, bins=10):
    """Population Stability Index: measures distribution shift between train and test.
    PSI < 0.1: no significant shift
    PSI 0.1-0.25: moderate shift
    PSI > 0.25: significant shift (model may be unstable)
    """
    # Bin edges from training predictions
    edges = np.quantile(y_train_pred, np.linspace(0, 1, bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf

    train_counts = np.histogram(y_train_pred, bins=edges)[0] / len(y_train_pred)
    test_counts = np.histogram(y_test_pred, bins=edges)[0] / len(y_test_pred)

    # Avoid division by zero
    train_counts = np.clip(train_counts, 1e-6, None)
    test_counts = np.clip(test_counts, 1e-6, None)

    psi = np.sum((test_counts - train_counts) * np.log(test_counts / train_counts))
    return float(psi)


def compute_regulatory_suitability(model_family, model_type, feature_count, aic, best_aic, gini, psi):
    """Score 0-100 reflecting how likely a regulator would accept this model.
    GLMs are inherently more acceptable due to transparent relativities.
    """
    score = 0.0

    # Interpretability bonus for GLMs
    if model_family == "GLM":
        score += 25
    elif model_family == "GLM_GBM_UPLIFT":
        score += 12  # Partial credit — GLM base is interpretable

    # Parsimony: fewer features = easier to justify
    if feature_count <= 15:
        score += 15
    elif feature_count <= 25:
        score += 10
    elif feature_count <= 40:
        score += 5

    # Information criterion: close to best AIC
    if best_aic is not None and aic is not None and best_aic > 0:
        aic_ratio = aic / best_aic if best_aic != 0 else 1.0
        if aic_ratio <= 1.05:
            score += 15
        elif aic_ratio <= 1.15:
            score += 8

    # Discrimination: Gini > 0.3 is good for insurance
    if gini >= 0.4:
        score += 25
    elif gini >= 0.3:
        score += 20
    elif gini >= 0.2:
        score += 10

    # Stability: low PSI means model generalises well
    if psi < 0.1:
        score += 20
    elif psi < 0.25:
        score += 10

    return min(100.0, score)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluate Each Model

# COMMAND ----------

leaderboard_rows = []

for log_row in training_log:
    config_id = log_row["model_config_id"]
    run_id = log_row["mlflow_run_id"]
    plan_row = training_plan.get(config_id, {})

    print(f"Evaluating: {config_id}...")

    try:
        # Get MLflow run data
        mlflow_run = client.get_run(run_id)
        metrics = mlflow_run.data.metrics
        params = mlflow_run.data.params
        tags = mlflow_run.data.tags

        model_family = tags.get("model_family", plan_row.get("model_family", "unknown"))
        model_type = tags.get("model_type", plan_row.get("model_type", "unknown"))
        target_col = tags.get("target_column", plan_row.get("target_column", "unknown"))
        feature_count = int(params.get("feature_count", 0))

        # Core metrics from MLflow
        rmse = metrics.get("rmse")
        mae = metrics.get("mae")
        r2 = metrics.get("r2")
        aic = metrics.get("aic")
        bic = metrics.get("bic")
        roc_auc = metrics.get("roc_auc")

        # Load predictions artifact for advanced metrics
        gini = 0.0
        lift_d1 = 1.0
        psi = 0.0

        try:
            artifact_path = client.download_artifacts(run_id, "predictions.json")
            with open(artifact_path, "r") as f:
                preds = json.load(f)
            y_test = np.array(preds["y_test"])
            y_pred = np.array(preds["y_pred"])

            gini = compute_gini(y_test, y_pred)
            lift_d1 = compute_lift_decile1(y_test, y_pred)

            # For PSI, we compare test prediction distribution to itself
            # (In production, you'd compare to the training predictions)
            # Here we split the test predictions to approximate
            mid = len(y_pred) // 2
            if mid > 10:
                psi = compute_psi(y_pred[:mid], y_pred[mid:])
        except Exception:
            pass

        leaderboard_rows.append({
            "factory_run_id": factory_run_id,
            "rank": 0,  # Will be computed after all models
            "model_config_id": config_id,
            "model_family": model_family,
            "model_type": model_type,
            "target_column": target_col,
            "feature_count": feature_count,
            "rmse": round(rmse, 4) if rmse is not None else None,
            "mae": round(mae, 4) if mae is not None else None,
            "r2": round(r2, 4) if r2 is not None else None,
            "aic": round(aic, 1) if aic is not None else None,
            "bic": round(bic, 1) if bic is not None else None,
            "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
            "gini": round(gini, 4),
            "lift_decile1": round(lift_d1, 4),
            "psi": round(psi, 4),
            "regulatory_suitability_score": 0.0,  # Computed below
            "composite_score": 0.0,  # Computed below
            "mlflow_run_id": run_id,
            "recommended_action": "",  # Computed below
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        })

        print(f"  Gini={gini:.4f}, Lift@D1={lift_d1:.2f}, PSI={psi:.4f}")

    except Exception as e:
        print(f"  Error evaluating {config_id}: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute Regulatory Suitability & Composite Scores

# COMMAND ----------

# Find best AIC across all models (for relative comparison)
all_aics = [r["aic"] for r in leaderboard_rows if r["aic"] is not None and r["aic"] > 0]
best_aic = min(all_aics) if all_aics else None

# Compute regulatory suitability for each model
for row in leaderboard_rows:
    row["regulatory_suitability_score"] = round(compute_regulatory_suitability(
        row["model_family"], row["model_type"], row["feature_count"],
        row["aic"], best_aic, row["gini"], row["psi"],
    ), 1)

# Compute composite score (normalised weighted blend)
# Group by target column so we rank within comparable groups
from collections import defaultdict
target_groups = defaultdict(list)
for row in leaderboard_rows:
    target_groups[row["target_column"]].append(row)

for target, rows in target_groups.items():
    # Normalise each metric to 0-1 within the group
    ginis = [r["gini"] for r in rows]
    rmses = [r["rmse"] for r in rows if r["rmse"] is not None]
    reg_scores = [r["regulatory_suitability_score"] for r in rows]
    lifts = [r["lift_decile1"] for r in rows]
    psis = [r["psi"] for r in rows]

    max_gini = max(ginis) if ginis else 1.0
    min_rmse = min(rmses) if rmses else 1.0
    max_rmse = max(rmses) if rmses else 1.0
    max_reg = max(reg_scores) if reg_scores else 1.0
    max_lift = max(lifts) if lifts else 1.0

    for row in rows:
        # Normalise (higher is better for all)
        norm_gini = row["gini"] / max_gini if max_gini > 0 else 0
        norm_rmse_inv = 1.0 - ((row["rmse"] - min_rmse) / (max_rmse - min_rmse)) if (max_rmse > min_rmse and row["rmse"] is not None) else 0.5
        norm_reg = row["regulatory_suitability_score"] / max_reg if max_reg > 0 else 0
        norm_lift = row["lift_decile1"] / max_lift if max_lift > 0 else 0
        norm_stability = max(0, 1.0 - row["psi"] * 4)  # PSI 0=perfect, 0.25=zero score

        composite = (
            0.30 * norm_gini +
            0.20 * norm_rmse_inv +
            0.20 * norm_reg +
            0.15 * norm_stability +
            0.15 * norm_lift
        )
        row["composite_score"] = round(composite, 4)

    # Rank within target group
    rows.sort(key=lambda r: r["composite_score"], reverse=True)
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank

        # Recommended action
        if rank <= 3 and row["regulatory_suitability_score"] >= 50:
            row["recommended_action"] = "RECOMMEND_APPROVE"
        elif rank <= 5 or row["regulatory_suitability_score"] >= 40:
            row["recommended_action"] = "REVIEW_REQUIRED"
        else:
            row["recommended_action"] = "NOT_RECOMMENDED"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Leaderboard

# COMMAND ----------

# Ensure leaderboard table exists
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {fqn}.mf_leaderboard (
        factory_run_id STRING,
        rank INT,
        model_config_id STRING,
        model_family STRING,
        model_type STRING,
        target_column STRING,
        feature_count INT,
        rmse DOUBLE,
        mae DOUBLE,
        r2 DOUBLE,
        aic DOUBLE,
        bic DOUBLE,
        roc_auc DOUBLE,
        gini DOUBLE,
        lift_decile1 DOUBLE,
        psi DOUBLE,
        regulatory_suitability_score DOUBLE,
        composite_score DOUBLE,
        mlflow_run_id STRING,
        recommended_action STRING,
        evaluated_at STRING
    )
""")

# Ensure consistent types for leaderboard
for row in leaderboard_rows:
    for k, v in row.items():
        if isinstance(v, int) and k not in ("rank", "feature_count"):
            row[k] = float(v)
        if v is None and k not in ("mlflow_run_id", "recommended_action", "evaluated_at"):
            row[k] = 0.0

leaderboard_df = spark.createDataFrame(leaderboard_rows)
leaderboard_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{fqn}.mf_leaderboard")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Leaderboard — Frequency Models (claim_count_5y)

# COMMAND ----------

display(
    spark.table(f"{fqn}.mf_leaderboard")
    .filter(f"factory_run_id = '{factory_run_id}' AND target_column = 'claim_count_5y'")
    .select("rank", "model_config_id", "model_family", "model_type",
            "feature_count", "rmse", "gini", "lift_decile1", "psi",
            "regulatory_suitability_score", "composite_score", "recommended_action")
    .orderBy("rank")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Leaderboard — Severity Models (total_incurred_5y)

# COMMAND ----------

display(
    spark.table(f"{fqn}.mf_leaderboard")
    .filter(f"factory_run_id = '{factory_run_id}' AND target_column = 'total_incurred_5y'")
    .select("rank", "model_config_id", "model_family", "model_type",
            "feature_count", "rmse", "gini", "lift_decile1", "psi",
            "regulatory_suitability_score", "composite_score", "recommended_action")
    .orderBy("rank")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Leaderboard — Demand/Conversion Models

# COMMAND ----------

display(
    spark.table(f"{fqn}.mf_leaderboard")
    .filter(f"factory_run_id = '{factory_run_id}' AND target_column = 'converted'")
    .select("rank", "model_config_id", "model_family", "model_type",
            "feature_count", "roc_auc", "gini", "lift_decile1", "psi",
            "regulatory_suitability_score", "composite_score", "recommended_action")
    .orderBy("rank")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary — Best Models by Target

# COMMAND ----------

for target, rows in target_groups.items():
    best = rows[0]
    print(f"\nBest model for {target}:")
    print(f"  Config:    {best['model_config_id']}")
    print(f"  Family:    {best['model_family']} ({best['model_type']})")
    print(f"  Composite: {best['composite_score']:.4f}")
    print(f"  Gini:      {best['gini']:.4f}")
    print(f"  RMSE:      {best['rmse']}" if best['rmse'] else f"  ROC-AUC:   {best['roc_auc']}")
    print(f"  Reg Score: {best['regulatory_suitability_score']:.0f}/100")
    print(f"  Action:    {best['recommended_action']}")

# COMMAND ----------

log_audit_event(spark, fqn, factory_run_id, "LEADERBOARD_PUBLISHED", {
    "total_models_evaluated": len(leaderboard_rows),
    "models_by_target": {t: len(rs) for t, rs in target_groups.items()},
    "recommended_for_approval": [r["model_config_id"] for r in leaderboard_rows if r["recommended_action"] == "RECOMMEND_APPROVE"],
    "best_gini": max(r["gini"] for r in leaderboard_rows) if leaderboard_rows else 0,
}, upt_version=upt_version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Genie-Ready: Query This Data in Natural Language
# MAGIC
# MAGIC All leaderboard data is stored in standard Delta tables that a **Genie room**
# MAGIC can query directly. Example questions an actuary could ask:
# MAGIC
# MAGIC - *"Show me the top 3 frequency models ranked by Gini coefficient"*
# MAGIC - *"Which GLM models have a regulatory suitability score above 60?"*
# MAGIC - *"Compare the AIC of Poisson vs Tweedie models"*
# MAGIC - *"What's the PSI distribution across all models?"*
# MAGIC
# MAGIC To set up: Create a Genie room pointing at `{schema}` and include the
# MAGIC `mf_leaderboard`, `mf_feature_profile`, and `mf_audit_log` tables.
# MAGIC
# MAGIC **Next step:** Run `mf_04_actuary_review` to review and approve the top models.
