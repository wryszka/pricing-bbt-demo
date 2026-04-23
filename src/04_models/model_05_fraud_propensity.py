# Databricks notebook source
# MAGIC %md
# MAGIC # Model 5: Fraud Propensity — Claims Fraud Detection
# MAGIC
# MAGIC Binary classifier predicting the likelihood a claim is fraudulent.
# MAGIC In a real P&C insurer, fraud scores are used to:
# MAGIC - Triage claims for investigation (above threshold → manual review)
# MAGIC - Adjust pricing with a fraud load factor for high-risk segments
# MAGIC - Feed into SIU (Special Investigations Unit) workflows
# MAGIC
# MAGIC **Target:** `is_fraud` (binary, ~3% prevalence — realistic for P&C)
# MAGIC **Method:** LightGBM classifier
# MAGIC **Features:** Claim patterns, policy age, coverage amounts, geographic risk

# COMMAND ----------

dbutils.widgets.text("catalog_name", "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name", "pricing_upt")

catalog = dbutils.widgets.get("catalog_name")
schema = dbutils.widgets.get("schema_name")
fqn = f"{catalog}.{schema}"

# COMMAND ----------

import mlflow
import mlflow.data
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from mlflow.models.signature import infer_signature
import pyspark.sql.functions as F
from pyspark.sql.functions import col

mlflow.set_registry_uri("databricks-uc")
try:
    user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
    mlflow.set_experiment(f"/Workspace/Users/{user}/pricing_workbench_fraud_propensity")
except Exception:
    pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load UPT and generate synthetic fraud labels
# MAGIC
# MAGIC Fraud indicators based on realistic P&C patterns:
# MAGIC - New policies with large claims (policy < 1 year, high claim amount)
# MAGIC - High claim frequency relative to industry norm
# MAGIC - Claims on recently increased coverage
# MAGIC - Geographic fraud hotspots (high crime areas)
# MAGIC - Unusual claim patterns (multiple perils, rapid succession)

# COMMAND ----------

upt_table_name = f"{fqn}.unified_pricing_table_live"
upt = spark.table(upt_table_name)

upt_history = spark.sql(f"DESCRIBE HISTORY {upt_table_name} LIMIT 1").collect()
upt_delta_version = upt_history[0]["version"] if upt_history else None

# Build fraud labels from realistic indicators
pdf = upt.select(
    "policy_id",
    "annual_turnover", "sum_insured", "current_premium", "building_age_years",
    "claim_count_5y", "total_incurred_5y", "open_claims_count",
    "flood_zone_rating", "crime_theft_index", "subsidence_risk",
    "composite_location_risk", "credit_score", "ccj_count", "years_trading",
    "business_stability_score", "loss_ratio_5y",
    "credit_default_probability", "director_stability_score",
    "employee_count_est", "revenue_growth_3y_pct",
    "industry_risk_tier", "location_risk_tier", "credit_risk_tier",
).toPandas()

feature_cols = [
    "annual_turnover", "sum_insured", "current_premium", "building_age_years",
    "claim_count_5y", "total_incurred_5y", "open_claims_count",
    "flood_zone_rating", "crime_theft_index", "subsidence_risk",
    "composite_location_risk", "credit_score", "ccj_count", "years_trading",
    "business_stability_score", "loss_ratio_5y",
    "credit_default_probability", "director_stability_score",
    "employee_count_est", "revenue_growth_3y_pct",
]

pdf[feature_cols] = pdf[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

# Generate synthetic fraud labels (~3% fraud rate)
# Fraud is more likely when: high claims relative to premium, poor credit,
# high crime area, new business, multiple open claims
np.random.seed(42)
fraud_score = (
    (pdf["loss_ratio_5y"].clip(0, 5) / 5 * 0.25) +
    (pdf["claim_count_5y"].clip(0, 10) / 10 * 0.2) +
    ((100 - pdf["crime_theft_index"].clip(0, 100)) / 100 * 0.15) +
    (pdf["ccj_count"].clip(0, 5) / 5 * 0.15) +
    (pdf["credit_default_probability"].clip(0, 0.5) / 0.5 * 0.15) +
    (pdf["open_claims_count"].clip(0, 5) / 5 * 0.1)
)
# Add noise and threshold at ~3%
fraud_prob = fraud_score + np.random.normal(0, 0.15, len(pdf))
threshold = np.percentile(fraud_prob, 97)
pdf["is_fraud"] = (fraud_prob >= threshold).astype(int)

fraud_rate = pdf["is_fraud"].mean()
print(f"Training data: {len(pdf)} policies")
print(f"Fraud rate: {fraud_rate:.1%} ({pdf['is_fraud'].sum()} fraudulent)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train/test split and model training

# COMMAND ----------

pdf["split_hash"] = pdf["policy_id"].apply(lambda x: abs(hash(x)) % 100)
train_pdf = pdf[pdf["split_hash"] < 80].copy()
test_pdf = pdf[pdf["split_hash"] >= 80].copy()

X_train = train_pdf[feature_cols].values
X_test = test_pdf[feature_cols].values
y_train = train_pdf["is_fraud"].values
y_test = test_pdf["is_fraud"].values

with mlflow.start_run(run_name="lgbm_fraud_propensity") as run:
    mlflow.log_param("model_type", "LightGBM_Classifier")
    mlflow.log_param("target", "is_fraud")
    mlflow.log_param("fraud_rate", round(fraud_rate, 3))
    mlflow.log_param("features", len(feature_cols))
    mlflow.log_param("upt_table", upt_table_name)
    mlflow.log_param("upt_delta_version", upt_delta_version)
    mlflow.log_param("train_rows", len(train_pdf))
    mlflow.log_param("test_rows", len(test_pdf))

    try:
        input_dataset = mlflow.data.from_spark(upt, table_name=upt_table_name, version=str(upt_delta_version))
        mlflow.log_input(input_dataset, context="training")
    except Exception:
        pass
    mlflow.set_tag("feature_table", upt_table_name)

    model = LGBMClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=int(1 / fraud_rate),  # Handle class imbalance
        random_state=42, verbose=-1,
    )
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    roc_auc = roc_auc_score(y_test, y_proba)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)

    sig = infer_signature(pd.DataFrame(X_train, columns=feature_cols), y_proba)
    mlflow.sklearn.log_model(
        model, "lgbm_fraud_model", signature=sig,
        registered_model_name=f"{catalog}.{schema}.lgbm_fraud_model",
    )

    print(f"Fraud Propensity Results:")
    print(f"  ROC AUC:   {roc_auc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  MLflow Run ID: {run.info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Importance — What drives fraud detection?

# COMMAND ----------

importances = model.feature_importances_
imp_data = [{"feature": feature_cols[i], "importance": int(importances[i])}
            for i in range(len(feature_cols)) if importances[i] > 0]

display(spark.createDataFrame(imp_data).orderBy(col("importance").desc()))
