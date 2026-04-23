# Databricks notebook source
# MAGIC %md
# MAGIC # Model 6: Retention/Churn — Policy Non-Renewal Prediction
# MAGIC
# MAGIC Binary classifier predicting whether a policy will NOT renew.
# MAGIC In a real P&C insurer, retention scores drive:
# MAGIC - Pricing flexibility (discount for at-risk policies to retain them)
# MAGIC - Proactive outreach before renewal date
# MAGIC - Portfolio value analysis (which customers are worth retaining?)
# MAGIC
# MAGIC **Target:** `is_churned` (binary, ~15% churn rate — realistic for commercial P&C)
# MAGIC **Method:** LightGBM classifier
# MAGIC **Features:** Tenure, premium changes, claim history, market competitiveness

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
    mlflow.set_experiment(f"/Workspace/Users/{user}/pricing_workbench_retention")
except Exception:
    pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load UPT and generate synthetic churn labels
# MAGIC
# MAGIC Churn indicators based on realistic commercial P&C patterns:
# MAGIC - We're expensive vs market (high market_position_ratio)
# MAGIC - Recent premium increase (rate_per_1k_si above average)
# MAGIC - No claims history (less stickiness — "shopping around")
# MAGIC - Low business stability (company in trouble)
# MAGIC - Competitor actively quoting (competitor_quoted flag)

# COMMAND ----------

upt_table_name = f"{fqn}.unified_pricing_table_live"
upt = spark.table(upt_table_name)

upt_history = spark.sql(f"DESCRIBE HISTORY {upt_table_name} LIMIT 1").collect()
upt_delta_version = upt_history[0]["version"] if upt_history else None

pdf = upt.select(
    "policy_id",
    "annual_turnover", "sum_insured", "current_premium", "building_age_years",
    "claim_count_5y", "total_incurred_5y", "loss_ratio_5y",
    "market_median_rate", "market_position_ratio", "rate_per_1k_si",
    "price_index_trend", "competitor_ratio",
    "credit_score", "business_stability_score", "years_trading",
    "composite_location_risk", "combined_risk_score",
    "flood_zone_rating", "crime_theft_index",
    "quote_count", "competitor_quote_count",
).toPandas()

feature_cols = [
    "annual_turnover", "sum_insured", "current_premium", "building_age_years",
    "claim_count_5y", "total_incurred_5y", "loss_ratio_5y",
    "market_median_rate", "market_position_ratio", "rate_per_1k_si",
    "price_index_trend", "competitor_ratio",
    "credit_score", "business_stability_score", "years_trading",
    "composite_location_risk", "combined_risk_score",
    "flood_zone_rating", "crime_theft_index",
    "quote_count", "competitor_quote_count",
]

pdf[feature_cols] = pdf[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

# Generate synthetic churn labels (~15% churn rate)
# Churn is more likely when: overpriced vs market, no claims (less sticky),
# low stability, high competitor activity
np.random.seed(99)
churn_score = (
    (pdf["market_position_ratio"].clip(0.5, 2.0) - 0.5) / 1.5 * 0.30 +  # Overpriced → churn
    (1.0 - pdf["claim_count_5y"].clip(0, 5) / 5) * 0.20 +  # No claims → less sticky
    (1.0 - pdf["business_stability_score"].clip(0, 100) / 100) * 0.15 +  # Unstable business
    (pdf["competitor_quote_count"].clip(0, 5) / 5) * 0.15 +  # Competitor activity
    (pdf["price_index_trend"].clip(-10, 20) / 20) * 0.10 +  # Rising market
    (1.0 - pdf["years_trading"].clip(0, 50) / 50) * 0.10  # Young business
)
churn_prob = churn_score + np.random.normal(0, 0.12, len(pdf))
threshold = np.percentile(churn_prob, 85)
pdf["is_churned"] = (churn_prob >= threshold).astype(int)

churn_rate = pdf["is_churned"].mean()
print(f"Training data: {len(pdf)} policies")
print(f"Churn rate: {churn_rate:.1%} ({pdf['is_churned'].sum()} churned)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train/test split and model training

# COMMAND ----------

pdf["split_hash"] = pdf["policy_id"].apply(lambda x: abs(hash(x)) % 100)
train_pdf = pdf[pdf["split_hash"] < 80].copy()
test_pdf = pdf[pdf["split_hash"] >= 80].copy()

X_train = train_pdf[feature_cols].values
X_test = test_pdf[feature_cols].values
y_train = train_pdf["is_churned"].values
y_test = test_pdf["is_churned"].values

with mlflow.start_run(run_name="lgbm_retention_churn") as run:
    mlflow.log_param("model_type", "LightGBM_Classifier")
    mlflow.log_param("target", "is_churned")
    mlflow.log_param("churn_rate", round(churn_rate, 3))
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
        model, "lgbm_retention_model", signature=sig,
        registered_model_name=f"{catalog}.{schema}.lgbm_retention_model",
    )

    print(f"Retention/Churn Results:")
    print(f"  ROC AUC:   {roc_auc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  MLflow Run ID: {run.info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Importance — What drives churn?

# COMMAND ----------

importances = model.feature_importances_
imp_data = [{"feature": feature_cols[i], "importance": int(importances[i])}
            for i in range(len(feature_cols)) if importances[i] > 0]

display(spark.createDataFrame(imp_data).orderBy(col("importance").desc()))
