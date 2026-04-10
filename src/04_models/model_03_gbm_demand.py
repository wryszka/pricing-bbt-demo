# Databricks notebook source
# MAGIC %md
# MAGIC # Model 3: GBM Demand — Conversion Propensity
# MAGIC
# MAGIC Trains a gradient boosted model to predict **conversion probability**.
# MAGIC Drives the commercial pricing overlay — price elasticity and demand curves.
# MAGIC
# MAGIC **Target:** `converted` (binary)
# MAGIC **Method:** LightGBM classifier

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
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import pyspark.sql.functions as F
from pyspark.sql.functions import col, when
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

mlflow.set_registry_uri("databricks-uc")
try:
    user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
    mlflow.set_experiment(f"/Workspace/Users/{user}/pricing_upt_demand_gbm")
except Exception:
    pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load quote data enriched with UPT features

# COMMAND ----------

upt_table_name = f"{fqn}.unified_pricing_table_live"
quotes = spark.table(f"{fqn}.internal_quote_history")
upt = spark.table(upt_table_name)

upt_history = spark.sql(f"DESCRIBE HISTORY {upt_table_name} LIMIT 1").collect()
upt_delta_version = upt_history[0]["version"] if upt_history else None
print(f"Training from: {upt_table_name} (Delta version {upt_delta_version})")

# Get representative features per SIC+postcode from UPT
upt_features = (upt
    .select("sic_code", "postcode_sector",
            "flood_zone_rating", "crime_theft_index", "subsidence_risk",
            "composite_location_risk",
            "market_median_rate", "competitor_a_min_premium", "price_index_trend",
            "credit_default_probability", "business_stability_score",
            "population_density_per_km2", "distance_to_coast_km")
    .dropDuplicates(["sic_code", "postcode_sector"])
)

enriched = (quotes
    .withColumn("converted_flag", when(col("converted") == "Y", 1).otherwise(0))
    .withColumn("competitor_flag", when(col("competitor_quoted") == "Y", 1).otherwise(0))
    .join(upt_features, ["sic_code", "postcode_sector"], "left")
    .withColumn("quote_to_market_ratio",
        when(col("market_median_rate").isNotNull() & (col("market_median_rate") > 0),
             (col("quoted_premium") / (col("sum_insured") / 1000)) / col("market_median_rate"))
        .otherwise(None))
    .withColumn("log_premium", F.log1p(col("quoted_premium")))
    .withColumn("log_si", F.log1p(col("sum_insured")))
    .withColumn("log_turnover", F.log1p(col("annual_turnover")))
)

feature_cols = [
    "log_premium", "log_si", "log_turnover",
    "competitor_flag", "quote_to_market_ratio",
    "flood_zone_rating", "crime_theft_index", "subsidence_risk",
    "composite_location_risk",
    "market_median_rate", "competitor_a_min_premium", "price_index_trend",
    "credit_default_probability", "business_stability_score",
    "population_density_per_km2", "distance_to_coast_km",
]

pdf = enriched.select("quote_id", "converted_flag", *feature_cols).toPandas()
pdf[feature_cols] = pdf[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

print(f"Total quotes: {len(pdf)}, Conversion rate: {pdf['converted_flag'].mean():.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train/test split

# COMMAND ----------

pdf["split_hash"] = pdf["quote_id"].apply(lambda x: abs(hash(x)) % 100)
train_pdf = pdf[pdf["split_hash"] < 80].copy()
test_pdf = pdf[pdf["split_hash"] >= 80].copy()

print(f"Train: {len(train_pdf)}, Test: {len(test_pdf)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train LightGBM Classifier

# COMMAND ----------

X_train = train_pdf[feature_cols].values
X_test = test_pdf[feature_cols].values
y_train = train_pdf["converted_flag"].values
y_test = test_pdf["converted_flag"].values

with mlflow.start_run(run_name="lgbm_demand_conversion") as run:
    mlflow.log_param("model_type", "LightGBM_Classifier")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("features", len(feature_cols))
    mlflow.log_param("upt_delta_version", upt_delta_version)

    try:
        input_dataset = mlflow.data.from_spark(upt, table_name=upt_table_name, version=str(upt_delta_version))
        mlflow.log_input(input_dataset, context="training")
    except Exception as e:
        print(f"Note: mlflow.data.from_spark not available — {e}")
    mlflow.set_tag("feature_table", upt_table_name)
    mlflow.log_param("upt_table", f"{fqn}.unified_pricing_table_live")
    mlflow.log_param("train_rows", len(train_pdf))
    mlflow.log_param("test_rows", len(test_pdf))

    model = LGBMClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    roc_auc = roc_auc_score(y_test, y_proba)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    # Demand model operates at QUOTE level (not policy level) so it cannot use
    # FeatureLookup against the policy-keyed UPT directly. Log with standard
    # mlflow and register in UC. Feature enrichment for this model happens at
    # quote ingestion time, not at serving time.
    from mlflow.models.signature import infer_signature
    sig = infer_signature(pd.DataFrame(X_train, columns=feature_cols), y_pred)
    mlflow.sklearn.log_model(
        model, "lgbm_demand_model",
        signature=sig,
        registered_model_name=f"{catalog}.{schema}.lgbm_demand_model",
    )

    print(f"LightGBM Demand Results:")
    print(f"  ROC AUC:   {roc_auc:.4f}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  MLflow Run ID: {run.info.run_id}")
    print(f"  ✓ Model registered in UC (quote-level — enrichment at ingestion)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Importance

# COMMAND ----------

importances = model.feature_importances_
imp_data = [{"feature": feature_cols[i], "importance": int(importances[i])}
            for i in range(len(feature_cols)) if importances[i] > 0]

display(spark.createDataFrame(imp_data).orderBy(col("importance").desc()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Demand Curve: Conversion vs Price Ratio

# COMMAND ----------

test_pdf["predicted_conversion"] = y_proba

demand_df = (spark.createDataFrame(test_pdf[["quote_to_market_ratio", "converted_flag", "predicted_conversion"]])
    .withColumn("price_bucket", F.round(col("quote_to_market_ratio"), 1))
    .filter(col("price_bucket").between(0.3, 3.0))
    .groupBy("price_bucket")
    .agg(
        F.avg("converted_flag").alias("actual_conversion_rate"),
        F.avg("predicted_conversion").alias("predicted_conversion_rate"),
        F.count("*").alias("quote_count"),
    )
    .filter(col("quote_count") >= 10)
    .orderBy("price_bucket")
)

display(demand_df)
