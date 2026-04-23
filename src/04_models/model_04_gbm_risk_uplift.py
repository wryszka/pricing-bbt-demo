# Databricks notebook source
# MAGIC %md
# MAGIC # Model 4: GBM Risk Uplift — GLM Residual Learner
# MAGIC
# MAGIC Trains a GBM on the **residuals of the frequency GLM** to capture
# MAGIC non-linear interactions the GLM missed.
# MAGIC
# MAGIC **Target:** `glm_residual` (actual - GLM predicted)
# MAGIC **Method:** LightGBM on expanded feature set

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
import statsmodels.api as sm
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

mlflow.set_registry_uri("databricks-uc")
try:
    user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
    mlflow.set_experiment(f"/Workspace/Users/{user}/pricing_workbench_risk_uplift_gbm")
except Exception:
    pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load data with expanded feature set

# COMMAND ----------

upt_table_name = f"{fqn}.unified_pricing_table_live"
upt = spark.table(upt_table_name)

upt_history = spark.sql(f"DESCRIBE HISTORY {upt_table_name} LIMIT 1").collect()
upt_delta_version = upt_history[0]["version"] if upt_history else None
print(f"Training from: {upt_table_name} (Delta version {upt_delta_version})")

# GLM features (same as Model 1)
glm_features = [
    "annual_turnover", "sum_insured", "building_age_years", "current_premium",
    "flood_zone_rating", "proximity_to_fire_station_km", "crime_theft_index", "subsidence_risk",
    "composite_location_risk", "credit_score", "ccj_count", "years_trading",
    "business_stability_score", "market_median_rate",
    "credit_default_probability", "director_stability_score",
    "employee_count_est", "distance_to_coast_km", "population_density_per_km2",
    "elevation_metres", "annual_rainfall_mm",
]

# Extra features the GBM can use (non-linear interactions)
gbm_extra = [
    "traffic_density_index", "air_quality_index", "average_property_value_k",
    "commercial_density_score", "historic_flood_events_10y",
    "distance_to_hospital_km", "distance_to_motorway_km",
    "broadband_speed_mbps", "average_wind_speed_mph",
    "soil_clay_content_pct", "radon_risk_level",
    "debt_to_equity_ratio", "working_capital_ratio", "revenue_growth_3y_pct",
    "supplier_concentration_score", "invoice_dispute_rate_pct",
    "profit_margin_est_pct", "asset_tangibility_ratio",
    "company_age_months", "management_experience_score",
]

all_features = glm_features + gbm_extra
select_cols = ["policy_id", "claim_count_5y"] + all_features

pdf = upt.select(*select_cols).toPandas()
pdf["claim_frequency"] = pdf["claim_count_5y"].fillna(0).astype(float)
pdf[all_features] = pdf[all_features].apply(pd.to_numeric, errors="coerce").fillna(0)

pdf["split_hash"] = pdf["policy_id"].apply(lambda x: abs(hash(x)) % 100)
train_pdf = pdf[pdf["split_hash"] < 80].copy()
test_pdf = pdf[pdf["split_hash"] >= 80].copy()

print(f"Train: {len(train_pdf)}, Test: {len(test_pdf)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Train base GLM and compute residuals

# COMMAND ----------

X_train_glm = sm.add_constant(train_pdf[glm_features].values)
X_test_glm = sm.add_constant(test_pdf[glm_features].values)
y_train = train_pdf["claim_frequency"].values
y_test = test_pdf["claim_frequency"].values

glm = sm.GLM(y_train, X_train_glm, family=sm.families.Poisson(link=sm.families.links.Log()))
glm_result = glm.fit()

train_pdf["glm_pred"] = glm_result.predict(X_train_glm)
test_pdf["glm_pred"] = glm_result.predict(X_test_glm)

train_pdf["glm_residual"] = train_pdf["claim_frequency"] - train_pdf["glm_pred"]
test_pdf["glm_residual"] = test_pdf["claim_frequency"] - test_pdf["glm_pred"]

glm_rmse = np.sqrt(mean_squared_error(y_test, test_pdf["glm_pred"]))
glm_r2 = r2_score(y_test, test_pdf["glm_pred"])

print(f"Base GLM: RMSE={glm_rmse:.4f}, R2={glm_r2:.4f}")
print(f"Mean residual: {train_pdf['glm_residual'].mean():.6f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Train LightGBM on residuals with expanded features

# COMMAND ----------

X_train_gbm = train_pdf[all_features].values
X_test_gbm = test_pdf[all_features].values

with mlflow.start_run(run_name="lgbm_risk_uplift") as run:
    mlflow.log_param("model_type", "LightGBM_Uplift")
    mlflow.log_param("approach", "GLM_residual_learning")
    mlflow.log_param("features_glm", len(glm_features))
    mlflow.log_param("features_gbm_total", len(all_features))
    mlflow.log_param("features_gbm_extra", len(gbm_extra))
    mlflow.log_param("upt_table", upt_table_name)
    mlflow.log_param("upt_delta_version", upt_delta_version)

    try:
        input_dataset = mlflow.data.from_spark(upt, table_name=upt_table_name, version=str(upt_delta_version))
        mlflow.log_input(input_dataset, context="training")
    except Exception as e:
        print(f"Note: mlflow.data.from_spark not available — {e}")
    mlflow.set_tag("feature_table", upt_table_name)

    gbm = LGBMRegressor(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    gbm.fit(X_train_gbm, train_pdf["glm_residual"].values)

    uplift_pred = gbm.predict(X_test_gbm)
    combined_pred = np.maximum(0, test_pdf["glm_pred"].values + uplift_pred)

    # Compare: GLM only vs GLM + GBM
    combined_rmse = np.sqrt(mean_squared_error(y_test, combined_pred))
    combined_r2 = r2_score(y_test, combined_pred)
    improvement_pct = (glm_rmse - combined_rmse) / glm_rmse * 100

    mlflow.log_metric("glm_only_rmse", glm_rmse)
    mlflow.log_metric("glm_only_r2", glm_r2)
    mlflow.log_metric("combined_rmse", combined_rmse)
    mlflow.log_metric("combined_r2", combined_r2)
    mlflow.log_metric("rmse_improvement_pct", round(improvement_pct, 2))

    # Log with fe.log_model() for automatic feature lookup at serving time
    fe = FeatureEngineeringClient()
    training_set = fe.create_training_set(
        df=spark.createDataFrame(train_pdf[["policy_id", "glm_residual"]]),
        feature_lookups=[FeatureLookup(
            table_name=upt_table_name,
            feature_names=all_features,
            lookup_key="policy_id",
        )],
        label="glm_residual",
    )

    fe.log_model(
        model=gbm,
        artifact_path="lgbm_uplift_model",
        flavor=mlflow.sklearn,
        training_set=training_set,
        registered_model_name=f"{catalog}.{schema}.lgbm_uplift_model",
    )

    print(f"Model Comparison:")
    print(f"  {'Model':<20} {'RMSE':>10} {'R2':>10}")
    print(f"  {'-'*40}")
    print(f"  {'GLM Only':<20} {glm_rmse:>10.4f} {glm_r2:>10.4f}")
    print(f"  {'GLM + GBM Uplift':<20} {combined_rmse:>10.4f} {combined_r2:>10.4f}")
    print(f"  {'Improvement':<20} {improvement_pct:>9.1f}%")
    print(f"\n  MLflow Run ID: {run.info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Importance: What did the GLM miss?

# COMMAND ----------

importances = gbm.feature_importances_
imp_data = [{"feature": all_features[i], "importance": int(importances[i])}
            for i in range(len(all_features)) if importances[i] > 0]

display(spark.createDataFrame(imp_data).orderBy(col("importance").desc()).limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Segments where GLM underperforms

# COMMAND ----------

test_pdf["uplift"] = uplift_pred
test_pdf["combined_pred"] = combined_pred

# Bring back to Spark for grouping
seg_df = spark.createDataFrame(test_pdf[["policy_id", "claim_frequency", "glm_pred", "combined_pred", "uplift"]].head(10000))
upt_seg = upt.select("policy_id", "industry_risk_tier", "location_risk_tier")

display(
    seg_df.join(upt_seg, "policy_id", "left")
    .groupBy("industry_risk_tier", "location_risk_tier")
    .agg(
        F.count("*").alias("policies"),
        F.round(F.avg("glm_pred"), 4).alias("avg_glm"),
        F.round(F.avg("combined_pred"), 4).alias("avg_combined"),
        F.round(F.avg("claim_frequency"), 4).alias("avg_actual"),
        F.round(F.avg(F.abs(col("uplift"))), 4).alias("avg_abs_uplift"),
    )
    .orderBy(col("avg_abs_uplift").desc())
)
