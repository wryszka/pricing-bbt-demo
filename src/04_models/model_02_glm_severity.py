# Databricks notebook source
# MAGIC %md
# MAGIC # Model 2: GLM Severity — Average Claim Cost Prediction
# MAGIC
# MAGIC Trains a Gamma GLM to predict **claim severity** (average claim cost given a claim).
# MAGIC Combined with frequency: `Technical Price = Frequency x Severity`
# MAGIC
# MAGIC **Target:** `avg_claim_severity`
# MAGIC **Distribution:** Gamma (log link)
# MAGIC **Filter:** Only policies with claims

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
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

mlflow.set_registry_uri("databricks-uc")
try:
    user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
    mlflow.set_experiment(f"/Workspace/Users/{user}/pricing_workbench_severity_glm")
except Exception:
    pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load and prepare — policies with claims only

# COMMAND ----------

upt_table_name = f"{fqn}.unified_pricing_table_live"
upt = spark.table(upt_table_name)

upt_history = spark.sql(f"DESCRIBE HISTORY {upt_table_name} LIMIT 1").collect()
upt_delta_version = upt_history[0]["version"] if upt_history else None
print(f"Training from: {upt_table_name} (Delta version {upt_delta_version})")

feature_cols = [
    "annual_turnover", "sum_insured", "building_age_years",
    "flood_zone_rating", "proximity_to_fire_station_km", "crime_theft_index", "subsidence_risk",
    "composite_location_risk", "credit_score", "ccj_count", "years_trading",
    "business_stability_score",
    "credit_default_probability", "employee_count_est",
    "distance_to_coast_km", "population_density_per_km2",
    "elevation_metres", "annual_rainfall_mm", "soil_clay_content_pct",
]

select_cols = ["policy_id", "claim_count_5y", "total_incurred_5y"] + feature_cols
pdf = upt.select(*select_cols).toPandas()
pdf[feature_cols] = pdf[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

# Filter to policies with claims and compute severity
pdf = pdf[(pdf["claim_count_5y"] > 0) & (pdf["total_incurred_5y"] > 0)].copy()
pdf["avg_claim_severity"] = pdf["total_incurred_5y"] / pdf["claim_count_5y"]
pdf = pdf[pdf["avg_claim_severity"] > 0]

print(f"Training data: {len(pdf)} rows (policies with claims)")
print(f"Mean severity: £{pdf['avg_claim_severity'].mean():,.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train/test split

# COMMAND ----------

pdf["split_hash"] = pdf["policy_id"].apply(lambda x: abs(hash(x)) % 100)
train_pdf = pdf[pdf["split_hash"] < 80].copy()
test_pdf = pdf[pdf["split_hash"] >= 80].copy()

print(f"Train: {len(train_pdf)}, Test: {len(test_pdf)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Gamma GLM

# COMMAND ----------

X_train = sm.add_constant(train_pdf[feature_cols].values)
X_test = sm.add_constant(test_pdf[feature_cols].values)
y_train = train_pdf["avg_claim_severity"].values
y_test = test_pdf["avg_claim_severity"].values

with mlflow.start_run(run_name="glm_severity_gamma") as run:
    mlflow.log_param("model_type", "GLM_Gamma")
    mlflow.log_param("family", "Gamma")
    mlflow.log_param("link", "log")
    mlflow.log_param("features", len(feature_cols))
    mlflow.log_param("upt_table", upt_table_name)
    mlflow.log_param("upt_delta_version", upt_delta_version)
    mlflow.log_param("train_rows", len(train_pdf))

    try:
        input_dataset = mlflow.data.from_spark(upt, table_name=upt_table_name, version=str(upt_delta_version))
        mlflow.log_input(input_dataset, context="training")
    except Exception as e:
        print(f"Note: mlflow.data.from_spark not available — {e}")
    mlflow.set_tag("feature_table", upt_table_name)
    mlflow.log_param("test_rows", len(test_pdf))

    glm = sm.GLM(y_train, X_train, family=sm.families.Gamma(link=sm.families.links.Log()))
    glm_result = glm.fit()

    y_pred = glm_result.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("aic", glm_result.aic)

    summary_text = str(glm_result.summary())
    with open("/tmp/glm_severity_summary.txt", "w") as f:
        f.write(summary_text)
    mlflow.log_artifact("/tmp/glm_severity_summary.txt")

    # sklearn wrapper for fe.log_model()
    class GammaGLMWrapper(BaseEstimator, RegressorMixin):
        def __init__(self, glm_result, feature_names):
            self.glm_result = glm_result
            self.feature_names = feature_names
        def predict(self, X):
            return self.glm_result.predict(sm.add_constant(X))
        def fit(self, X, y):
            return self

    sklearn_model = GammaGLMWrapper(glm_result, feature_cols)

    fe = FeatureEngineeringClient()
    training_set = fe.create_training_set(
        df=spark.createDataFrame(train_pdf[["policy_id", "avg_claim_severity"]]),
        feature_lookups=[FeatureLookup(
            table_name=upt_table_name,
            feature_names=feature_cols,
            lookup_key="policy_id",
        )],
        label="avg_claim_severity",
    )

    fe.log_model(
        model=sklearn_model,
        artifact_path="glm_severity_model",
        flavor=mlflow.sklearn,
        training_set=training_set,
        registered_model_name=f"{catalog}.{schema}.glm_severity_model",
    )

    print(f"GLM Severity Results:")
    print(f"  RMSE: £{rmse:,.0f}")
    print(f"  MAE:  £{mae:,.0f}")
    print(f"  R2:   {r2:.4f}")
    print(f"  AIC:  {glm_result.aic:.1f}")
    print(f"  MLflow Run ID: {run.info.run_id}")
    print(f"  ✓ Logged with fe.log_model() — auto feature lookup enabled")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Severity Relativities

# COMMAND ----------

import math

coef_names = ["intercept"] + feature_cols
relativities = []
for i, name in enumerate(coef_names):
    coef = glm_result.params[i]
    pval = glm_result.pvalues[i]
    relativities.append({
        "feature": name,
        "coefficient": round(float(coef), 6),
        "relativity": round(math.exp(coef), 4),
        "p_value": round(float(pval), 4),
        "significant": "Yes" if pval < 0.05 else "No",
    })

display(spark.createDataFrame(relativities).orderBy("p_value"))
