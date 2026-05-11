# Databricks notebook source
# MAGIC %md
# MAGIC # Motor Demand GBM — production champion
# MAGIC
# MAGIC LightGBM binary classifier on a synthetic "quote_accepted" label.
# MAGIC Predicts the probability a customer will accept the quote at their
# MAGIC current_premium given their attributes. Used by the live serving
# MAGIC rating engine to apply a small +/- adjustment: lower demand → small
# MAGIC discount to win the renewal, higher demand → small loading.
# MAGIC
# MAGIC Registered in UC as `{fqn}.demand_gbm_motor` with the `champion` alias.
# MAGIC
# MAGIC Signal mix: driver demographics, vehicle group, premium positioning
# MAGIC (current_premium relative to vehicle_value), behaviour_score. Price-
# MAGIC competitiveness terms dominate, NCD years and behaviour_score act as
# MAGIC loyalty proxies.

# COMMAND ----------

dbutils.widgets.text("catalog_name", "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name",  "pricing_upt")
dbutils.widgets.text("run_name",     "champion")

catalog = dbutils.widgets.get("catalog_name")
schema  = dbutils.widgets.get("schema_name")
fqn     = f"{catalog}.{schema}"
run_name= dbutils.widgets.get("run_name")

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering lightgbm --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog = dbutils.widgets.get("catalog_name")
schema  = dbutils.widgets.get("schema_name")
fqn     = f"{catalog}.{schema}"
run_name= dbutils.widgets.get("run_name")

import json, numpy as np, pandas as pd, lightgbm as lgb, mlflow
from sklearn.metrics import roc_auc_score, log_loss
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
import pyspark.sql.functions as F

mlflow.set_registry_uri("databricks-uc")
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
mlflow.set_experiment(f"/Workspace/Users/{user}/pricing_workbench_production_motor_demand")
fe = FeatureEngineeringClient()

# COMMAND ----------

KEY = "policy_id"
FEATURES = [
    "driver_age", "license_years_held", "no_claims_years",
    "gender", "marital_status", "occupation_class",
    "vehicle_group", "vehicle_value", "vehicle_age", "fuel_type",
    "annual_mileage", "parking_overnight", "business_use",
    "current_premium",                       # premium positioning
    "behaviour_score",                       # loyalty proxy
    "claim_count_5y", "at_fault_count_5y",
]

upt_table = f"{fqn}.unified_motor_table_live"
mart = spark.table(upt_table)

# Synthetic acceptance label — high acceptance when price-per-£SI is low and
# customer has loyalty signals (high NCD, high behaviour_score).
labels_df = (mart
    .withColumn("_age",   F.col("driver_age").cast("double"))
    .withColumn("_ncd",   F.coalesce(F.col("no_claims_years").cast("double"), F.lit(0.0)))
    .withColumn("_prem",  F.coalesce(F.col("current_premium").cast("double"), F.lit(1500.0)))
    .withColumn("_val",   F.coalesce(F.col("vehicle_value").cast("double"),   F.lit(10000.0)))
    .withColumn("_bs",    F.coalesce(F.col("behaviour_score").cast("double"),  F.lit(75.0)))
    # Price competitiveness: lower is "cheaper" (better for acceptance)
    .withColumn("_ppk",   F.col("_prem") / (F.col("_val") / F.lit(1000.0)))
    .withColumn("_z",
        F.lit(1.5)
        - F.col("_ppk") * 0.005           # cheaper rate → higher acceptance
        + F.col("_ncd") * 0.10            # loyal customers accept more
        + (F.col("_bs") - F.lit(70)) * 0.015  # better behaviour → loyalty
        - F.expr("CASE WHEN _age < 22 THEN 0.6 ELSE 0 END")  # young drivers shop around
    )
    .withColumn("_rand", (F.abs(F.hash(F.col(KEY), F.lit("dem"))) % 1000000) / 1000000.0)
    .withColumn("_p",    F.expr("1.0 / (1.0 + exp(-_z))"))
    .withColumn("accepted", (F.col("_rand") < F.col("_p")).cast("int"))
    .select(KEY, "accepted")
)
acc_rate = labels_df.filter("accepted = 1").count() / max(1, labels_df.count())
print(f"Synthetic acceptance rate: {acc_rate:.1%}")

# COMMAND ----------

# Keep policy_id in pdf so the train/test mask aligns with the FE-resolved rows.
labels_df = labels_df.sample(0.20, seed=42)
training_set = fe.create_training_set(
    df              = labels_df,
    feature_lookups = [FeatureLookup(table_name=upt_table, feature_names=FEATURES, lookup_key=KEY)],
    label           = "accepted",
)
pdf = training_set.load_df().toPandas()
print(f"Training set: {len(pdf):,}")

cat_cols = [c for c in FEATURES if pdf[c].dtype == "object"]
for c in cat_cols:
    pdf[c] = pdf[c].astype("category")
for c in FEATURES:
    if pdf[c].dtype == "bool":
        pdf[c] = pdf[c].astype(int)

mask = pdf[KEY].apply(lambda s: abs(hash(s)) % 100 < 80).values
X = pdf[FEATURES]; y = pdf["accepted"].astype(int)
X_train, y_train = X[mask], y[mask]
X_test,  y_test  = X[~mask], y[~mask]
print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")

# COMMAND ----------

tags = {"feature_table": upt_table, "model_type": "LightGBM_binary_motor_demand",
        "story": "champion"}

with mlflow.start_run(run_name=f"demand_gbm_motor_{run_name}", tags=tags) as run:
    params = dict(objective="binary", metric=["binary_logloss", "auc"],
                  learning_rate=0.05, num_leaves=63, min_child_samples=100,
                  feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
                  lambda_l2=1.0, verbose=-1)
    mlflow.log_params({**params, "train_rows": len(X_train), "test_rows": len(X_test),
                       "features": len(FEATURES), "acceptance_rate": round(acc_rate, 4)})

    train_ds = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
    valid_ds = lgb.Dataset(X_test,  label=y_test,  categorical_feature=cat_cols, reference=train_ds)
    model = lgb.train(params, train_ds, num_boost_round=400,
                      valid_sets=[train_ds, valid_ds], valid_names=["train","valid"],
                      callbacks=[lgb.early_stopping(30), lgb.log_evaluation(100)])

    y_prob = model.predict(X_test)
    auc = float(roc_auc_score(y_test, y_prob))
    ll  = float(log_loss(y_test, np.clip(y_prob, 1e-7, 1-1e-7)))
    mlflow.log_metrics({"auc": auc, "logloss": ll, "gini": 2*auc-1,
                        "best_iteration": float(model.best_iteration or 0)})
    print(f"AUC={auc:.4f}")

    from mlflow.models.signature import infer_signature
    sample_X    = X_train.head(5).copy()
    sample_pred = model.predict(sample_X)
    signature   = infer_signature(sample_X, sample_pred)

    fe.log_model(model=model, artifact_path="model", flavor=mlflow.lightgbm,
                 training_set=training_set,
                 registered_model_name=f"{fqn}.demand_gbm_motor",
                 signature=signature, input_example=sample_X)

from mlflow.tracking import MlflowClient
mc = MlflowClient()
latest = max(int(v.version) for v in mc.search_model_versions(f"name='{fqn}.demand_gbm_motor'"))
mc.set_registered_model_alias(name=f"{fqn}.demand_gbm_motor", alias="champion", version=str(latest))
print(f"alias 'champion' → v{latest}")

dbutils.notebook.exit(json.dumps({"model": f"{fqn}.demand_gbm_motor", "version": latest, "auc": auc}))
