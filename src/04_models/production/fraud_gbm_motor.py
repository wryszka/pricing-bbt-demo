# Databricks notebook source
# MAGIC %md
# MAGIC # Motor Fraud GBM — production champion
# MAGIC
# MAGIC LightGBM binary classifier on a synthetic fraud label that's a
# MAGIC deterministic function of telematics behaviour + prior history.
# MAGIC The behaviour_score and recent_event counts dominate — exactly what
# MAGIC the live-serving demo needs the prediction to move on when John's
# MAGIC black box reports a new event.
# MAGIC Registered as `{fqn}.fraud_gbm_motor`.

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
mlflow.set_experiment(f"/Workspace/Users/{user}/pricing_workbench_production_motor_fraud")
fe = FeatureEngineeringClient()

# COMMAND ----------

KEY = "policy_id"
FEATURES = [
    "driver_age", "license_years_held", "no_claims_years",
    "gender", "occupation_class",
    "vehicle_group", "vehicle_value", "vehicle_age", "annual_mileage",
    "prior_convictions", "prior_accidents_5y",
    "behaviour_score", "avg_speed_mph", "night_driving_pct",
    "recent_speeding_events", "recent_curfew_breaches", "recent_harsh_braking_30d",
    "telematics_recent_event_count",
    "claim_count_5y", "at_fault_count_5y", "open_claims_count", "distinct_perils",
]

upt_table = f"{fqn}.unified_motor_table_live"
mart = spark.table(upt_table)

# Synthetic fraud label — telematics signal dominates, with NCD inversely correlated
labels_df = (mart
    .withColumn("_bs",     F.coalesce(F.col("behaviour_score").cast("double"),    F.lit(80.0)))
    .withColumn("_rec",    F.coalesce(F.col("telematics_recent_event_count").cast("double"), F.lit(0.0)))
    .withColumn("_prior",  F.coalesce(F.col("prior_convictions").cast("double"),  F.lit(0.0)))
    .withColumn("_ncd",    F.coalesce(F.col("no_claims_years").cast("double"),    F.lit(0.0)))
    .withColumn("_age",    F.col("driver_age").cast("double"))
    .withColumn("_z",
        -3.0
        + (80 - F.col("_bs")) * 0.06
        + F.col("_rec") * 0.30
        + F.col("_prior") * 0.40
        - F.col("_ncd") * 0.08
        + F.expr("CASE WHEN _age < 25 THEN 0.5 ELSE 0 END")
    )
    .withColumn("_rand", (F.abs(F.hash(F.col(KEY))) % 1000000) / 1000000.0)
    .withColumn("_p",    F.expr("1.0 / (1.0 + exp(-_z))"))
    .withColumn("fraud", (F.col("_rand") < F.col("_p")).cast("int"))
    .select(KEY, "fraud")
)
pos_rate = labels_df.filter("fraud = 1").count() / max(1, labels_df.count())
print(f"Synthetic fraud positive rate: {pos_rate:.1%}")

# COMMAND ----------

training_set = fe.create_training_set(
    df              = labels_df,
    feature_lookups = [FeatureLookup(table_name=upt_table, feature_names=FEATURES, lookup_key=KEY)],
    label           = "fraud",
    exclude_columns = [KEY],
)
# Sample 20% for tractable LightGBM fit on 1M source
pdf = training_set.load_df().sample(0.20, seed=42).toPandas()
print(f"Training set: {len(pdf):,}")

cat_cols = [c for c in FEATURES if pdf[c].dtype == "object"]
for c in cat_cols:
    pdf[c] = pdf[c].astype("category")
for c in FEATURES:
    if pdf[c].dtype == "bool":
        pdf[c] = pdf[c].astype(int)

# Deterministic split from the same pdf
rng = np.random.default_rng(7)
mask = rng.integers(0, 100, len(pdf)) < 80
X = pdf[FEATURES]; y = pdf["fraud"].astype(int)
X_train, y_train = X[mask], y[mask]
X_test,  y_test  = X[~mask], y[~mask]
print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")

# COMMAND ----------

tags = {"feature_table": upt_table, "model_type": "LightGBM_binary_motor_fraud", "story": "champion"}

with mlflow.start_run(run_name=f"fraud_gbm_motor_{run_name}", tags=tags) as run:
    params = dict(objective="binary", metric=["binary_logloss", "auc"],
                  scale_pos_weight=float((1 - pos_rate) / max(pos_rate, 1e-4)),
                  learning_rate=0.06, num_leaves=63, min_child_samples=100,
                  feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
                  lambda_l2=1.0, verbose=-1)
    mlflow.log_params({**params, "train_rows": len(X_train), "test_rows": len(X_test),
                       "features": len(FEATURES), "positive_rate": round(pos_rate, 4)})

    train_ds = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
    valid_ds = lgb.Dataset(X_test,  label=y_test,  categorical_feature=cat_cols, reference=train_ds)
    model = lgb.train(params, train_ds, num_boost_round=300,
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
                 registered_model_name=f"{fqn}.fraud_gbm_motor",
                 signature=signature, input_example=sample_X)
    print(f"UC model: {fqn}.fraud_gbm_motor")

from mlflow.tracking import MlflowClient
mc = MlflowClient()
latest = max(int(v.version) for v in mc.search_model_versions(f"name='{fqn}.fraud_gbm_motor'"))
mc.set_registered_model_alias(name=f"{fqn}.fraud_gbm_motor", alias="champion", version=str(latest))
print(f"alias 'champion' → v{latest}")

dbutils.notebook.exit(json.dumps({"model": f"{fqn}.fraud_gbm_motor", "version": latest, "auc": auc}))
