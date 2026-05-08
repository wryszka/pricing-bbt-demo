# Databricks notebook source
# MAGIC %md
# MAGIC # Fraud GBM — production champion
# MAGIC
# MAGIC LightGBM binary classifier for fraud propensity at the policy level.
# MAGIC Synthetic fraud labels (~3% rate) generated deterministically from policy
# MAGIC attributes so retraining produces stable-but-slightly-drifting results.
# MAGIC Features pulled from the Modelling Mart via FeatureLookup.

# COMMAND ----------

dbutils.widgets.text("catalog_name", "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name",  "pricing_upt")
dbutils.widgets.text("run_name",     "champion")
dbutils.widgets.text("simulation_date", "")

catalog = dbutils.widgets.get("catalog_name")
schema  = dbutils.widgets.get("schema_name")
fqn     = f"{catalog}.{schema}"
run_name= dbutils.widgets.get("run_name")
sim_date= dbutils.widgets.get("simulation_date") or None

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering lightgbm shap --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog = dbutils.widgets.get("catalog_name")
schema  = dbutils.widgets.get("schema_name")
fqn     = f"{catalog}.{schema}"
run_name= dbutils.widgets.get("run_name")
sim_date= dbutils.widgets.get("simulation_date") or None

import json, hashlib
import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
from sklearn.metrics import roc_auc_score, log_loss, precision_score, recall_score
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
import pyspark.sql.functions as F

mlflow.set_registry_uri("databricks-uc")
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
mlflow.set_experiment(f"/Workspace/Users/{user}/pricing_workbench_production_fraud")

fe = FeatureEngineeringClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Synthetic fraud labels
# MAGIC
# MAGIC Deterministic score of policy attributes → logistic → ~3% positives.
# MAGIC Drivers: high CCJ count, low credit score, recent policy inception,
# MAGIC high-risk industry, unusual coverage ratios. This mimics a real fraud
# MAGIC model's label without needing a labelled fraud dataset.

# COMMAND ----------

KEY      = "policy_id"
FEATURES = [
    "sum_insured", "annual_turnover", "current_premium",
    "industry_risk_tier", "construction_type", "year_built",
    "credit_score", "ccj_count", "years_trading",
    "flood_zone_rating", "crime_theft_index",
    "urban_score", "is_coastal", "director_stability_score",
    "employee_count_est", "claim_count_5y", "total_incurred_5y",
    "open_claims_count", "distinct_perils",
]

mart = spark.table(f"{fqn}.unified_pricing_table_live")
labels_df = (
    mart
    .withColumn("_ccj_score",   F.coalesce(F.col("ccj_count"), F.lit(0)).cast("double"))
    .withColumn("_credit_score",F.coalesce(F.col("credit_score"), F.lit(600)).cast("double"))
    .withColumn("_claim_score", F.coalesce(F.col("claim_count_5y"), F.lit(0)).cast("double"))
    .withColumn("_loss_ratio",  F.coalesce(F.col("loss_ratio_5y"), F.lit(0)).cast("double"))
    # Linear predictor → high CCJ + low credit + high claims + high loss ratio → fraud
    .withColumn("_z",
        -3.5
        + F.col("_ccj_score") * 0.4
        + (600 - F.col("_credit_score")) * 0.003
        + F.col("_claim_score") * 0.20
        + F.col("_loss_ratio") * 0.05
    )
    # Deterministic Bernoulli — hash of policy_id ∈ [0,1)
    .withColumn("_rand",  (F.abs(F.hash(F.col(KEY))) % 1000000) / 1000000.0)
    .withColumn("_p",     F.expr("1.0 / (1.0 + exp(-_z))"))
    .withColumn("fraud",  (F.col("_rand") < F.col("_p")).cast("int"))
    .select(KEY, "fraud")
)
pos_rate = labels_df.filter("fraud = 1").count() / max(1, labels_df.count())
print(f"Synthetic fraud positive rate: {pos_rate:.1%}")

# COMMAND ----------

training_set = fe.create_training_set(
    df              = labels_df,
    feature_lookups = [FeatureLookup(
        table_name    = f"{fqn}.unified_pricing_table_live",
        feature_names = FEATURES,
        lookup_key    = KEY,
    )],
    label           = "fraud",
    exclude_columns = [KEY],
)
pdf = training_set.load_df().toPandas()

cat_cols = [c for c in FEATURES if pdf[c].dtype == "object"]
for c in cat_cols:
    pdf[c] = pdf[c].astype("category")

hashes = labels_df.toPandas()[KEY].apply(lambda s: abs(hash(s)) % 100).values
train_mask = hashes < 80
X = pdf[FEATURES]; y = pdf["fraud"].astype(int)
X_train, y_train = X[train_mask], y[train_mask]
X_test,  y_test  = X[~train_mask], y[~train_mask]

# COMMAND ----------

tags = {"feature_table": f"{fqn}.unified_pricing_table_live",
        "model_type": "LightGBM_binary_fraud",
        "simulated": "false", "story": "champion"}
if sim_date:
    tags["simulation_date"] = sim_date
    tags["simulated"]       = "true"

with mlflow.start_run(run_name=f"fraud_gbm_{run_name}", tags=tags) as run:
    params = dict(objective="binary",
                  metric=["binary_logloss", "auc"],
                  scale_pos_weight=float((1 - pos_rate) / max(pos_rate, 1e-4)),
                  learning_rate=0.05, num_leaves=63, min_child_samples=50,
                  feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
                  lambda_l2=1.0, verbose=-1)
    mlflow.log_params({**params, "train_rows": len(X_train), "test_rows": len(X_test),
                       "features": len(FEATURES), "positive_rate": round(pos_rate, 4)})

    train_ds = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
    valid_ds = lgb.Dataset(X_test,  label=y_test,  categorical_feature=cat_cols, reference=train_ds)
    model = lgb.train(params, train_ds, num_boost_round=400,
                      valid_sets=[train_ds, valid_ds], valid_names=["train","valid"],
                      callbacks=[lgb.early_stopping(30), lgb.log_evaluation(100)])

    y_prob = model.predict(X_test)
    y_hat  = (y_prob >= 0.5).astype(int)
    auc = float(roc_auc_score(y_test, y_prob))
    ll  = float(log_loss(y_test, np.clip(y_prob, 1e-7, 1-1e-7)))
    prec= float(precision_score(y_test, y_hat, zero_division=0))
    rec = float(recall_score(y_test, y_hat, zero_division=0))
    gini= 2 * auc - 1

    mlflow.log_metrics({"auc": auc, "logloss": ll, "precision": prec, "recall": rec, "gini": gini,
                        "best_iteration": float(model.best_iteration or 0)})
    print(f"AUC={auc:.4f}  Precision={prec:.3f}  Recall={rec:.3f}")

    imp = pd.DataFrame({"feature": model.feature_name(),
                        "gain":    model.feature_importance(importance_type="gain")})
    imp.to_csv("/tmp/fraud_importance.csv", index=False)
    mlflow.log_artifact("/tmp/fraud_importance.csv")

    # SHAP explanations — log summary plot + per-feature mean-abs SHAP CSV
    # so the governance pack and Review & Promote UI can surface both.
    try:
        import shap, matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sample = X_test.sample(min(1000, len(X_test)), random_state=42)
        explainer  = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
        # Binary classifier returns (n, features, 2) in newer SHAP or a
        # 2-element list in older — normalise to class-1 contributions.
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
            shap_values = shap_values[..., 1]

        shap.summary_plot(shap_values, sample, show=False, max_display=15)
        plt.tight_layout()
        plt.savefig("/tmp/fraud_shap_summary.png", bbox_inches="tight", dpi=130)
        plt.close()
        mlflow.log_artifact("/tmp/fraud_shap_summary.png")

        mean_abs = np.abs(shap_values).mean(axis=0)
        shap_df  = pd.DataFrame({"feature": list(sample.columns),
                                 "mean_abs_shap": mean_abs}) \
                    .sort_values("mean_abs_shap", ascending=False)
        shap_df.to_csv("/tmp/fraud_shap_importance.csv", index=False)
        mlflow.log_artifact("/tmp/fraud_shap_importance.csv")
        print(f"SHAP logged: top feature = {shap_df.iloc[0]['feature']}")
    except Exception as e:
        print(f"SHAP computation failed: {e}")

    from mlflow.models.signature import infer_signature
    sample_X    = X_train.head(5).copy()
    sample_pred = model.predict(sample_X)
    signature   = infer_signature(sample_X, sample_pred)

    fe.log_model(
        model                 = model,
        artifact_path         = "model",
        flavor                = mlflow.lightgbm,
        training_set          = training_set,
        registered_model_name = f"{fqn}.fraud_gbm",
        signature             = signature,
        input_example         = sample_X,
    )
    print(f"UC model: {fqn}.fraud_gbm")

try:
    det = json.dumps({"auc": auc, "precision": prec, "recall": rec, "gini": gini,
                      "mlflow_run_id": run.info.run_id,
                      "simulated": bool(sim_date), "simulation_date": sim_date, "story": run_name}).replace("'", "''")
    spark.sql(f"""
        INSERT INTO {fqn}.audit_log
          (event_id, event_type, entity_type, entity_id, entity_version, user_id, timestamp, details, source)
        SELECT uuid(), 'model_trained', 'model', 'fraud_gbm', '{run_name}', '{user}',
               current_timestamp(), '{det}', 'notebook'
    """)
except Exception as e:
    print(f"audit: {e}")
