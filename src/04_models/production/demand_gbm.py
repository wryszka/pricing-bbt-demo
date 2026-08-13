# Databricks notebook source
# MAGIC %md
# MAGIC # Demand GBM — production champion
# MAGIC
# MAGIC LightGBM binary classifier on `quotes.converted` — predicts probability a
# MAGIC quote converts to a bound policy. First registers the `quotes` table as
# MAGIC an offline feature table (idempotent), then trains via FeatureLookup so
# MAGIC lineage is captured.

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

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
from sklearn.metrics import roc_auc_score, log_loss
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
import pyspark.sql.functions as F

mlflow.set_registry_uri("databricks-uc")
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
mlflow.set_experiment(f"/Workspace/Users/{user}/pricing_workbench_production_demand")

fe = FeatureEngineeringClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register `quotes` as a feature table (idempotent)

# COMMAND ----------

QUOTES_TABLE = f"{fqn}.quotes"
try:
    fe.get_table(name=QUOTES_TABLE)
    print("quotes already registered as FE table")
except Exception:
    print("Registering quotes as a feature table…")
    # FE requires the source table to have a PRIMARY KEY constraint on the
    # lookup key. Add NOT NULL + PK idempotently before registering.
    spark.sql(f"ALTER TABLE {QUOTES_TABLE} ALTER COLUMN transaction_id SET NOT NULL")
    try:
        spark.sql(f"ALTER TABLE {QUOTES_TABLE} ADD CONSTRAINT quotes_pk PRIMARY KEY(transaction_id)")
    except Exception as e:
        if "already exists" not in str(e).lower():
            raise
    # Build a feature-only view (drop the label + IDs that aren't features)
    quotes_df = spark.table(QUOTES_TABLE)
    feat_cols = [c for c in quotes_df.columns if c not in ("converted", "quote_status", "policy_id")]
    feat_df = quotes_df.select(*feat_cols)
    fe.create_table(
        name          = QUOTES_TABLE,
        primary_keys  = "transaction_id",
        df            = feat_df,
        description   = "Quote-level features for demand/conversion modelling.",
    )
    print("Registered.")

# COMMAND ----------

FEATURES = [
    "channel", "region", "construction_type", "flood_zone",
    "year_built", "floor_area_sqm",
    "buildings_si", "contents_si", "liability_si", "voluntary_excess",
    "gross_premium_quoted", "log_gross_premium", "log_buildings_si",
    "rate_per_1k_si", "vs_market_rate",
    # Price-elasticity signal emitted by the quote generator (offer vs market).
    "gross_premium", "market_premium",
    "market_median_rate", "competitor_a_min_rate", "price_index",
    "annual_turnover", "credit_score",
    "flood_zone_rating", "crime_theft_index",
    "sprinklered", "alarmed",
]
# Some of these cols may not exist in quotes; filter to those that do.
quotes_cols = set(spark.table(QUOTES_TABLE).columns)
FEATURES = [f for f in FEATURES if f in quotes_cols]
print(f"Using {len(FEATURES)} features from quotes")

# COMMAND ----------

labels_df = spark.table(QUOTES_TABLE).select(
    "transaction_id",
    F.when(F.col("converted").cast("string").isin("Y", "1", "true", "True"), 1).otherwise(0).alias("converted"),
)
training_set = fe.create_training_set(
    df              = labels_df,
    feature_lookups = [FeatureLookup(table_name=QUOTES_TABLE, feature_names=FEATURES, lookup_key="transaction_id")],
    label           = "converted",
    # Keep transaction_id on the pandas frame so the train/test mask stays
    # aligned with the loaded rows (FeatureLookup may drop unmatched rows;
    # deriving the mask from the quotes table directly causes a length mismatch).
)
pdf = training_set.load_df().toPandas()
print(f"Rows: {len(pdf):,}  Conversion rate: {pdf['converted'].mean():.1%}")

# Encode for LightGBM (categoricals → pandas category)
cat_cols = [c for c in FEATURES if pdf[c].dtype == "object"]
for c in cat_cols:
    pdf[c] = pdf[c].astype("category")
for c in FEATURES:
    if pdf[c].dtype == "bool":
        pdf[c] = pdf[c].astype(int)

# Build train/test mask from the SAME pdf that was loaded — guarantees alignment.
train_mask = pdf["transaction_id"].apply(lambda s: abs(hash(s)) % 100 < 80).values
X = pdf[FEATURES]
y = pdf["converted"].astype(int)
X_train, y_train = X[train_mask], y[train_mask]
X_test,  y_test  = X[~train_mask], y[~train_mask]
print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")

# COMMAND ----------

tags = {"feature_table": QUOTES_TABLE, "model_type": "LightGBM_binary",
        "simulated": "false", "story": "champion"}
if sim_date:
    tags["simulation_date"] = sim_date
    tags["simulated"]       = "true"

with mlflow.start_run(run_name=f"demand_gbm_{run_name}", tags=tags) as run:
    params = dict(objective="binary", metric=["binary_logloss", "auc"],
                  learning_rate=0.05, num_leaves=63, min_child_samples=100,
                  feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
                  lambda_l2=1.0, verbose=-1)
    mlflow.log_params(params | {"train_rows": len(X_train), "test_rows": len(X_test), "features": len(FEATURES)})

    train_ds = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
    valid_ds = lgb.Dataset(X_test,  label=y_test,  categorical_feature=cat_cols, reference=train_ds)
    model = lgb.train(params, train_ds, num_boost_round=600,
                      valid_sets=[train_ds, valid_ds], valid_names=["train","valid"],
                      callbacks=[lgb.early_stopping(40), lgb.log_evaluation(100)])

    y_pred = model.predict(X_test)
    auc = float(roc_auc_score(y_test, y_pred))
    ll  = float(log_loss(y_test, np.clip(y_pred, 1e-7, 1-1e-7)))
    # Gini on binary target
    gini = 2 * auc - 1

    mlflow.log_metrics({"auc": auc, "logloss": ll, "gini": gini,
                        "best_iteration": float(model.best_iteration or 0)})
    print(f"AUC={auc:.4f}  LogLoss={ll:.4f}  Gini={gini:.4f}")

    # Feature importance
    imp = pd.DataFrame({"feature": model.feature_name(),
                        "gain":    model.feature_importance(importance_type="gain")})
    imp.to_csv("/tmp/demand_importance.csv", index=False)
    mlflow.log_artifact("/tmp/demand_importance.csv")

    # SHAP explanations — log summary plot + per-feature mean-abs SHAP CSV.
    try:
        import shap, matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sample = X_test.sample(min(1000, len(X_test)), random_state=42)
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
            shap_values = shap_values[..., 1]

        shap.summary_plot(shap_values, sample, show=False, max_display=15)
        plt.tight_layout()
        plt.savefig("/tmp/demand_shap_summary.png", bbox_inches="tight", dpi=130)
        plt.close()
        mlflow.log_artifact("/tmp/demand_shap_summary.png")

        mean_abs = np.abs(shap_values).mean(axis=0)
        shap_df  = pd.DataFrame({"feature": list(sample.columns),
                                 "mean_abs_shap": mean_abs}) \
                    .sort_values("mean_abs_shap", ascending=False)
        shap_df.to_csv("/tmp/demand_shap_importance.csv", index=False)
        mlflow.log_artifact("/tmp/demand_shap_importance.csv")
        print(f"SHAP logged: top feature = {shap_df.iloc[0]['feature']}")
    except Exception as e:
        print(f"SHAP computation failed: {e}")

    # Pin an explicit signature so serving / Compare & Test can pass plain
    # rows from the quotes table — object cols become "category" in training,
    # so reflect that in the input example and let MLflow infer types from it.
    from mlflow.models.signature import infer_signature
    sample_X    = X_train.head(5).copy()
    sample_pred = model.predict(sample_X)
    signature   = infer_signature(sample_X, sample_pred)

    fe.log_model(
        model                 = model,
        artifact_path         = "model",
        flavor                = mlflow.lightgbm,
        training_set          = training_set,
        registered_model_name = f"{fqn}.demand_gbm",
        signature             = signature,
        input_example         = sample_X,
    )
    print(f"UC model: {fqn}.demand_gbm")

try:
    det = json.dumps({"auc": auc, "logloss": ll, "gini": gini,
                      "mlflow_run_id": run.info.run_id,
                      "simulated": bool(sim_date), "simulation_date": sim_date, "story": run_name}).replace("'", "''")
    spark.sql(f"""
        INSERT INTO {fqn}.audit_log
          (event_id, event_type, entity_type, entity_id, entity_version, user_id, timestamp, details, source)
        SELECT uuid(), 'model_trained', 'model', 'demand_gbm', '{run_name}', '{user}',
               current_timestamp(), '{det}', 'notebook'
    """)
except Exception as e:
    print(f"audit: {e}")
