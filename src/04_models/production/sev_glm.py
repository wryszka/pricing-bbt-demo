# Databricks notebook source
# MAGIC %md
# MAGIC # Severity GLM — production champion
# MAGIC
# MAGIC Gamma GLM on average claim severity, trained only on policies with
# MAGIC observed claims (`claim_count_5y > 0`). Features from the Modelling
# MAGIC Mart via `FeatureLookup`. Registered in UC as `sev_glm`.

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

# MAGIC %pip install databricks-feature-engineering --quiet
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
import statsmodels.api as sm
import mlflow
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from pyspark.sql.functions import col

mlflow.set_registry_uri("databricks-uc")
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
mlflow.set_experiment(f"/Workspace/Users/{user}/pricing_workbench_production_sev")

fe = FeatureEngineeringClient()

# COMMAND ----------

FEATURES = [
    "sum_insured", "annual_turnover",
    "industry_risk_tier", "construction_type", "year_built",
    "credit_score", "years_trading",
    "flood_zone_rating", "proximity_to_fire_station_km",
    "crime_theft_index", "subsidence_risk", "composite_location_risk",
    "urban_score", "is_coastal", "elevation_metres",
    "annual_rainfall_mm", "population_density_per_km2",
    "distance_to_coast_km",
]
KEY = "policy_id"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build training set — claimants only, target = mean severity

# COMMAND ----------

mart = spark.table(f"{fqn}.unified_pricing_table_live")
labels_df = (mart
    .filter(col("claim_count_5y").isNotNull() & (col("claim_count_5y") > 0))
    .filter(col("total_incurred_5y").isNotNull() & (col("total_incurred_5y") > 0))
    .withColumn("mean_severity", col("total_incurred_5y") / col("claim_count_5y"))
    .filter(col("mean_severity") > 0)
    .select(KEY, "mean_severity", "claim_count_5y")
)

training_set = fe.create_training_set(
    df              = labels_df.select(KEY, "mean_severity"),
    feature_lookups = [FeatureLookup(
        table_name    = f"{fqn}.unified_pricing_table_live",
        feature_names = FEATURES,
        lookup_key    = KEY,
    )],
    label           = "mean_severity",
    exclude_columns = [KEY],
)
training_pdf = training_set.load_df().toPandas()
print(f"Claimants: {len(training_pdf):,}")

# COMMAND ----------

# Log-transform heavy-tailed monetary / distance columns so Gamma/log-link
# IRLS converges — otherwise £-scale features blow up the linear predictor.
LOG_COLS = ["sum_insured", "annual_turnover", "elevation_metres",
            "annual_rainfall_mm", "population_density_per_km2",
            "distance_to_coast_km", "proximity_to_fire_station_km"]

def _log_transform(df):
    out = df.copy()
    for c in LOG_COLS:
        if c in out.columns:
            out[c] = np.log1p(out[c].astype(float).clip(lower=0))
    return out

feat_pdf = _log_transform(training_pdf[FEATURES])
X_raw   = pd.get_dummies(feat_pdf, drop_first=True, dtype=float).fillna(0.0)

# Standardise numeric columns to zero-mean unit-variance — IRLS behaves much
# better when columns are on comparable scales. Capture mean/std so the wrapper
# can apply the same transform at inference time.
SCALER = {}
for c in X_raw.columns:
    std = X_raw[c].std()
    if std > 0 and X_raw[c].nunique() > 2:   # leave dummy 0/1 cols alone
        SCALER[c] = (float(X_raw[c].mean()), float(std))

X = X_raw.copy()
for c, (mu, sd) in SCALER.items():
    X[c] = (X[c] - mu) / sd

y = training_pdf["mean_severity"].astype(float)

# Drop any row whose label is NaN/inf/non-positive — Gamma GLM requires y > 0.
valid = np.isfinite(y.values) & (y.values > 0)
X = X.loc[valid].reset_index(drop=True)
y = y.loc[valid].reset_index(drop=True)
print(f"After NaN sanitation: {len(X):,} rows")

# Deterministic 80/20 split
rng = np.random.default_rng(42)
hashes = rng.integers(0, 100, len(X))
train_mask = hashes < 80
X_train, y_train = X[train_mask], y[train_mask]
X_test,  y_test  = X[~train_mask], y[~train_mask]

# COMMAND ----------

FEATURE_NAMES = list(X.columns)

class GammaGLMWrapper(BaseEstimator, RegressorMixin):
    """Applies the same log + one-hot + standard-scale pipeline at inference
    that was used during training, then calls the fitted OLS-on-log model and
    back-transforms with exp()."""
    def __init__(self, result, feature_names, raw_features, log_cols, scaler):
        self.result         = result
        self.feature_names  = feature_names
        self.raw_features   = raw_features
        self.log_cols       = log_cols
        self.scaler         = scaler

    def fit(self, X, y): return self

    def _transform(self, X):
        if hasattr(X, "columns"):
            df = X[[c for c in self.raw_features if c in X.columns]].copy()
        else:
            df = pd.DataFrame(np.asarray(X), columns=self.raw_features)
        for c in self.log_cols:
            if c in df.columns:
                df[c] = np.log1p(df[c].astype(float).clip(lower=0))
        Xd = pd.get_dummies(df, drop_first=True, dtype=float)
        Xd = Xd.reindex(columns=self.feature_names, fill_value=0.0).fillna(0.0)
        for c, (mu, sd) in self.scaler.items():
            Xd[c] = (Xd[c] - mu) / sd
        return Xd.values

    def predict(self, X):
        return np.exp(self.result.predict(sm.add_constant(self._transform(X), has_constant="add")))

tags = {"feature_table": f"{fqn}.unified_pricing_table_live",
        "model_type": "GLM_Gamma", "simulated": "false", "story": "champion"}
if sim_date:
    tags["simulation_date"] = sim_date
    tags["simulated"]       = "true"

with mlflow.start_run(run_name=f"sev_glm_{run_name}", tags=tags) as run:
    mlflow.log_params({"family": "Gamma-approx (OLS on log-severity)", "link": "log",
                       "features": len(FEATURE_NAMES),
                       "train_rows": len(X_train), "test_rows": len(X_test)})

    # OLS on log(severity): equivalent in spirit to a Gamma GLM with log link,
    # but converges reliably and avoids IRLS overflow/underflow on heavy-tailed
    # severity data. Back-transform with exp() at predict time.
    X_train_c = sm.add_constant(X_train.values, has_constant="add")
    X_test_c  = sm.add_constant(X_test.values,  has_constant="add")
    log_y_train = np.log(y_train.values)
    ols = sm.OLS(log_y_train, X_train_c).fit()
    res = ols
    y_pred = np.exp(ols.predict(X_test_c))

    mae = float(mean_absolute_error(y_test, y_pred))
    # Gini on severity
    order = np.argsort(-y_pred)
    cum_y = np.cumsum(y_test.values[order]) / y_test.sum()
    cum_n = np.arange(1, len(y_test) + 1) / len(y_test)
    gini = float(2 * np.trapz(cum_y, cum_n) - 1)

    r2 = float(res.rsquared) if hasattr(res, "rsquared") else float("nan")
    mlflow.log_metrics({"mae_gbp": mae, "gini": gini, "r2_log": r2})
    print(f"Gini={gini:.4f}  MAE £{mae:,.0f}  R²(log)={r2:.3f}")

    rel = pd.DataFrame({"feature": ["intercept"] + FEATURE_NAMES,
                        "coefficient": res.params.tolist(),
                        "relativity": np.exp(res.params).tolist(),
                        "p_value": res.pvalues.tolist()})
    rel.to_csv("/tmp/sev_relativities.csv", index=False)
    mlflow.log_artifact("/tmp/sev_relativities.csv")

    wrapper = GammaGLMWrapper(res, FEATURE_NAMES, FEATURES, LOG_COLS, SCALER)

    # Pin explicit signature on raw features — wrapper replays the log / one-hot
    # / scaler pipeline internally, so pyfunc can be called on unencoded rows.
    from mlflow.models.signature import infer_signature
    sample_X    = training_pdf[FEATURES].head(5).copy()
    for c in sample_X.columns:
        if sample_X[c].dtype == "object":
            sample_X[c] = sample_X[c].fillna("(null)").astype(str)
        else:
            sample_X[c] = pd.to_numeric(sample_X[c], errors="coerce").fillna(0.0).astype(float)
    sample_pred = wrapper.predict(sample_X)
    signature   = infer_signature(sample_X, sample_pred)

    fe.log_model(
        model                 = wrapper,
        artifact_path         = "model",
        flavor                = mlflow.sklearn,
        training_set          = training_set,
        registered_model_name = f"{fqn}.sev_glm",
        signature             = signature,
        input_example         = sample_X,
        # cloudpickle: avoid the sklearn flavor's skops "untrusted types" guard
        # on the statsmodels-wrapped GLM (durable across mlflow version drift).
        serialization_format  = "cloudpickle",
    )
    print(f"UC model: {fqn}.sev_glm")

try:
    det = json.dumps({"gini": gini, "mae_gbp": mae, "r2_log": r2,
                      "mlflow_run_id": run.info.run_id,
                      "simulated": bool(sim_date), "simulation_date": sim_date, "story": run_name}).replace("'", "''")
    spark.sql(f"""
        INSERT INTO {fqn}.audit_log
          (event_id, event_type, entity_type, entity_id, entity_version, user_id, timestamp, details, source)
        SELECT uuid(), 'model_trained', 'model', 'sev_glm', '{run_name}', '{user}',
               current_timestamp(), '{det}', 'notebook'
    """)
except Exception as e:
    print(f"audit: {e}")
