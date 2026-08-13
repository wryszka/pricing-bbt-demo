# Databricks notebook source
# MAGIC %md
# MAGIC # Motor Severity GLM — production champion
# MAGIC
# MAGIC Gamma GLM on motor mean claim severity, trained only on claimants
# MAGIC (`claim_count_5y > 0`). Registered as `{fqn}.sev_glm_motor`.

# COMMAND ----------

dbutils.widgets.text("catalog_name", "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name",  "pricing_upt")
dbutils.widgets.text("run_name",     "champion")

catalog = dbutils.widgets.get("catalog_name")
schema  = dbutils.widgets.get("schema_name")
fqn     = f"{catalog}.{schema}"
run_name= dbutils.widgets.get("run_name")

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog = dbutils.widgets.get("catalog_name")
schema  = dbutils.widgets.get("schema_name")
fqn     = f"{catalog}.{schema}"
run_name= dbutils.widgets.get("run_name")

import json, numpy as np, pandas as pd, statsmodels.api as sm, mlflow
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from pyspark.sql.functions import col

mlflow.set_registry_uri("databricks-uc")
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
mlflow.set_experiment(f"/Workspace/Users/{user}/pricing_workbench_production_motor_sev")
fe = FeatureEngineeringClient()

# COMMAND ----------

FEATURES = [
    "driver_age", "license_years_held",
    "vehicle_group", "vehicle_value", "vehicle_age", "fuel_type",
    "annual_mileage", "parking_overnight",
    "behaviour_score", "avg_speed_mph",
    "recent_speeding_events", "recent_harsh_braking_30d",
    "at_fault_count_5y", "distinct_perils",
]
KEY = "policy_id"

upt_table = f"{fqn}.unified_motor_table_live"
mart = spark.table(upt_table)
labels_df = (mart
    .filter(col("claim_count_5y").isNotNull() & (col("claim_count_5y") > 0))
    .filter(col("total_incurred_5y").isNotNull() & (col("total_incurred_5y") > 0))
    .withColumn("mean_severity", col("total_incurred_5y") / col("claim_count_5y"))
    .filter(col("mean_severity") > 0)
    .select(KEY, "mean_severity")
)

training_set = fe.create_training_set(
    df              = labels_df,
    feature_lookups = [FeatureLookup(table_name=upt_table, feature_names=FEATURES, lookup_key=KEY)],
    label           = "mean_severity",
    exclude_columns = [KEY],
)
training_pdf = training_set.load_df().toPandas()
print(f"Claimants: {len(training_pdf):,}")

# COMMAND ----------

LOG_COLS = ["vehicle_value", "annual_mileage", "avg_speed_mph"]

def _log_transform(df):
    out = df.copy()
    for c in LOG_COLS:
        if c in out.columns:
            out[c] = np.log1p(out[c].astype(float).clip(lower=0))
    return out

def _prep(df):
    out = df[FEATURES].copy()
    for c in out.columns:
        if out[c].dtype == "object":
            out[c] = out[c].astype(str).where(out[c].notna(), "(null)")
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).astype(float)
    return out

feat = _log_transform(_prep(training_pdf))
X_raw = pd.get_dummies(feat, drop_first=True, dtype=float).fillna(0.0)

# Standardise non-binary cols
SCALER = {}
for c in X_raw.columns:
    std = X_raw[c].std()
    if std > 0 and X_raw[c].nunique() > 2:
        SCALER[c] = (float(X_raw[c].mean()), float(std))
X = X_raw.copy()
for c, (mu, sd) in SCALER.items():
    X[c] = (X[c] - mu) / sd

y = training_pdf["mean_severity"].astype(float)
valid = np.isfinite(y.values) & (y.values > 0)
X = X.loc[valid].reset_index(drop=True)
y = y.loc[valid].reset_index(drop=True)

rng = np.random.default_rng(42)
hashes = rng.integers(0, 100, len(X))
mask = hashes < 80
X_train, y_train = X[mask], y[mask]
X_test,  y_test  = X[~mask], y[~mask]
print(f"Train: {len(X_train):,}   Test: {len(X_test):,}")

# COMMAND ----------

FEATURE_NAMES = list(X.columns)

class GammaGLMWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, result, feature_names, raw_features, log_cols, scaler):
        self.result=result; self.feature_names=feature_names; self.raw_features=raw_features
        self.log_cols=log_cols; self.scaler=scaler
    def fit(self, X, y): return self
    def _transform(self, X):
        if hasattr(X, "columns"):
            df = X[[c for c in self.raw_features if c in X.columns]].copy()
        else:
            df = pd.DataFrame(np.asarray(X), columns=self.raw_features)
        for c in df.columns:
            if df[c].dtype == "object":
                df[c] = df[c].astype(str).where(df[c].notna(), "(null)")
            else:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)
        for c in self.log_cols:
            if c in df.columns:
                df[c] = np.log1p(df[c].astype(float).clip(lower=0))
        Xd = pd.get_dummies(df, drop_first=True, dtype=float)
        for c, (mu, sd) in self.scaler.items():
            if c in Xd.columns:
                Xd[c] = (Xd[c] - mu) / sd
        Xd = Xd.reindex(columns=self.feature_names, fill_value=0.0).fillna(0.0)
        return Xd.values
    def predict(self, X):
        log_y = self.result.predict(sm.add_constant(self._transform(X), has_constant="add"))
        return np.exp(log_y)

tags = {"feature_table": upt_table, "model_type": "GLM_Gamma_Motor", "story": "champion"}

with mlflow.start_run(run_name=f"sev_glm_motor_{run_name}", tags=tags) as run:
    mlflow.log_params({"family":"Gamma","link":"log","features":len(FEATURE_NAMES),
                       "train_rows":len(X_train),"test_rows":len(X_test)})
    # Fit OLS on log(y) — equivalent to Gamma GLM with log link for large samples
    log_y = np.log(y_train.values)
    res = sm.OLS(log_y, sm.add_constant(X_train.values, has_constant="add")).fit()
    y_pred = np.exp(res.predict(sm.add_constant(X_test.values, has_constant="add")))
    mae = float(mean_absolute_error(y_test, y_pred))
    mlflow.log_metrics({"mae": mae, "n_train": len(X_train)})
    print(f"MAE={mae:.0f}")

    wrapper = GammaGLMWrapper(res, FEATURE_NAMES, FEATURES, LOG_COLS, SCALER)

    from mlflow.models.signature import infer_signature
    sample_X    = _prep(training_pdf.head(5))
    sample_pred = wrapper.predict(sample_X)
    signature   = infer_signature(sample_X, sample_pred)

    # cloudpickle: avoid the sklearn flavor's skops "untrusted types" guard on
    # the statsmodels-wrapped GLM (see freq_glm_motor).
    fe.log_model(model=wrapper, artifact_path="model", flavor=mlflow.sklearn,
                 training_set=training_set,
                 registered_model_name=f"{fqn}.sev_glm_motor",
                 signature=signature, input_example=sample_X,
                 serialization_format="cloudpickle")
    print(f"UC model: {fqn}.sev_glm_motor")

from mlflow.tracking import MlflowClient
mc = MlflowClient()
latest = max(int(v.version) for v in mc.search_model_versions(f"name='{fqn}.sev_glm_motor'"))
mc.set_registered_model_alias(name=f"{fqn}.sev_glm_motor", alias="champion", version=str(latest))
print(f"alias 'champion' → v{latest}")

dbutils.notebook.exit(json.dumps({"model": f"{fqn}.sev_glm_motor", "version": latest, "mae": mae}))
