# Databricks notebook source
# MAGIC %md
# MAGIC # Motor Frequency GLM — production champion
# MAGIC
# MAGIC Poisson GLM on motor `claim_count_5y`. Features pulled via
# MAGIC `FeatureLookup` from `unified_motor_table_live`. Registered as
# MAGIC `{catalog}.{schema}.freq_glm_motor` with the `champion` alias.
# MAGIC
# MAGIC Signal mix: driver demographics + vehicle group + telematics
# MAGIC behaviour. The telematics signals (behaviour_score, recent
# MAGIC speeding/curfew events) carry most of the variance and drive the
# MAGIC live-serving demo's price-change story.

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
from sklearn.metrics import mean_squared_error
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

mlflow.set_registry_uri("databricks-uc")
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
mlflow.set_experiment(f"/Workspace/Users/{user}/pricing_workbench_production_motor_freq")

fe = FeatureEngineeringClient()

# COMMAND ----------

FEATURES = [
    "driver_age", "license_years_held", "no_claims_years",
    "gender", "marital_status", "occupation_class",
    "vehicle_group", "vehicle_value", "vehicle_age",
    "annual_mileage", "parking_overnight", "business_use", "fuel_type",
    "prior_convictions", "prior_accidents_5y",
    # Telematics
    "behaviour_score", "avg_speed_mph", "hours_driven_30d",
    "night_driving_pct",
    "recent_speeding_events", "recent_curfew_breaches", "recent_harsh_braking_30d",
    "telematics_recent_event_count",
]
TARGET = "claim_count_5y"
KEY    = "policy_id"

# COMMAND ----------

upt_table = f"{fqn}.unified_motor_table_live"
# Keep policy_id in the loaded pdf so the train/test mask aligns with the
# FE-resolved rows (FeatureLookup may drop unmatched rows; deriving the mask
# from labels_df separately gives a length mismatch).
labels_df = spark.table(upt_table).select(KEY, TARGET).sample(0.10, seed=42)

training_set = fe.create_training_set(
    df              = labels_df,
    feature_lookups = [FeatureLookup(table_name=upt_table, feature_names=FEATURES, lookup_key=KEY)],
    label           = TARGET,
    # NB: do NOT exclude_columns=[KEY] — we need it for mask alignment.
)
training_pdf = training_set.load_df().toPandas()
print(f"Training set: {len(training_pdf):,} rows × {len(training_pdf.columns)} cols")

# COMMAND ----------

def _prep_raw(df: pd.DataFrame) -> pd.DataFrame:
    out = df[FEATURES].copy()
    for c in out.columns:
        if out[c].dtype == "object":
            out[c] = out[c].astype(str).where(out[c].notna(), "(null)")
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).astype(float)
    return out

X = pd.get_dummies(_prep_raw(training_pdf), drop_first=True, dtype=float).fillna(0.0)
y = training_pdf[TARGET].fillna(0).astype(float)

# Mask derived from training_pdf itself so X / y / mask are always aligned.
mask = training_pdf[KEY].apply(lambda s: abs(hash(s)) % 100 < 80).values
X_train, y_train = X[mask], y[mask]
X_test,  y_test  = X[~mask], y[~mask]
print(f"Train: {len(X_train):,}   Test: {len(X_test):,}")

# COMMAND ----------

FEATURE_NAMES = list(X.columns)

class PoissonGLMWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, result, feature_names, raw_features):
        self.result = result; self.feature_names = feature_names; self.raw_features = raw_features
    def fit(self, X, y): return self
    def _transform(self, X):
        if hasattr(X, "columns"):
            df = pd.DataFrame(index=X.index if hasattr(X,"index") else None)
            for c in self.raw_features:
                df[c] = X[c] if c in X.columns else 0.0
        else:
            df = pd.DataFrame(np.asarray(X), columns=self.raw_features)
        for c in df.columns:
            if df[c].dtype == "object":
                df[c] = df[c].astype(str).where(df[c].notna(), "(null)")
            else:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)
        Xd = pd.get_dummies(df, drop_first=True, dtype=float)
        Xd = Xd.reindex(columns=self.feature_names, fill_value=0.0).fillna(0.0)
        return Xd.values
    def predict(self, X):
        return self.result.predict(sm.add_constant(self._transform(X), has_constant="add"))

tags = {"feature_table": upt_table, "model_type": "GLM_Poisson_Motor", "story": "champion"}

with mlflow.start_run(run_name=f"freq_glm_motor_{run_name}", tags=tags) as run:
    mlflow.log_params({"family":"Poisson","link":"log","features": len(FEATURE_NAMES),
                       "train_rows": len(X_train), "test_rows": len(X_test)})
    glm = sm.GLM(y_train.values, sm.add_constant(X_train.values, has_constant="add"),
                 family=sm.families.Poisson(link=sm.families.links.Log()))
    res = glm.fit(maxiter=50)
    y_pred = res.predict(sm.add_constant(X_test.values, has_constant="add"))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    order = np.argsort(-y_pred)
    cum_y = np.cumsum(y_test.values[order]) / (y_test.sum() + 1e-9)
    cum_n = np.arange(1, len(y_test) + 1) / len(y_test)
    gini = float(2 * np.trapz(cum_y, cum_n) - 1)
    mlflow.log_metrics({"rmse": rmse, "gini": gini, "aic": float(res.aic), "bic": float(res.bic)})
    print(f"Gini={gini:.4f}  RMSE={rmse:.4f}")

    wrapper = PoissonGLMWrapper(res, FEATURE_NAMES, FEATURES)

    from mlflow.models.signature import infer_signature
    sample_X    = _prep_raw(training_pdf.head(5))
    sample_pred = wrapper.predict(sample_X)
    signature   = infer_signature(sample_X, sample_pred)

    # serialization_format=cloudpickle: the statsmodels-wrapped GLM trips the
    # sklearn flavor's newer skops "untrusted types" guard on load; cloudpickle
    # avoids skops entirely (durable across mlflow version drift).
    fe.log_model(model=wrapper, artifact_path="model", flavor=mlflow.sklearn,
                 training_set=training_set,
                 registered_model_name=f"{fqn}.freq_glm_motor",
                 signature=signature, input_example=sample_X,
                 serialization_format="cloudpickle")
    print(f"UC model: {fqn}.freq_glm_motor")

# Promote latest version to champion alias
from mlflow.tracking import MlflowClient
mc = MlflowClient()
latest = max(int(v.version) for v in mc.search_model_versions(f"name='{fqn}.freq_glm_motor'"))
mc.set_registered_model_alias(name=f"{fqn}.freq_glm_motor", alias="champion", version=str(latest))
print(f"alias 'champion' → v{latest}")

import json
dbutils.notebook.exit(json.dumps({"model": f"{fqn}.freq_glm_motor", "version": latest, "gini": gini, "rmse": rmse}))
