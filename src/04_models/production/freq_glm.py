# Databricks notebook source
# MAGIC %md
# MAGIC # Frequency GLM — production champion
# MAGIC
# MAGIC Poisson GLM on `claim_count_5y` using a hand-picked rating-factor subset.
# MAGIC Features come from the Modelling Mart (already an FE table with
# MAGIC `policy_id` as PK), resolved via `FeatureLookup` so feature lineage is
# MAGIC captured at training time. Registered in UC as
# MAGIC `{catalog}.{schema}.freq_glm` with the version tagged `champion`.

# COMMAND ----------

dbutils.widgets.text("catalog_name", "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name",  "pricing_upt")
dbutils.widgets.text("run_name",     "champion")
dbutils.widgets.text("simulation_date", "")  # empty = current champion

catalog = dbutils.widgets.get("catalog_name")
schema  = dbutils.widgets.get("schema_name")
fqn     = f"{catalog}.{schema}"
run_name= dbutils.widgets.get("run_name")
sim_date= dbutils.widgets.get("simulation_date") or None

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Re-read widgets after restartPython
catalog = dbutils.widgets.get("catalog_name")
schema  = dbutils.widgets.get("schema_name")
fqn     = f"{catalog}.{schema}"
run_name= dbutils.widgets.get("run_name")
sim_date= dbutils.widgets.get("simulation_date") or None

import json, sys
from datetime import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm
import mlflow
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

mlflow.set_registry_uri("databricks-uc")
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
mlflow.set_experiment(f"/Workspace/Users/{user}/pricing_workbench_production_freq")

fe = FeatureEngineeringClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature selection — 22 rating factors

# COMMAND ----------

FEATURES = [
    "sum_insured", "annual_turnover", "current_premium",
    "industry_risk_tier", "construction_type",
    "credit_score", "ccj_count", "years_trading",
    "flood_zone_rating", "proximity_to_fire_station_km",
    "crime_theft_index", "subsidence_risk", "composite_location_risk",
    "urban_score", "is_coastal", "population_density_per_km2",
    "elevation_metres", "annual_rainfall_mm",
    "director_stability_score", "employee_count_est",
    "distance_to_coast_km", "neighbourhood_claim_frequency",
]
TARGET = "claim_count_5y"
KEY    = "policy_id"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build training set via FeatureLookup

# COMMAND ----------

# Label DataFrame = just (policy_id, claim_count_5y). FeatureLookup resolves
# the 22 features from the FE table unified_pricing_table_live.
labels_df = spark.table(f"{fqn}.unified_pricing_table_live").select(KEY, TARGET)

feature_lookups = [
    FeatureLookup(
        table_name     = f"{fqn}.unified_pricing_table_live",
        feature_names  = FEATURES,
        lookup_key     = KEY,
    )
]
training_set = fe.create_training_set(
    df              = labels_df,
    feature_lookups = feature_lookups,
    label           = TARGET,
    exclude_columns = [KEY],
)
training_pdf = training_set.load_df().toPandas()
print(f"Training set: {len(training_pdf):,} rows × {len(training_pdf.columns)} cols")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Encode categoricals + train/test split (80/20 by policy_id hash)

# COMMAND ----------

def _prep_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitise the raw feature frame so MLflow schema enforcement doesn't
    reject NaN-bearing int columns at serving time."""
    out = df[FEATURES].copy()
    for c in out.columns:
        if out[c].dtype == "object":
            out[c] = out[c].astype(str).where(out[c].notna(), "(null)")
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).astype(float)
    return out

X = pd.get_dummies(_prep_raw(training_pdf), drop_first=True, dtype=float).fillna(0.0)
y = training_pdf[TARGET].fillna(0).astype(float)

# Deterministic split so re-runs produce the same test set
hashes = pd.Series(labels_df.toPandas()[KEY].apply(lambda s: abs(hash(s)) % 100).values)
train_mask = hashes < 80
X_train, y_train = X[train_mask.values], y[train_mask.values]
X_test,  y_test  = X[~train_mask.values], y[~train_mask.values]
print(f"Train: {len(X_train):,}   Test: {len(X_test):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fit + metrics + log + register

# COMMAND ----------

FEATURE_NAMES = list(X.columns)   # one-hot-encoded, what the fitted GLM expects

class PoissonGLMWrapper(BaseEstimator, RegressorMixin):
    """Self-encoding wrapper: accepts a raw-features DataFrame (same columns as
    FEATURES) and replays the exact training pipeline — _prep_raw + get_dummies
    + reindex to the trained feature names — at predict time. This means
    Compare & Test, batch jobs and serving can all call pyfunc.predict() on a
    plain Modelling-Mart row set without any pre-encoding."""
    def __init__(self, result, feature_names, raw_features):
        self.result         = result
        self.feature_names  = feature_names
        self.raw_features   = raw_features

    def fit(self, X, y): return self

    def _transform(self, X):
        if hasattr(X, "columns"):
            df = pd.DataFrame(index=X.index if hasattr(X, "index") else None)
            for c in self.raw_features:
                df[c] = X[c] if c in X.columns else 0.0
        else:
            df = pd.DataFrame(np.asarray(X), columns=self.raw_features)
        # Match _prep_raw: object → string-or-"(null)"; numeric → float + NaN→0.
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

tags = {
    "feature_table":  f"{fqn}.unified_pricing_table_live",
    "model_type":     "GLM_Poisson",
    "simulated":      "false",
    "story":          "champion",
}
if sim_date:
    tags["simulation_date"] = sim_date
    tags["simulated"]       = "true"

with mlflow.start_run(run_name=f"freq_glm_{run_name}", tags=tags) as run:
    mlflow.log_params({
        "family": "Poisson", "link": "log",
        "features": len(FEATURE_NAMES),
        "train_rows": len(X_train), "test_rows": len(X_test),
    })

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
    print(f"Gini={gini:.4f}  RMSE={rmse:.4f}  AIC={res.aic:.0f}")

    # Relativities artifact — exp(coef) per feature
    rel = pd.DataFrame({
        "feature":    ["intercept"] + FEATURE_NAMES,
        "coefficient": res.params.tolist(),
        "relativity": np.exp(res.params).tolist(),
        "p_value":    res.pvalues.tolist(),
    })
    rel.to_csv("/tmp/relativities.csv", index=False)
    mlflow.log_artifact("/tmp/relativities.csv")

    wrapper = PoissonGLMWrapper(res, FEATURE_NAMES, FEATURES)

    # Pin an explicit signature on raw features so pyfunc.predict() accepts a
    # plain Modelling-Mart row set at inference time. Sanitising the sample
    # first keeps NaN-bearing int columns from being inferred as "integer"
    # slots (which would then fail enforcement at serving).
    from mlflow.models.signature import infer_signature
    sample_X    = _prep_raw(training_pdf.head(5))
    sample_pred = wrapper.predict(sample_X)
    signature   = infer_signature(sample_X, sample_pred)

    fe.log_model(
        model                 = wrapper,
        artifact_path         = "model",
        flavor                = mlflow.sklearn,
        training_set          = training_set,
        registered_model_name = f"{fqn}.freq_glm",
        signature             = signature,
        input_example         = sample_X,
        # cloudpickle: avoid the sklearn flavor's skops "untrusted types" guard
        # on the statsmodels-wrapped GLM (durable across mlflow version drift).
        serialization_format  = "cloudpickle",
    )

    print(f"run_id={run.info.run_id}")
    print(f"UC model: {fqn}.freq_glm")

# Audit — don't import _shared here because notebook pip restart breaks relative imports.
# Inline the insert.
try:
    det = json.dumps({"gini": gini, "rmse": rmse, "aic": float(res.aic),
                      "mlflow_run_id": run.info.run_id,
                      "simulated": bool(sim_date), "simulation_date": sim_date, "story": run_name}).replace("'", "''")
    spark.sql(f"""
        INSERT INTO {fqn}.audit_log
          (event_id, event_type, entity_type, entity_id, entity_version, user_id, timestamp, details, source)
        SELECT uuid(), 'model_trained', 'model', 'freq_glm', '{run_name}', '{user}',
               current_timestamp(), '{det}', 'notebook'
    """)
except Exception as e:
    print(f"audit: {e}")
