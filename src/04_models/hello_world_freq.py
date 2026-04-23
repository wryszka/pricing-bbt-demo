# Databricks notebook source
# MAGIC %md
# MAGIC # Hello, Pricing Workbench — A Minimal Frequency GLM
# MAGIC
# MAGIC **Read this first if you've never trained a pricing model on Databricks.**
# MAGIC
# MAGIC This notebook is the shortest possible end-to-end example: load the Modelling Mart,
# MAGIC fit a Poisson claim-frequency GLM, log it to MLflow, register it in Unity Catalog.
# MAGIC Every step has a comment explaining what it does and why.
# MAGIC
# MAGIC **What you'll do:**
# MAGIC 1. Read the training feature table (the Modelling Mart).
# MAGIC 2. Split into train / test.
# MAGIC 3. Fit a Poisson GLM with a handful of rating factors.
# MAGIC 4. Evaluate test-set metrics (Gini, deviance).
# MAGIC 5. Log the trained model to MLflow and register it in UC.
# MAGIC
# MAGIC **What this notebook deliberately skips** (see `model_01_glm_frequency.py` for the
# MAGIC production-grade version): relativity tables, coefficient review,
# MAGIC confidence intervals, `FeatureLookup` serving-time binding, extensive feature set.

# COMMAND ----------

# -------- Widgets --------------------------------------------------------
# Widgets turn a notebook into something parameterisable — the same code runs
# against different catalogs / schemas just by changing the widget values.
dbutils.widgets.text("catalog_name", "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name",  "pricing_upt")

catalog = dbutils.widgets.get("catalog_name")
schema  = dbutils.widgets.get("schema_name")
fqn     = f"{catalog}.{schema}"

# COMMAND ----------

# -------- Imports --------------------------------------------------------
# statsmodels = classical regression library actuaries already know (GLMs, GAMs,
# hypothesis tests). MLflow tracks every model run; the sklearn wrapper lets us
# save a statsmodels model in MLflow's standard format.
import numpy as np
import pandas as pd
import statsmodels.api as sm
import mlflow
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error

# Tell MLflow to register models in Unity Catalog (not the legacy workspace registry).
# This makes the model governed, versioned, and discoverable like any other UC asset.
mlflow.set_registry_uri("databricks-uc")

# Every notebook should have its own experiment path so runs don't mix.
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
mlflow.set_experiment(f"/Workspace/Users/{user}/pricing_workbench_hello_world")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Read the Modelling Mart
# MAGIC
# MAGIC The Modelling Mart is a single Delta table — one row per policy — with every
# MAGIC approved rating factor pre-joined. You just read it like any Spark table, pull
# MAGIC the columns you need into pandas, and you're ready to fit.

# COMMAND ----------

mart_fqn = f"{fqn}.unified_pricing_table_live"

# Keep it simple — five obvious rating factors + the claim-count target.
feature_cols = [
    "flood_zone_rating",
    "crime_theft_index",
    "sum_insured",
    "years_trading",
    "credit_score",
]
target_col = "claim_count_5y"

# Pull only what we need. .toPandas() runs the SQL and brings the result into
# memory — fine for a 50K-row table, not fine for billions.
pdf = (
    spark.table(mart_fqn)
         .select("policy_id", target_col, *feature_cols)
         .toPandas()
)
pdf = pdf.dropna()  # drop rows with any null — a real model would impute instead

print(f"Loaded {len(pdf):,} rows from {mart_fqn}")
print(f"Claim count distribution: mean {pdf[target_col].mean():.2f}, max {pdf[target_col].max()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Train / test split
# MAGIC
# MAGIC Hash the `policy_id` to a bucket so the split is deterministic — the same
# MAGIC policy always lands in the same bucket, which lets us cleanly re-run without
# MAGIC shuffling the evaluation set.

# COMMAND ----------

pdf["bucket"] = pdf["policy_id"].apply(lambda s: hash(s) % 10)
train = pdf[pdf["bucket"] < 8]  # 80% train
test  = pdf[pdf["bucket"] >= 8] # 20% test

X_train, y_train = sm.add_constant(train[feature_cols]), train[target_col].astype(float)
X_test,  y_test  = sm.add_constant(test[feature_cols]),  test[target_col].astype(float)
print(f"Train: {len(X_train):,}   Test: {len(X_test):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Fit a Poisson GLM
# MAGIC
# MAGIC Poisson distribution is the textbook choice for claim counts: non-negative
# MAGIC integers, variance = mean. The log link means each coefficient reads as a
# MAGIC multiplicative rating relativity — the form actuaries already know.

# COMMAND ----------

with mlflow.start_run(run_name="hello_world_freq_glm") as run:
    # Every parameter that matters for reproducibility goes into MLflow params.
    mlflow.log_params({
        "model_type":   "GLM_Poisson",
        "features":     ",".join(feature_cols),
        "train_rows":   len(train),
        "test_rows":    len(test),
        "source_table": mart_fqn,
    })

    glm    = sm.GLM(y_train, X_train, family=sm.families.Poisson(link=sm.families.links.Log()))
    result = glm.fit()

    y_pred = result.predict(X_test)

    # RMSE + Gini — the two numbers you'd quote to an actuarial manager.
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    order = np.argsort(-y_pred)
    cum_y = np.cumsum(y_test.values[order]) / y_test.sum()
    cum_n = np.arange(1, len(y_test) + 1) / len(y_test)
    gini = float(2 * np.trapz(cum_y, cum_n) - 1)

    mlflow.log_metrics({"rmse": rmse, "gini": gini, "aic": float(result.aic)})
    print(f"Test RMSE: {rmse:.4f}   Test Gini: {gini:.4f}   AIC: {result.aic:.0f}")

    # COMMAND ----------
    # MAGIC %md
    # MAGIC ## 4. Log + register the model
    # MAGIC
    # MAGIC Wrap the statsmodels result in a sklearn-shaped class so MLflow can save
    # MAGIC it in its standard sklearn flavour. Then `mlflow.sklearn.log_model(...,
    # MAGIC registered_model_name=...)` both logs it to the run and creates a UC model.

    class PoissonGLMWrapper(BaseEstimator, RegressorMixin):
        """Minimal sklearn-compatible wrapper around a statsmodels GLM result."""
        def __init__(self, result, feature_names):
            self.result, self.feature_names = result, feature_names
        def predict(self, X):
            return self.result.predict(sm.add_constant(X[self.feature_names], has_constant="add"))
        def fit(self, X, y):
            return self

    wrapper = PoissonGLMWrapper(result, feature_cols)

    # Signature tells MLflow the expected input/output schema. Good hygiene — makes
    # the served model self-describing.
    signature = mlflow.models.infer_signature(
        train[feature_cols].head(5), wrapper.predict(train[feature_cols].head(5))
    )

    mlflow.sklearn.log_model(
        sk_model               = wrapper,
        artifact_path          = "model",
        signature              = signature,
        registered_model_name  = f"{catalog}.{schema}.hello_world_freq_glm",
        input_example          = train[feature_cols].head(5),
    )

    print(f"\n✅ Logged MLflow run: {run.info.run_id}")
    print(f"✅ Registered UC model: {catalog}.{schema}.hello_world_freq_glm")

# COMMAND ----------

# MAGIC %md
# MAGIC ## That's it.
# MAGIC
# MAGIC Open the MLflow run from the experiment above — you'll see params, metrics,
# MAGIC coefficients, and the registered model version. From here the "full" frequency
# MAGIC notebook (`model_01_glm_frequency.py`) adds:
# MAGIC
# MAGIC - Relativity tables per factor (what a regulator actually reviews)
# MAGIC - `FeatureLookup` binding so the model fetches its feature vector from the
# MAGIC   online store at serving time (no client-side plumbing)
# MAGIC - Double-lift charts vs a baseline, p-value review, coverage audits
# MAGIC
# MAGIC Go fit a severity model next: `model_02_glm_severity.py`.
