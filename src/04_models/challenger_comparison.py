# Databricks notebook source
# MAGIC %md
# MAGIC # Challenger Comparison — Baseline vs +Urban Score vs +Both Factors
# MAGIC
# MAGIC Answers the business question: **does adding the derived factors actually
# MAGIC make the model better?** Trains three Poisson GLMs for claim frequency, each
# MAGIC an ablation of the derived-factors set, and reports Gini on a held-out
# MAGIC sample plus per-factor lift attribution.
# MAGIC
# MAGIC **Ablation ladder:**
# MAGIC | # | Feature set                                              | Cohort tag       |
# MAGIC |---|----------------------------------------------------------|------------------|
# MAGIC | 1 | Baseline features only                                   | `baseline`       |
# MAGIC | 2 | Baseline + `urban_score`                                 | `plus_urban`     |
# MAGIC | 3 | Baseline + `urban_score` + `neighbourhood_claim_frequency` | `plus_both`      |
# MAGIC
# MAGIC **Attribution:**
# MAGIC - urban_score lift        = Gini(plus_urban) - Gini(baseline)
# MAGIC - claim_frequency lift    = Gini(plus_both)  - Gini(plus_urban)
# MAGIC - Total lift              = Gini(plus_both)  - Gini(baseline)
# MAGIC
# MAGIC **Outputs:**
# MAGIC - Three MLflow runs (one per cohort) with tag `challenger_cohort`
# MAGIC - Summary table `challenger_comparison_latest` consumed by the app

# COMMAND ----------

dbutils.widgets.text("catalog_name", "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name", "pricing_upt")

catalog = dbutils.widgets.get("catalog_name")
schema = dbutils.widgets.get("schema_name")
fqn = f"{catalog}.{schema}"

# COMMAND ----------

import mlflow
import numpy as np
import pandas as pd
import statsmodels.api as sm
from datetime import datetime, timezone
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, TimestampType

try:
    user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
    mlflow.set_experiment(f"/Workspace/Users/{user}/pricing_upt_challenger_comparison")
except Exception:
    pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load UPT and prepare training frame

# COMMAND ----------

upt_table_name = f"{fqn}.unified_pricing_table_live"
upt = spark.table(upt_table_name)

# Delta version for reproducibility
upt_history = spark.sql(f"DESCRIBE HISTORY {upt_table_name} LIMIT 1").collect()
upt_delta_version = upt_history[0]["version"] if upt_history else None

# Baseline feature set — same shape as model_01_glm_frequency.py, MINUS the derived factors
BASELINE_FEATURES = [
    "annual_turnover", "sum_insured", "building_age_years", "current_premium",
    "flood_zone_rating", "proximity_to_fire_station_km", "crime_theft_index", "subsidence_risk",
    "composite_location_risk", "credit_score", "ccj_count", "years_trading",
    "business_stability_score", "market_median_rate",
    "credit_default_probability", "director_stability_score",
    "employee_count_est", "distance_to_coast_km", "population_density_per_km2",
    "elevation_metres", "annual_rainfall_mm",
]
FACTOR_URBAN = "urban_score"
FACTOR_CLAIM = "neighbourhood_claim_frequency"

all_cols = ["policy_id", "claim_count_5y"] + BASELINE_FEATURES + [FACTOR_URBAN, FACTOR_CLAIM]

# Check the derived factors exist in the UPT; fail loudly if not
upt_cols = set(upt.columns)
missing = [c for c in [FACTOR_URBAN, FACTOR_CLAIM] if c not in upt_cols]
if missing:
    raise ValueError(
        f"Derived factors {missing} not in UPT. "
        f"Run 03_gold/derive_factors.py then 03_gold/build_upt.py before this notebook."
    )

pdf = upt.select(*all_cols).toPandas()
pdf["claim_frequency"] = pdf["claim_count_5y"].fillna(0).astype(float)
for c in BASELINE_FEATURES + [FACTOR_URBAN, FACTOR_CLAIM]:
    pdf[c] = pd.to_numeric(pdf[c], errors="coerce").fillna(0)

# Deterministic 80/20 split by policy_id hash
pdf["split_hash"] = pdf["policy_id"].apply(lambda x: abs(hash(x)) % 100)
train_pdf = pdf[pdf["split_hash"] < 80].copy()
test_pdf  = pdf[pdf["split_hash"] >= 80].copy()
y_train = train_pdf["claim_frequency"].values
y_test  = test_pdf["claim_frequency"].values

print(f"Train: {len(train_pdf)}, Test: {len(test_pdf)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gini helper

# COMMAND ----------

def gini(y_true, y_pred):
    """Normalized Gini coefficient — insurance-standard uplift metric.
    Ranges from ~0 (no rank-order signal) to ~1 (perfect rank-order)."""
    arr = np.asarray(list(zip(y_true, y_pred, np.arange(len(y_true)))), dtype=float)
    # Sort by prediction desc; tie-break on original index
    arr = arr[np.lexsort((arr[:, 2], -arr[:, 1]))]
    total_pos = arr[:, 0].sum()
    if total_pos == 0:
        return 0.0
    cum_pos = np.cumsum(arr[:, 0]) / total_pos
    # Lorenz area vs diagonal
    giniSum = cum_pos.sum() - (len(arr) + 1) / 2.0
    raw = giniSum / len(arr)
    # Normalize by perfect model
    arr_perf = np.sort(y_true)[::-1]
    cum_perf = np.cumsum(arr_perf) / total_pos
    perfSum = cum_perf.sum() - (len(arr_perf) + 1) / 2.0
    perf = perfSum / len(arr_perf)
    if perf == 0:
        return 0.0
    return float(raw / perf)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fit three models and collect results

# COMMAND ----------

COHORTS = [
    ("baseline",   BASELINE_FEATURES),
    ("plus_urban", BASELINE_FEATURES + [FACTOR_URBAN]),
    ("plus_both",  BASELINE_FEATURES + [FACTOR_URBAN, FACTOR_CLAIM]),
]

results = []
run_ids = {}

for cohort_name, features in COHORTS:
    X_train = sm.add_constant(train_pdf[features].values)
    X_test  = sm.add_constant(test_pdf[features].values)

    with mlflow.start_run(run_name=f"challenger_{cohort_name}") as run:
        mlflow.set_tag("challenger_cohort",    cohort_name)
        mlflow.set_tag("challenger_comparison", "true")
        mlflow.set_tag("feature_table",         upt_table_name)
        mlflow.log_param("model_type",        "GLM_Poisson")
        mlflow.log_param("n_features",        len(features))
        mlflow.log_param("cohort",            cohort_name)
        mlflow.log_param("upt_delta_version", upt_delta_version)
        mlflow.log_param("train_rows",        len(train_pdf))
        mlflow.log_param("test_rows",         len(test_pdf))
        mlflow.log_param("features",          ",".join(features))

        glm = sm.GLM(y_train, X_train, family=sm.families.Poisson(link=sm.families.links.Log()))
        fit = glm.fit()
        y_pred = fit.predict(X_test)

        g = gini(y_test, y_pred)
        rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))

        mlflow.log_metric("gini",           g)
        mlflow.log_metric("rmse",           rmse)
        mlflow.log_metric("aic",            float(fit.aic))

        run_ids[cohort_name] = run.info.run_id
        results.append({"cohort": cohort_name, "gini": g, "rmse": rmse, "aic": float(fit.aic), "run_id": run.info.run_id})
        print(f"  {cohort_name:12s}  features={len(features):3d}  Gini={g:.4f}  RMSE={rmse:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Attribution

# COMMAND ----------

by_cohort = {r["cohort"]: r for r in results}
gini_baseline   = by_cohort["baseline"]["gini"]
gini_plus_urban = by_cohort["plus_urban"]["gini"]
gini_plus_both  = by_cohort["plus_both"]["gini"]

lift_urban        = gini_plus_urban - gini_baseline
lift_claim_freq   = gini_plus_both  - gini_plus_urban
lift_total        = gini_plus_both  - gini_baseline
# Relative lift vs baseline (handles tiny baselines without divide-by-zero)
lift_total_pct    = (lift_total / gini_baseline * 100.0) if gini_baseline > 0 else 0.0

print(f"""
Baseline Gini:         {gini_baseline:.4f}
+ urban_score:         {gini_plus_urban:.4f}  (+{lift_urban:+.4f})
+ neigh_claim_freq:    {gini_plus_both:.4f}   (+{lift_claim_freq:+.4f})
Total lift:            {lift_total:+.4f}  ({lift_total_pct:+.2f}% vs baseline)
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write summary table consumed by the app

# COMMAND ----------

summary_schema = StructType([
    StructField("cohort",               StringType()),
    StructField("n_features",           IntegerType()),
    StructField("gini",                 DoubleType()),
    StructField("rmse",                 DoubleType()),
    StructField("aic",                  DoubleType()),
    StructField("lift_vs_baseline",     DoubleType()),
    StructField("lift_vs_prev",         DoubleType()),
    StructField("attribution_factor",   StringType()),
    StructField("run_id",               StringType()),
    StructField("upt_delta_version",    StringType()),
    StructField("computed_at",          TimestampType()),
])

now = datetime.now(timezone.utc).replace(tzinfo=None)
rows = [
    ("baseline",   len(BASELINE_FEATURES),                         gini_baseline,   by_cohort["baseline"]["rmse"],   by_cohort["baseline"]["aic"],   0.0,            0.0,            None,                    run_ids["baseline"],   str(upt_delta_version), now),
    ("plus_urban", len(BASELINE_FEATURES) + 1,                     gini_plus_urban, by_cohort["plus_urban"]["rmse"], by_cohort["plus_urban"]["aic"], lift_urban,     lift_urban,     FACTOR_URBAN,            run_ids["plus_urban"], str(upt_delta_version), now),
    ("plus_both",  len(BASELINE_FEATURES) + 2,                     gini_plus_both,  by_cohort["plus_both"]["rmse"],  by_cohort["plus_both"]["aic"],  lift_total,     lift_claim_freq, FACTOR_CLAIM,           run_ids["plus_both"],  str(upt_delta_version), now),
]
summary_df = spark.createDataFrame(rows, schema=summary_schema)

summary_table = f"{fqn}.challenger_comparison_latest"
summary_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(summary_table)

spark.sql(f"""
    ALTER TABLE {summary_table}
    SET TBLPROPERTIES (
        'comment' = 'Latest challenger vs baseline comparison — consumed by the Pricing Workbench app.'
    )
""")
print(f"✓ {summary_table} — {summary_df.count()} rows")
display(summary_df)
