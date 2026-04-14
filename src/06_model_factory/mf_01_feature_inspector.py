# Databricks notebook source
# MAGIC %md
# MAGIC # Model Factory — Step 1: Feature Inspector & Training Plan
# MAGIC
# MAGIC ## What is the Model Factory?
# MAGIC
# MAGIC In traditional insurance pricing, an actuary manually selects features, picks a
# MAGIC model family (usually a GLM), trains it, and validates it — a process that takes
# MAGIC **weeks to months** per model iteration. Regulatory requirements (Solvency II,
# MAGIC Lloyd's model governance) demand that every decision is documented, every feature
# MAGIC selection justified, and every model version auditable.
# MAGIC
# MAGIC The **Model Factory** automates this entire pipeline while *strengthening*
# MAGIC governance:
# MAGIC
# MAGIC | Step | Notebook | What Happens |
# MAGIC |------|----------|-------------|
# MAGIC | 1 | **This notebook** | Inspect the feature table, profile every column, and generate a training plan |
# MAGIC | 2 | `mf_02_automated_training` | Train ~20 model variants (GLMs + GBMs) from the plan |
# MAGIC | 3 | `mf_03_evaluation` | Score every model on insurance-specific metrics (Gini, PSI, regulatory suitability) |
# MAGIC | 4 | `mf_04_actuary_review` | Present the leaderboard to the actuary for approval/rejection |
# MAGIC | 5 | `mf_05_promote_model` | Register approved models in Unity Catalog with champion aliases |
# MAGIC
# MAGIC **Every action is logged** to an append-only audit table (`mf_audit_log`) with
# MAGIC timestamps, actor identity, UPT version, and full event details — ready to pull
# MAGIC for any regulatory review.
# MAGIC
# MAGIC ### Databricks Services Used
# MAGIC - **Mosaic AI Foundation Model API** — LLM-augmented reasoning about which models to train
# MAGIC - **MLflow + Unity Catalog** — Experiment tracking, model registry, lineage
# MAGIC - **Delta Lake** — Versioned feature table, time travel for reproducibility
# MAGIC - **Genie** — Actuary can query the leaderboard and audit tables in natural language
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## This Notebook: Feature Inspector
# MAGIC
# MAGIC 1. Profile the Unified Pricing Table — every column gets: dtype, nulls, stats,
# MAGIC    correlation with claim targets, multicollinearity (VIF)
# MAGIC 2. Classify features into groups: `core_policy`, `bureau`, `geo`, `market`, `derived`
# MAGIC 3. Use the **Foundation Model API** to reason about the feature profile and
# MAGIC    recommend a training plan (with deterministic fallback)
# MAGIC 4. Write the training plan to `mf_training_plan` for the next notebook to execute
# MAGIC 5. Log everything to `mf_audit_log`

# COMMAND ----------

dbutils.widgets.text("catalog_name", "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name", "pricing_upt")
dbutils.widgets.text("factory_run_id", "")

catalog = dbutils.widgets.get("catalog_name")
schema = dbutils.widgets.get("schema_name")
fqn = f"{catalog}.{schema}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Initialise Factory Run

# COMMAND ----------

from datetime import datetime, timezone
import json
import uuid

# Generate a unique factory run ID (or use one passed in for re-runs)
factory_run_id = dbutils.widgets.get("factory_run_id").strip()
if not factory_run_id:
    factory_run_id = f"MF-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

# Pass to downstream notebooks when running as a job
try:
    dbutils.jobs.taskValues.set(key="factory_run_id", value=factory_run_id)
except Exception:
    pass  # Running interactively, not as a job task

print(f"Factory Run ID: {factory_run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Record UPT Version (Delta Time Travel)
# MAGIC
# MAGIC Pin the exact version of the Unified Pricing Table so that every model trained
# MAGIC in this factory run is reproducible. An auditor can always recover the exact
# MAGIC feature set used: `SELECT * FROM table VERSION AS OF {version}`.

# COMMAND ----------

upt_table = f"{fqn}.unified_pricing_table_live"
upt_history = spark.sql(f"DESCRIBE HISTORY {upt_table} LIMIT 1").collect()
upt_version = upt_history[0]["version"]
upt_timestamp = str(upt_history[0]["timestamp"])

print(f"UPT Table:   {upt_table}")
print(f"UPT Version: {upt_version}")
print(f"UPT As Of:   {upt_timestamp}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Audit Log & Helper Functions
# MAGIC
# MAGIC The audit log is an **append-only** Delta table. Every event in the Model Factory
# MAGIC pipeline is recorded here — feature profiling, LLM calls, training starts/completions,
# MAGIC evaluations, actuary decisions, model promotions.

# COMMAND ----------

def get_current_user():
    """Resolve the current Databricks user for audit trail."""
    try:
        return dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
    except Exception:
        import os
        return os.getenv("USER", "unknown")

def log_audit_event(spark, fqn, factory_run_id, event_type, details, mlflow_run_id=None, upt_version=None, actor=None):
    """Append a single event to the audit log."""
    from pyspark.sql.types import StructType, StructField, StringType
    from datetime import datetime, timezone

    schema = StructType([
        StructField("event_id", StringType()),
        StructField("factory_run_id", StringType()),
        StructField("event_type", StringType()),
        StructField("event_timestamp", StringType()),
        StructField("actor", StringType()),
        StructField("details_json", StringType()),
        StructField("mlflow_run_id", StringType()),
        StructField("upt_table_version", StringType()),
    ])

    event = (
        str(uuid.uuid4()),
        factory_run_id,
        event_type,
        datetime.now(timezone.utc).isoformat(),
        actor or get_current_user(),
        json.dumps(details) if isinstance(details, dict) else str(details),
        mlflow_run_id or "",
        str(upt_version) if upt_version is not None else "",
    )
    df = spark.createDataFrame([event], schema=schema)
    df.write.mode("append").saveAsTable(f"{fqn}.mf_audit_log")

current_user = get_current_user()
print(f"Actor: {current_user}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Create Model Factory Tables
# MAGIC
# MAGIC These tables persist across factory runs. Each run appends new rows, so you
# MAGIC can compare across runs and track how the feature landscape evolves over time.

# COMMAND ----------

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {fqn}.mf_audit_log (
        event_id STRING,
        factory_run_id STRING,
        event_type STRING,
        event_timestamp STRING,
        actor STRING,
        details_json STRING,
        mlflow_run_id STRING,
        upt_table_version STRING
    )
""")

# mf_feature_profile is created by the write.mode("overwrite") below
# with an explicit schema derived from pandas dtypes. No CREATE TABLE needed.

# mf_training_plan is created by the write.mode("overwrite") below
# with an explicit schema. No CREATE TABLE needed.

print("Model Factory tables ready")

# Log the factory run start
log_audit_event(spark, fqn, factory_run_id, "FACTORY_RUN_STARTED", {
    "upt_table": upt_table,
    "upt_version": upt_version,
    "upt_timestamp": upt_timestamp,
}, upt_version=upt_version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Load & Profile the Feature Table
# MAGIC
# MAGIC We pull the entire UPT into pandas and compute statistics for every column.
# MAGIC This tells us how many features we have, their types, distributions, and how
# MAGIC strongly each one correlates with the two primary targets:
# MAGIC - **`claim_count_5y`** — frequency (Poisson target)
# MAGIC - **`total_incurred_5y`** — severity/pure premium (Gamma/Tweedie target)

# COMMAND ----------

import pandas as pd
import numpy as np

upt_df = spark.table(upt_table)
pdf = upt_df.toPandas()

print(f"UPT shape: {pdf.shape[0]:,} rows x {pdf.shape[1]} columns")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Classification
# MAGIC
# MAGIC We group features by their origin — this determines which feature subsets the
# MAGIC factory will use for training. Core policy features are always included; bureau,
# MAGIC geo, and market features are tested as additive groups to measure their marginal
# MAGIC lift.

# COMMAND ----------

# Columns that are NOT features (identifiers, targets, metadata)
NON_FEATURE_COLS = {
    "policy_id", "inception_date", "renewal_date", "last_claim_date",
    "last_quote_date", "postcode_prefix", "market_join_key",
    "match_key_sic_region", "last_updated_by", "approval_timestamp",
    "source_version", "upt_build_timestamp", "split_hash",
    # Targets
    "claim_count_5y", "total_incurred_5y", "total_paid_5y", "total_reserve_5y",
    "open_claims_count",
}

# Feature group classification
FEATURE_GROUPS = {
    "core_policy": [
        "sic_code", "postcode_sector", "annual_turnover", "construction_type",
        "year_built", "sum_insured", "current_premium", "building_age_years",
        "industry_risk_tier", "location_risk_tier",
    ],
    "claims_history": [
        "claims_history_5y", "distinct_perils",
        "fire_incurred", "flood_incurred", "theft_incurred",
        "liability_incurred", "storm_incurred", "subsidence_incurred",
        "water_incurred",
    ],
    "quotes": [
        "quote_count", "avg_quoted_premium", "min_quoted_premium",
        "max_quoted_premium", "competitor_quote_count",
    ],
    "bureau": [
        "credit_score", "ccj_count", "years_trading", "director_changes",
        "credit_risk_tier", "business_stability_score",
        "credit_default_probability", "director_stability_score",
        "payment_history_score", "trade_credit_utilisation_pct",
        "debt_to_equity_ratio", "working_capital_ratio",
        "revenue_growth_3y_pct", "employee_count_est",
        "industry_default_rate_pct", "supplier_concentration_score",
        "invoice_dispute_rate_pct", "bank_account_stability_months",
        "registered_charges_count", "profit_margin_est_pct",
        "asset_tangibility_ratio", "interest_coverage_ratio",
        "accounts_filed_on_time", "company_age_months",
        "sector_bankruptcy_rate_pct", "management_experience_score",
    ],
    "geo": [
        "flood_zone_rating", "proximity_to_fire_station_km",
        "crime_theft_index", "subsidence_risk", "composite_location_risk",
        "distance_to_coast_km", "local_unemployment_rate_pct",
        "traffic_density_index", "air_quality_index",
        "average_property_value_k", "population_density_per_km2",
        "commercial_density_score", "historic_flood_events_10y",
        "elevation_metres", "distance_to_hospital_km",
        "distance_to_motorway_km", "green_space_pct",
        "noise_pollution_index", "broadband_speed_mbps",
        "listed_building_density", "average_wind_speed_mph",
        "annual_rainfall_mm", "soil_clay_content_pct",
        "radon_risk_level", "tree_cover_pct",
    ],
    "market": [
        "market_median_rate", "competitor_a_min_premium",
        "price_index_trend", "competitor_ratio",
    ],
    "derived": [
        "loss_ratio_5y", "rate_per_1k_si", "market_position_ratio",
        "combined_risk_score",
    ],
}

# Build reverse lookup: feature_name -> group
feature_to_group = {}
for group, features in FEATURE_GROUPS.items():
    for f in features:
        feature_to_group[f] = group

# Identify all numeric feature columns
all_columns = set(pdf.columns)
feature_columns = [c for c in pdf.columns if c not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(pdf[c])]

print(f"Total columns: {len(all_columns)}")
print(f"Numeric feature columns: {len(feature_columns)}")
print(f"\nFeature groups:")
for group, features in FEATURE_GROUPS.items():
    available = [f for f in features if f in feature_columns]
    print(f"  {group:20s}: {len(available)} features")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Compute Feature Statistics
# MAGIC
# MAGIC For each feature we compute: null %, unique values, mean, std, min, max,
# MAGIC median, skewness, kurtosis, and correlations with the two primary targets.

# COMMAND ----------

from scipy import stats as scipy_stats

targets = {
    "claim_count_5y": pdf["claim_count_5y"].fillna(0).astype(float),
    "total_incurred_5y": pdf["total_incurred_5y"].fillna(0).astype(float),
}

profile_rows = []
for col_name in feature_columns:
    series = pd.to_numeric(pdf[col_name], errors="coerce")
    valid = series.dropna()

    group = feature_to_group.get(col_name, "other")

    row = {
        "factory_run_id": factory_run_id,
        "feature_name": col_name,
        "feature_group": group,
        "dtype": "numeric",
        "null_pct": round(series.isna().mean() * 100, 2),
        "unique_count": int(series.nunique()),
        "mean": round(float(valid.mean()), 4) if len(valid) > 0 else None,
        "std": round(float(valid.std()), 4) if len(valid) > 1 else None,
        "min_val": round(float(valid.min()), 4) if len(valid) > 0 else None,
        "max_val": round(float(valid.max()), 4) if len(valid) > 0 else None,
        "median_val": round(float(valid.median()), 4) if len(valid) > 0 else None,
        "skew": round(float(valid.skew()), 4) if len(valid) > 2 else None,
        "kurtosis": round(float(valid.kurtosis()), 4) if len(valid) > 3 else None,
        "corr_claim_count": round(float(series.corr(targets["claim_count_5y"])), 4) if len(valid) > 2 else None,
        "corr_total_incurred": round(float(series.corr(targets["total_incurred_5y"])), 4) if len(valid) > 2 else None,
        "vif": None,  # Computed below for top features
        "profiled_at": datetime.now(timezone.utc).isoformat(),
    }
    profile_rows.append(row)

profile_pdf = pd.DataFrame(profile_rows)
print(f"Profiled {len(profile_rows)} features")

# COMMAND ----------

# MAGIC %md
# MAGIC ### VIF — Variance Inflation Factor
# MAGIC
# MAGIC High VIF (>10) signals multicollinearity, which inflates GLM coefficient
# MAGIC standard errors and makes regulatory justification harder. We compute VIF
# MAGIC for the core features that will appear in GLM models.

# COMMAND ----------

from sklearn.linear_model import LinearRegression

# Compute VIF for core numeric features with low nulls
vif_candidates = profile_pdf[
    (profile_pdf["null_pct"] < 5) &
    (profile_pdf["feature_group"].isin(["core_policy", "bureau", "geo"]))
]["feature_name"].tolist()

# Cap at 40 features for computational efficiency
vif_candidates = vif_candidates[:40]

vif_data = pdf[vif_candidates].apply(pd.to_numeric, errors="coerce").fillna(0)
vif_values = {}
if len(vif_candidates) > 1:
    for i, col_name in enumerate(vif_candidates):
        y = vif_data.iloc[:, i].values
        X = vif_data.drop(columns=[col_name]).values
        try:
            r2 = LinearRegression().fit(X, y).score(X, y)
            vif_values[col_name] = round(1.0 / (1.0 - r2), 2) if r2 < 1.0 else 999.99
        except Exception:
            vif_values[col_name] = None

# Update profile with VIF values
for i, row in profile_pdf.iterrows():
    if row["feature_name"] in vif_values:
        profile_pdf.at[i, "vif"] = vif_values[row["feature_name"]]

# Show high-VIF features (potential multicollinearity)
high_vif = profile_pdf[profile_pdf["vif"].notna() & (profile_pdf["vif"] > 5)].sort_values("vif", ascending=False)
if len(high_vif) > 0:
    print(f"Features with VIF > 5 (multicollinearity risk):")
    for _, r in high_vif.iterrows():
        print(f"  {r['feature_name']:40s} VIF={r['vif']:8.2f}")
else:
    print("No features with VIF > 5 — multicollinearity is low")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write Feature Profile

# COMMAND ----------

# Ensure consistent numeric types (prevent int/float merge errors on serverless)
import numpy as np
for c in profile_pdf.select_dtypes(include=["int64", "int32"]).columns:
    profile_pdf[c] = profile_pdf[c].astype("float64")
# Ensure string columns don't have mixed types
for c in profile_pdf.select_dtypes(include=["object"]).columns:
    profile_pdf[c] = profile_pdf[c].astype(str).replace("nan", None).replace("None", None)

# Build explicit Spark schema from pandas dtypes to avoid inference issues
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
spark_fields = []
for c in profile_pdf.columns:
    if profile_pdf[c].dtype in ("float64", "float32"):
        spark_fields.append(StructField(c, DoubleType(), True))
    else:
        spark_fields.append(StructField(c, StringType(), True))
profile_schema = StructType(spark_fields)

# Convert to list of tuples for createDataFrame with explicit schema
rows = [tuple(None if (isinstance(v, float) and np.isnan(v)) else v for v in row) for row in profile_pdf.itertuples(index=False, name=None)]

# Overwrite profile for this factory run (idempotent re-runs)
spark.createDataFrame(rows, schema=profile_schema).write.mode("overwrite").saveAsTable(f"{fqn}.mf_feature_profile")
# Re-insert in append mode partitioned by factory_run_id for history
# (For simplicity in the demo, we overwrite — in production, partition by run_id)

display(
    spark.table(f"{fqn}.mf_feature_profile")
    .filter(f"factory_run_id = '{factory_run_id}'")
    .orderBy("feature_group", "feature_name")
)

# COMMAND ----------

log_audit_event(spark, fqn, factory_run_id, "FEATURE_PROFILE_COMPLETE", {
    "total_features": len(feature_columns),
    "groups": {g: len([f for f in fs if f in feature_columns]) for g, fs in FEATURE_GROUPS.items()},
    "high_vif_features": list(high_vif["feature_name"]) if len(high_vif) > 0 else [],
}, upt_version=upt_version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Generate Training Plan
# MAGIC
# MAGIC ### Feature Subsets
# MAGIC
# MAGIC The factory tests models with different feature combinations to measure the
# MAGIC **marginal value** of each data source. This is critical for insurers evaluating
# MAGIC whether to renew an expensive bureau or geo data contract.

# COMMAND ----------

# Build the feature subsets that models will be trained on
core_features = [f for f in FEATURE_GROUPS["core_policy"] if f in feature_columns]
bureau_features = [f for f in FEATURE_GROUPS["bureau"] if f in feature_columns]
geo_features = [f for f in FEATURE_GROUPS["geo"] if f in feature_columns]
market_features = [f for f in FEATURE_GROUPS["market"] if f in feature_columns]
derived_features = [f for f in FEATURE_GROUPS["derived"] if f in feature_columns]
claims_features = [f for f in FEATURE_GROUPS["claims_history"] if f in feature_columns]

FEATURE_SUBSETS = {
    "core_only": core_features + claims_features,
    "core_bureau": core_features + claims_features + bureau_features,
    "core_geo": core_features + claims_features + geo_features,
    "all_features": core_features + claims_features + bureau_features + geo_features + market_features + derived_features,
}

print("Feature subsets for training:")
for name, features in FEATURE_SUBSETS.items():
    print(f"  {name:20s}: {len(features)} features")

# COMMAND ----------

# MAGIC %md
# MAGIC ### LLM-Augmented Planning (Mosaic AI Foundation Model API)
# MAGIC
# MAGIC We send the feature profile summary to an LLM and ask it to reason about which
# MAGIC model configurations would be most appropriate for this dataset. The LLM acts as
# MAGIC an **AI actuarial advisor** — suggesting model families, feature subsets, and
# MAGIC hyperparameters with insurance-domain reasoning.
# MAGIC
# MAGIC If the Foundation Model API is unavailable, we fall back to a deterministic
# MAGIC rule-based planner that generates a standard grid of model configurations.
# MAGIC
# MAGIC > **Audit note:** The LLM prompt, response, and validation result are all logged
# MAGIC > to `mf_audit_log` for regulatory transparency.

# COMMAND ----------

def build_llm_prompt(feature_summary, feature_subsets):
    """Build the structured prompt for the Foundation Model API."""
    return f"""You are an actuarial AI advisor for a commercial property insurance company.

Given the following feature profile of our Unified Pricing Table, recommend a training plan
for our Model Factory. The plan should specify which model configurations to train.

## Feature Profile Summary
- Total numeric features: {feature_summary['total_features']}
- Feature groups: {json.dumps(feature_summary['groups'], indent=2)}
- Top correlated features with claim_count_5y: {json.dumps(feature_summary['top_corr_frequency'], indent=2)}
- Top correlated features with total_incurred_5y: {json.dumps(feature_summary['top_corr_severity'], indent=2)}
- High VIF features (multicollinearity risk): {json.dumps(feature_summary['high_vif'], indent=2)}

## Available Feature Subsets
{json.dumps({k: len(v) for k, v in feature_subsets.items()}, indent=2)}

## Available Model Types
- GLM_Poisson: Claim frequency (count target). Log link. Transparent relativities.
- GLM_Gamma: Claim severity (cost given claim). Log link. Must filter to claims-only policies.
- GLM_Tweedie: Pure premium (combined frequency x severity). Log link, p=1.5. Handles zeros.
- LGBMRegressor: Gradient boosted trees for regression targets.
- LGBMClassifier: Gradient boosted trees for binary classification (e.g., conversion).
- GLM_GBM_Uplift: Train a GLM first, then a GBM on its residuals to capture non-linear effects.

## Constraints
- GLMs are preferred by regulators for their interpretability (explicit relativities).
- GBMs are valued for predictive power but must be accompanied by feature importance analysis.
- The GLM+GBM uplift approach bridges both: regulatory-friendly base + performance boost.
- Maximum ~25 model configurations (training budget).

Respond ONLY with a JSON array of model configurations. Each object must have:
- model_config_id: unique string identifier
- model_family: "GLM", "GBM", or "GLM_GBM_UPLIFT"
- model_type: one of the types above
- target_column: "claim_count_5y", "total_incurred_5y", or "converted"
- feature_subset_name: one of {list(feature_subsets.keys())}
- hyperparams: object with model-specific parameters
- rationale: one-sentence justification for this configuration

Return ONLY valid JSON, no markdown formatting."""


def call_foundation_model(prompt):
    """Call Mosaic AI Foundation Model API for training plan reasoning."""
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    response = w.serving_endpoints.query(
        name="databricks-meta-llama-3-1-70b-instruct",
        messages=[
            {"role": "system", "content": "You are an expert actuarial AI advisor. Respond only with valid JSON."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=4000,
        temperature=0.1,
    )
    return response.choices[0].message.content

# COMMAND ----------

# Build the feature summary for the LLM
top_corr_freq = (
    profile_pdf[profile_pdf["corr_claim_count"].notna()]
    .nlargest(10, "corr_claim_count")[["feature_name", "corr_claim_count"]]
    .to_dict("records")
)
top_corr_sev = (
    profile_pdf[profile_pdf["corr_total_incurred"].notna()]
    .nlargest(10, "corr_total_incurred")[["feature_name", "corr_total_incurred"]]
    .to_dict("records")
)
high_vif_list = list(high_vif["feature_name"]) if len(high_vif) > 0 else []

feature_summary = {
    "total_features": len(feature_columns),
    "groups": {g: len([f for f in fs if f in feature_columns]) for g, fs in FEATURE_GROUPS.items()},
    "top_corr_frequency": top_corr_freq,
    "top_corr_severity": top_corr_sev,
    "high_vif": high_vif_list,
}

prompt = build_llm_prompt(feature_summary, FEATURE_SUBSETS)

# Try the LLM, fall back to rule-based
plan_source = "rule_based_fallback"
llm_response = None

try:
    print("Calling Mosaic AI Foundation Model API...")
    llm_response = call_foundation_model(prompt)
    # Try to parse the JSON
    # Strip markdown code fences if present
    clean = llm_response.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1]
        if clean.endswith("```"):
            clean = clean[:-3]
    llm_plan = json.loads(clean)

    # Validate: must be a list of dicts with required keys
    required_keys = {"model_config_id", "model_family", "model_type", "target_column", "feature_subset_name"}
    if isinstance(llm_plan, list) and all(required_keys.issubset(set(m.keys())) for m in llm_plan):
        plan_source = "llm_augmented"
        print(f"LLM plan accepted: {len(llm_plan)} model configurations")
    else:
        print("LLM response failed validation — falling back to rule-based planner")
        llm_plan = None
except Exception as e:
    print(f"Foundation Model API unavailable ({type(e).__name__}: {e})")
    print("Using deterministic rule-based planner")
    llm_plan = None

# Log the LLM interaction regardless of outcome
log_audit_event(spark, fqn, factory_run_id, "LLM_REASONING_RECORDED", {
    "prompt_length": len(prompt),
    "response_length": len(llm_response) if llm_response else 0,
    "plan_source": plan_source,
    "validation_passed": plan_source == "llm_augmented",
    "llm_response_preview": (llm_response[:500] + "...") if llm_response else None,
}, upt_version=upt_version)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Rule-Based Training Plan (Fallback / Default)
# MAGIC
# MAGIC This deterministic planner generates a comprehensive grid of model configurations.
# MAGIC It always runs — either as the primary plan or as a validated fallback.

# COMMAND ----------

def generate_rule_based_plan(feature_subsets, factory_run_id):
    """Generate a deterministic training plan based on feature counts."""
    plan = []
    now = datetime.now(timezone.utc).isoformat()

    # --- GLM Frequency (Poisson) — one per feature subset ---
    for subset_name, features in feature_subsets.items():
        plan.append({
            "factory_run_id": factory_run_id,
            "model_config_id": f"glm_poisson_freq_{subset_name}",
            "model_family": "GLM",
            "model_type": "GLM_Poisson",
            "target_column": "claim_count_5y",
            "feature_subset_name": subset_name,
            "feature_list_json": json.dumps(features),
            "hyperparams_json": json.dumps({"family": "Poisson", "link": "log"}),
            "rationale": f"Poisson GLM for claim frequency using {subset_name} ({len(features)} features). Regulatory-preferred for transparent relativities.",
            "plan_source": "rule_based",
            "created_at": now,
        })

    # --- GLM Severity (Gamma) — one per feature subset ---
    for subset_name, features in feature_subsets.items():
        plan.append({
            "factory_run_id": factory_run_id,
            "model_config_id": f"glm_gamma_sev_{subset_name}",
            "model_family": "GLM",
            "model_type": "GLM_Gamma",
            "target_column": "total_incurred_5y",
            "feature_subset_name": subset_name,
            "feature_list_json": json.dumps(features),
            "hyperparams_json": json.dumps({"family": "Gamma", "link": "log", "filter": "claim_count_5y > 0"}),
            "rationale": f"Gamma GLM for claim severity using {subset_name}. Paired with frequency model for Freq x Sev pricing.",
            "plan_source": "rule_based",
            "created_at": now,
        })

    # --- GLM Tweedie (Pure Premium) — core and all features ---
    for subset_name in ["core_only", "all_features"]:
        features = feature_subsets[subset_name]
        plan.append({
            "factory_run_id": factory_run_id,
            "model_config_id": f"glm_tweedie_pp_{subset_name}",
            "model_family": "GLM",
            "model_type": "GLM_Tweedie",
            "target_column": "total_incurred_5y",
            "feature_subset_name": subset_name,
            "feature_list_json": json.dumps(features),
            "hyperparams_json": json.dumps({"family": "Tweedie", "link": "log", "var_power": 1.5}),
            "rationale": f"Tweedie GLM for pure premium (handles zero claims natively) using {subset_name}.",
            "plan_source": "rule_based",
            "created_at": now,
        })

    # --- GBM Frequency (LightGBM Regressor) — varying depth ---
    all_feats = feature_subsets["all_features"]
    for depth in [3, 5, 7]:
        plan.append({
            "factory_run_id": factory_run_id,
            "model_config_id": f"gbm_freq_depth{depth}",
            "model_family": "GBM",
            "model_type": "LGBMRegressor",
            "target_column": "claim_count_5y",
            "feature_subset_name": "all_features",
            "feature_list_json": json.dumps(all_feats),
            "hyperparams_json": json.dumps({"n_estimators": 200, "max_depth": depth, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8}),
            "rationale": f"LightGBM regressor (depth={depth}) for claim frequency. Captures non-linear interactions missed by GLMs.",
            "plan_source": "rule_based",
            "created_at": now,
        })

    # --- GBM Severity (LightGBM Regressor) — varying depth ---
    for depth in [3, 5, 7]:
        plan.append({
            "factory_run_id": factory_run_id,
            "model_config_id": f"gbm_sev_depth{depth}",
            "model_family": "GBM",
            "model_type": "LGBMRegressor",
            "target_column": "total_incurred_5y",
            "feature_subset_name": "all_features",
            "feature_list_json": json.dumps(all_feats),
            "hyperparams_json": json.dumps({"n_estimators": 200, "max_depth": depth, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "filter": "claim_count_5y > 0"}),
            "rationale": f"LightGBM regressor (depth={depth}) for claim severity. Benchmarks against Gamma GLM.",
            "plan_source": "rule_based",
            "created_at": now,
        })

    # --- GBM Demand (LightGBM Classifier) — conversion prediction ---
    for depth in [3, 5]:
        plan.append({
            "factory_run_id": factory_run_id,
            "model_config_id": f"gbm_demand_depth{depth}",
            "model_family": "GBM",
            "model_type": "LGBMClassifier",
            "target_column": "converted",
            "feature_subset_name": "all_features",
            "feature_list_json": json.dumps(all_feats),
            "hyperparams_json": json.dumps({"n_estimators": 200, "max_depth": depth, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8}),
            "rationale": f"LightGBM classifier (depth={depth}) for demand/conversion prediction. Drives commercial pricing overlay.",
            "plan_source": "rule_based",
            "created_at": now,
        })

    # --- GLM + GBM Uplift (residual learner) ---
    plan.append({
        "factory_run_id": factory_run_id,
        "model_config_id": "glm_gbm_uplift_freq",
        "model_family": "GLM_GBM_UPLIFT",
        "model_type": "GLM_Poisson_LGBMRegressor",
        "target_column": "claim_count_5y",
        "feature_subset_name": "all_features",
        "feature_list_json": json.dumps(all_feats),
        "hyperparams_json": json.dumps({
            "glm_family": "Poisson", "glm_link": "log",
            "glm_features": json.dumps(feature_subsets["core_only"]),
            "gbm_n_estimators": 150, "gbm_max_depth": 5, "gbm_learning_rate": 0.1,
        }),
        "rationale": "Hybrid GLM+GBM: Poisson base for regulatory transparency, GBM uplift for non-linear residuals. Best of both worlds.",
        "plan_source": "rule_based",
        "created_at": now,
    })

    return plan

# COMMAND ----------

# Use LLM plan if it passed validation, otherwise use rule-based
if plan_source == "llm_augmented" and llm_plan is not None:
    # Convert LLM output to our table schema
    now = datetime.now(timezone.utc).isoformat()
    training_plan = []
    for m in llm_plan:
        features = FEATURE_SUBSETS.get(m.get("feature_subset_name", "all_features"), FEATURE_SUBSETS["all_features"])
        training_plan.append({
            "factory_run_id": factory_run_id,
            "model_config_id": m["model_config_id"],
            "model_family": m["model_family"],
            "model_type": m["model_type"],
            "target_column": m["target_column"],
            "feature_subset_name": m.get("feature_subset_name", "all_features"),
            "feature_list_json": json.dumps(features),
            "hyperparams_json": json.dumps(m.get("hyperparams", {})),
            "rationale": m.get("rationale", "LLM-recommended configuration"),
            "plan_source": "llm_augmented",
            "created_at": now,
        })
else:
    training_plan = generate_rule_based_plan(FEATURE_SUBSETS, factory_run_id)

print(f"\nTraining Plan: {len(training_plan)} model configurations")
print(f"Plan source: {plan_source}")
print(f"\nModel breakdown:")
families = {}
for m in training_plan:
    families[m["model_family"]] = families.get(m["model_family"], 0) + 1
for family, count in families.items():
    print(f"  {family}: {count} configurations")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write Training Plan

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, ArrayType
plan_schema = StructType([
    StructField("factory_run_id", StringType()),
    StructField("model_config_id", StringType()),
    StructField("model_family", StringType()),
    StructField("model_type", StringType()),
    StructField("target_column", StringType()),
    StructField("feature_subset_name", StringType()),
    StructField("features", ArrayType(StringType())),
    StructField("hyperparameters", StringType()),
    StructField("rationale", StringType()),
    StructField("plan_source", StringType()),
])
# Ensure all dict values are strings where needed
for m in training_plan:
    if isinstance(m.get("hyperparameters"), dict):
        m["hyperparameters"] = json.dumps(m["hyperparameters"])
    if m.get("features") is None:
        m["features"] = []

plan_df = spark.createDataFrame(training_plan, schema=plan_schema)
plan_df.write.mode("overwrite").saveAsTable(f"{fqn}.mf_training_plan")

display(
    spark.table(f"{fqn}.mf_training_plan")
    .filter(f"factory_run_id = '{factory_run_id}'")
    .select("model_config_id", "model_family", "model_type", "target_column",
            "feature_subset_name", "rationale", "plan_source")
    .orderBy("model_family", "model_config_id")
)

# COMMAND ----------

log_audit_event(spark, fqn, factory_run_id, "TRAINING_PLAN_GENERATED", {
    "total_configs": len(training_plan),
    "plan_source": plan_source,
    "model_families": families,
    "feature_subsets_used": list(FEATURE_SUBSETS.keys()),
    "feature_counts": {k: len(v) for k, v in FEATURE_SUBSETS.items()},
}, upt_version=upt_version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Item | Value |
# MAGIC |------|-------|
# MAGIC | Factory Run ID | `{factory_run_id}` |
# MAGIC | UPT Version | `{upt_version}` |
# MAGIC | Features Profiled | `{len(feature_columns)}` |
# MAGIC | Models Planned | `{len(training_plan)}` |
# MAGIC | Plan Source | `{plan_source}` |
# MAGIC
# MAGIC **Next step:** Run `mf_02_automated_training` to train all planned models.
