# Databricks notebook source
# MAGIC %md
# MAGIC # Apply UC descriptions to the 4 motor production models
# MAGIC
# MAGIC `databricks-feature-engineering` logs models into UC but doesn't set the
# MAGIC registered-model `comment` (description) field that Catalog Explorer
# MAGIC surfaces. This notebook patches each motor model with a clear
# MAGIC description of what it does, what features it uses, and what target
# MAGIC it predicts. Idempotent — re-runs overwrite the description.

# COMMAND ----------

dbutils.widgets.text("catalog_name", "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name",  "pricing_upt")

catalog = dbutils.widgets.get("catalog_name")
schema  = dbutils.widgets.get("schema_name")
fqn     = f"{catalog}.{schema}"

# COMMAND ----------

# MAGIC %pip install mlflow --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog = dbutils.widgets.get("catalog_name")
schema  = dbutils.widgets.get("schema_name")
fqn     = f"{catalog}.{schema}"

import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_registry_uri("databricks-uc")
mc = MlflowClient()

DESCRIPTIONS = {
    "freq_glm_motor": (
        "Motor Frequency model. Poisson GLM that predicts the expected "
        "claim count per policy over a 5-year window. Features: driver "
        "demographics (age, license_years_held, no_claims_years, occupation), "
        "vehicle attributes (group, value, age, annual_mileage, parking), "
        "and telematics behaviour (behaviour_score, recent_speeding_events, "
        "recent_curfew_breaches, recent_harsh_braking_30d). Sourced from "
        f"{fqn}.unified_motor_table_live via FeatureLookup. Used inside the "
        "motor_pricing_scorer pyfunc for technical-premium derivation."
    ),
    "sev_glm_motor": (
        "Motor Severity model. Gamma GLM that predicts the mean claim "
        "severity (£/claim) for policies with at least one observed claim. "
        "Features: driver demographics, vehicle attributes, telematics "
        "behaviour, and at-fault claim history. Trained only on claimants "
        "(claim_count_5y > 0). Used inside motor_pricing_scorer to combine "
        "with frequency into the technical premium."
    ),
    "demand_gbm_motor": (
        "Motor Demand / Conversion model. LightGBM binary classifier that "
        "predicts the probability a customer will accept the quote at their "
        "current_premium. Features: driver demographics, vehicle attributes, "
        "price-per-£1k-SI positioning, behaviour_score (loyalty proxy), and "
        "claim history. Used by the motor rating engine to apply a small "
        "+/- adjustment: low demand → discount, high demand → loading."
    ),
    "fraud_gbm_motor": (
        "Motor Fraud Propensity model. LightGBM binary classifier flagging "
        "policies where the driver is likely to file a fraudulent claim. "
        "Features: driver demographics, prior_convictions, prior_accidents, "
        "telematics behaviour (behaviour_score, recent_speeding_events, "
        "recent_curfew_breaches, night_driving_pct), and claim history. "
        "Used inside motor_pricing_scorer to trigger a fraud loading on the "
        "quote when probability exceeds 0.20."
    ),
}

for name, comment in DESCRIPTIONS.items():
    full = f"{fqn}.{name}"
    try:
        mc.update_registered_model(name=full, description=comment)
        print(f"✓ {full}")
    except Exception as e:
        print(f"✗ {full}: {e}")

# Also describe the unified pricing scorer itself
SCORER_DESC = (
    "Unified Motor Pricing Scorer. PyFunc model that wraps the 4 motor "
    "champions (freq_glm_motor, sev_glm_motor, demand_gbm_motor, "
    "fraud_gbm_motor) plus a deterministic rating engine in a single "
    "Model Serving call. Takes a policy_id, resolves features from "
    f"{fqn}.unified_motor_table_live via FeatureLookup against the "
    "motor-pricing-online-store Lakebase publish, runs all four models, "
    "applies the rating engine (technical = freq*sev → expense + commission "
    "→ young-driver loading → telematics-event surcharge → fraud loading "
    "→ demand adjustment → clip), and returns final_premium plus every "
    "intermediate value the audit log needs. Backs the live-serving demo."
)
try:
    mc.update_registered_model(name=f"{fqn}.motor_pricing_scorer", description=SCORER_DESC)
    print(f"✓ {fqn}.motor_pricing_scorer")
except Exception as e:
    print(f"✗ scorer: {e}")

# COMMAND ----------

import json
dbutils.notebook.exit(json.dumps({"models_described": list(DESCRIPTIONS.keys()) + ["motor_pricing_scorer"]}))
