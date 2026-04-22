# Databricks notebook source
# MAGIC %md
# MAGIC # Run Full Demo Pipeline
# MAGIC
# MAGIC Executes the entire pricing accelerator from scratch. Use this to rebuild
# MAGIC the demo environment on any workspace.
# MAGIC
# MAGIC **Time:** ~15-20 minutes at SCALE_FACTOR=1
# MAGIC
# MAGIC **Steps:**
# MAGIC 1. Setup (schema, tables, test data)
# MAGIC 2. Ingestion (CSVs → Bronze → Silver)
# MAGIC 3. Gold (build Unified Pricing Table)
# MAGIC 4. Models (train 4 core + 2 supplementary)
# MAGIC 5. Use cases (shadow pricing, PIT, enriched pricing)

# COMMAND ----------

dbutils.widgets.text("catalog_name", "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name", "pricing_upt")
dbutils.widgets.text("volume_name", "external_landing")
dbutils.widgets.text("scale_factor", "1")

catalog = dbutils.widgets.get("catalog_name")
schema = dbutils.widgets.get("schema_name")
volume = dbutils.widgets.get("volume_name")
scale = dbutils.widgets.get("scale_factor")

print(f"Full demo pipeline — {catalog}.{schema} (scale={scale}x)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Setup

# COMMAND ----------

dbutils.notebook.run("setup", 600, {
    "catalog_name": catalog,
    "schema_name": schema,
    "volume_name": volume,
    "scale_factor": scale,
})
print("✓ Step 1: Setup complete")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Ingestion (Bronze → Silver via DLT)
# MAGIC
# MAGIC Note: DLT pipeline runs as a separate job. The ingestion notebooks below
# MAGIC load CSVs to bronze. Silver is built by the DLT pipeline (run separately
# MAGIC via `databricks bundle run ingest_external_data`).

# COMMAND ----------

# Run ingestion notebooks in sequence (DLT handles silver)
for nb in ["../01_ingestion/ingest_market_pricing",
           "../01_ingestion/ingest_geospatial_hazard",
           "../01_ingestion/ingest_credit_bureau",
           "../01_ingestion/ingest_ons_reference"]:
    dbutils.notebook.run(nb, 300, {
        "catalog_name": catalog,
        "schema_name": schema,
        "volume_name": volume,
    })

print("✓ Step 2: Bronze ingestion complete")
print("  Note: Run the DLT pipeline separately to build silver tables:")
print("  databricks bundle run ingest_external_data")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Derived factors + Gold (Unified Pricing Table)

# COMMAND ----------

# Derived factors (urban_score, neighbourhood_claim_frequency) — postcode-level.
# Must run before build_upt so the UPT picks them up.
dbutils.notebook.run("../03_gold/derive_factors", 300, {
    "catalog_name": catalog,
    "schema_name": schema,
})
print("✓ Step 3a: Derived factors built")

dbutils.notebook.run("../03_gold/build_upt", 600, {
    "catalog_name": catalog,
    "schema_name": schema,
})
print("✓ Step 3b: Gold UPT built")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Train Core Models

# COMMAND ----------

# GLM Frequency
dbutils.notebook.run("../04_models/model_01_glm_frequency", 600, {
    "catalog_name": catalog,
    "schema_name": schema,
})
print("✓ GLM Frequency trained")

# COMMAND ----------

# GLM Severity
dbutils.notebook.run("../04_models/model_02_glm_severity", 600, {
    "catalog_name": catalog,
    "schema_name": schema,
})
print("✓ GLM Severity trained")

# COMMAND ----------

# GBM Demand
dbutils.notebook.run("../04_models/model_03_gbm_demand", 600, {
    "catalog_name": catalog,
    "schema_name": schema,
})
print("✓ GBM Demand trained")

# COMMAND ----------

# GBM Risk Uplift (depends on frequency GLM)
dbutils.notebook.run("../04_models/model_04_gbm_risk_uplift", 600, {
    "catalog_name": catalog,
    "schema_name": schema,
})
print("✓ GBM Risk Uplift trained")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Train Supplementary Models

# COMMAND ----------

dbutils.notebook.run("../04_models/model_05_fraud_propensity", 600, {
    "catalog_name": catalog,
    "schema_name": schema,
})
print("✓ Fraud Propensity trained")

# COMMAND ----------

dbutils.notebook.run("../04_models/model_06_retention", 600, {
    "catalog_name": catalog,
    "schema_name": schema,
})
print("✓ Retention/Churn trained")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Run Use Cases

# COMMAND ----------

dbutils.notebook.run("../05_use_cases/uc1_shadow_pricing", 600, {
    "catalog_name": catalog,
    "schema_name": schema,
})
print("✓ UC1: Shadow Pricing complete")

# COMMAND ----------

dbutils.notebook.run("../05_use_cases/uc2_point_in_time", 600, {
    "catalog_name": catalog,
    "schema_name": schema,
    "snapshot_version": "0",
})
print("✓ UC2: Point-in-Time Backtesting complete")

# COMMAND ----------

dbutils.notebook.run("../05_use_cases/uc5_enriched_pricing", 600, {
    "catalog_name": catalog,
    "schema_name": schema,
})
print("✓ UC5: Enriched Pricing complete")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Done!

# COMMAND ----------

print(f"""
═══════════════════════════════════════════════
  Full Demo Pipeline Complete
═══════════════════════════════════════════════
  Catalog:   {catalog}
  Schema:    {schema}
  Scale:     {scale}x

  ✓ Setup (tables + test data)
  ✓ Bronze ingestion (3 external datasets)
  ✓ Gold UPT built
  ✓ 6 models trained (freq, sev, demand, uplift, fraud, retention)
  ✓ Use cases run (shadow pricing, PIT, enriched pricing)

  Next steps:
  1. Run DLT pipeline: databricks bundle run ingest_external_data
  2. Open the HITL app in the Serving UI
  3. Optionally: databricks bundle run setup_online_store
  4. Optionally: databricks bundle run deploy_model_endpoint
═══════════════════════════════════════════════
""")
