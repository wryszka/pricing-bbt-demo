# Databricks notebook source
# MAGIC %md
# MAGIC # UC4: Lineage & Governance Demo
# MAGIC
# MAGIC **What this demonstrates:** Unity Catalog automatically tracks data-to-model
# MAGIC lineage. The approval layer (`audit_log`) adds human governance on top.
# MAGIC Together they provide end-to-end traceability from raw source → silver →
# MAGIC gold feature table → model — with a record of who approved every step.
# MAGIC
# MAGIC **Key compliance statements:**
# MAGIC - "We can trace any model prediction back to the exact feature table version it was trained on"
# MAGIC - "Every data merge was approved by a named actuary with a timestamp"
# MAGIC - "Delta Time Travel lets us reconstruct the exact state at any point in history"

# COMMAND ----------

dbutils.widgets.text("catalog_name", "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name", "pricing_upt")

catalog = dbutils.widgets.get("catalog_name")
schema = dbutils.widgets.get("schema_name")
fqn = f"{catalog}.{schema}"
upt_table = f"{fqn}.unified_pricing_table_live"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Feature Table Discovery
# MAGIC The UPT is registered as a UC feature table (via PRIMARY KEY constraint).
# MAGIC Tags and column comments make it discoverable in the Features UI.

# COMMAND ----------

# Show the table metadata — tags, constraints, properties
display(spark.sql(f"DESCRIBE DETAIL {upt_table}"))

# COMMAND ----------

# Show table tags
display(spark.sql(f"SELECT * FROM information_schema.table_tags WHERE schema_name = '{schema}' AND table_name = 'unified_pricing_table_live'"))

# COMMAND ----------

# Show column comments — these appear in the Features UI for discoverability
display(spark.sql(f"""
    SELECT column_name, comment, data_type
    FROM information_schema.columns
    WHERE table_schema = '{schema}'
      AND table_name = 'unified_pricing_table_live'
      AND comment IS NOT NULL AND comment != ''
    ORDER BY ordinal_position
"""))

# COMMAND ----------

# Show primary key constraint — this is what makes it a feature table
display(spark.sql(f"""
    SELECT constraint_name, column_name
    FROM information_schema.constraint_column_usage
    WHERE table_schema = '{schema}'
      AND table_name = 'unified_pricing_table_live'
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Delta Version History
# MAGIC Every write to the UPT creates a new Delta version. Combined with the
# MAGIC audit_log, we know exactly which human approved which version.

# COMMAND ----------

display(spark.sql(f"DESCRIBE HISTORY {upt_table} LIMIT 20"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Approval Audit Trail
# MAGIC The `audit_log` table records every human decision — dataset approvals,
# MAGIC model approvals, manual overrides. Each event includes the Delta version
# MAGIC at the time of the decision.

# COMMAND ----------

display(spark.sql(f"""
    SELECT event_id, event_type, entity_type, entity_id,
           user_id, timestamp, source,
           details
    FROM {fqn}.audit_log
    ORDER BY timestamp DESC
    LIMIT 20
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Model → Feature Table Lineage
# MAGIC When models are trained, they log the UPT as an input dataset via
# MAGIC `mlflow.log_input()`. This creates a lineage link visible in
# MAGIC Catalog Explorer: Model → Feature Table → Source Tables.

# COMMAND ----------

import mlflow

# Find the most recent model training runs that reference our feature table
client = mlflow.tracking.MlflowClient()

# Search for runs tagged with our feature table
experiments = client.search_experiments()
recent_runs = []
for exp in experiments:
    if "pricing_upt" in (exp.name or ""):
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=f"tags.feature_table = '{upt_table}'",
            max_results=5,
            order_by=["start_time DESC"],
        )
        recent_runs.extend(runs)

if recent_runs:
    run_data = []
    for r in recent_runs[:10]:
        run_data.append({
            "run_id": r.info.run_id,
            "run_name": r.data.tags.get("mlflow.runName", ""),
            "experiment": r.info.experiment_id,
            "model_type": r.data.params.get("model_type", ""),
            "upt_delta_version": r.data.params.get("upt_delta_version", ""),
            "feature_table": r.data.tags.get("feature_table", ""),
            "start_time": r.info.start_time,
            "rmse": r.data.metrics.get("rmse", r.data.metrics.get("combined_rmse", None)),
            "r2": r.data.metrics.get("r2", r.data.metrics.get("combined_r2", None)),
        })

    display(spark.createDataFrame(run_data))
    print(f"Found {len(recent_runs)} model runs linked to {upt_table}")
else:
    print("No model runs found with feature_table tag — run model training first")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Point-in-Time Comparison
# MAGIC For any model run, we can retrieve the exact feature table version it trained on
# MAGIC and compare it to the current live table.

# COMMAND ----------

# Get the latest version and version 0 for comparison
latest_version = spark.sql(f"DESCRIBE HISTORY {upt_table} LIMIT 1").collect()[0]["version"]

print(f"Current version: {latest_version}")
print(f"Comparing version 0 (initial) vs version {latest_version} (current)")

# Version 0 stats
v0_stats = spark.read.format("delta").option("versionAsOf", 0).table(upt_table).agg(
    F.count("*").alias("rows"),
    F.avg("current_premium").alias("avg_premium"),
    F.avg("combined_risk_score").alias("avg_risk_score"),
)

# Current stats
current_stats = spark.table(upt_table).agg(
    F.count("*").alias("rows"),
    F.avg("current_premium").alias("avg_premium"),
    F.avg("combined_risk_score").alias("avg_risk_score"),
)

import pyspark.sql.functions as F

comparison = (v0_stats.withColumn("version", F.lit("v0 (initial)"))
    .unionByName(current_stats.withColumn("version", F.lit(f"v{latest_version} (current)")))
)
display(comparison)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Combined View: UC Lineage + Human Governance
# MAGIC
# MAGIC This is the complete picture:
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────┐
# MAGIC │                  Unity Catalog (Automatic)                  │
# MAGIC ├─────────────────────────────────────────────────────────────┤
# MAGIC │                                                             │
# MAGIC │  Raw Tables ──→ Silver (DLT) ──→ Gold UPT ──→ Models       │
# MAGIC │       │              │              │            │          │
# MAGIC │   (lineage)     (lineage)     (lineage)    (lineage)       │
# MAGIC │                                                             │
# MAGIC └─────────────────────────────────────────────────────────────┘
# MAGIC
# MAGIC ┌─────────────────────────────────────────────────────────────┐
# MAGIC │              audit_log (Human Governance)                   │
# MAGIC ├─────────────────────────────────────────────────────────────┤
# MAGIC │                                                             │
# MAGIC │  dataset_approved → Jane @ 2026-01-15 → Delta v3 → v4      │
# MAGIC │  model_approved   → John @ 2026-01-16 → trained on v4      │
# MAGIC │  model_deployed   → Jane @ 2026-01-17 → endpoint live      │
# MAGIC │                                                             │
# MAGIC └─────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC **What Unity Catalog handles natively:**
# MAGIC - Data lineage (which tables feed into which)
# MAGIC - Feature discovery (searchable in Features UI)
# MAGIC - Version history (Delta Time Travel)
# MAGIC - Access control (ACLs)
# MAGIC - Column-level metadata (comments, tags)
# MAGIC
# MAGIC **What the audit_log adds:**
# MAGIC - Human approval chain (who approved what, when, why)
# MAGIC - Delta version bookmarks (before/after each approval)
# MAGIC - Regulatory evidence (named reviewer + timestamp + notes)
# MAGIC - Cross-entity correlation (link a model approval to the dataset approvals it depends on)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print(f"""
Lineage & Governance Summary
=============================
Feature Table:   {upt_table}
Delta Versions:  {latest_version + 1} versions available
Primary Key:     policy_id
Tags:            business_line, pricing_domain, table_owner, refresh_cadence, demo_environment, contains_pii

Model Lineage:   {len(recent_runs) if recent_runs else 0} model runs linked to this feature table
Audit Events:    see audit_log table above

Navigate to Catalog Explorer → {catalog} → {schema} → unified_pricing_table_live
to see the Features UI, lineage graph, and Delta history.
""")
