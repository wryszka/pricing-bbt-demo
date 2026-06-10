# Databricks notebook source
# MAGIC %md
# MAGIC # UC2: Point-in-Time Backtesting — Delta Time Travel
# MAGIC
# MAGIC **The "Time Machine":** A Data Scientist enters a target date, and the system
# MAGIC queries the Unified Pricing Table as it existed at that exact moment using
# MAGIC Delta Lake's native Time Travel. It then compares the frozen snapshot to the
# MAGIC current live table to detect feature drift.
# MAGIC
# MAGIC **Compliance value:** Proves to auditors that the 2024 pricing model had
# MAGIC zero visibility into 2025 data. Instant IFC compliance.

# COMMAND ----------

dbutils.widgets.text("catalog_name", "pricing_workbench")
dbutils.widgets.text("schema_name", "pricing_upt")
dbutils.widgets.text("snapshot_version", "0")

catalog = dbutils.widgets.get("catalog_name")
schema = dbutils.widgets.get("schema_name")
snapshot_version = int(dbutils.widgets.get("snapshot_version"))
fqn_prefix = f"{catalog}.{schema}"
upt_table = f"{fqn_prefix}.unified_pricing_table_live"

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.functions import col, when, lit, round as spark_round, abs as spark_abs

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Explore Delta History
# MAGIC Show all available versions of the UPT with timestamps and operations.

# COMMAND ----------

history = spark.sql(f"DESCRIBE HISTORY {upt_table}")
display(history.select("version", "timestamp", "operation", "operationParameters", "operationMetrics"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Load frozen snapshot vs live table

# COMMAND ----------

# Live (current) version
live_df = spark.table(upt_table)
live_count = live_df.count()
live_cols = len(live_df.columns)

# Historical version using Time Travel
frozen_df = spark.read.format("delta").option("versionAsOf", snapshot_version).table(upt_table)
frozen_count = frozen_df.count()

# Get the timestamp of the frozen version
version_info = history.filter(col("version") == snapshot_version).collect()
frozen_timestamp = version_info[0]["timestamp"] if version_info else "unknown"

print(f"Live UPT:   version=latest, rows={live_count:,}, columns={live_cols}")
print(f"Frozen UPT: version={snapshot_version}, rows={frozen_count:,}, timestamp={frozen_timestamp}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Feature Drift Analysis
# MAGIC Compare statistical distributions of key features between frozen and live.

# COMMAND ----------

drift_features = [
    "current_premium", "sum_insured", "annual_turnover",
    "flood_zone_rating", "crime_theft_index", "subsidence_risk",
    "composite_location_risk", "credit_score", "business_stability_score",
    "market_median_rate", "loss_ratio_5y", "combined_risk_score",
    "building_age_years",
]

# Compute stats for both versions
def compute_stats(df, label):
    aggs = []
    for feat in drift_features:
        aggs.extend([
            F.avg(col(feat).cast("double")).alias(f"{feat}_mean"),
            F.stddev(col(feat).cast("double")).alias(f"{feat}_std"),
            F.min(col(feat).cast("double")).alias(f"{feat}_min"),
            F.max(col(feat).cast("double")).alias(f"{feat}_max"),
            (F.count(when(col(feat).isNull(), 1)) / F.count("*") * 100).alias(f"{feat}_null_pct"),
        ])
    stats = df.agg(*aggs).collect()[0]
    return {feat: {
        "mean": stats[f"{feat}_mean"],
        "std": stats[f"{feat}_std"],
        "min": stats[f"{feat}_min"],
        "max": stats[f"{feat}_max"],
        "null_pct": stats[f"{feat}_null_pct"],
    } for feat in drift_features}

live_stats = compute_stats(live_df, "live")
frozen_stats = compute_stats(frozen_df, "frozen")

# Build drift summary
drift_rows = []
for feat in drift_features:
    live_mean = live_stats[feat]["mean"]
    frozen_mean = frozen_stats[feat]["mean"]
    if live_mean and frozen_mean and frozen_mean != 0:
        drift_pct = float((live_mean - frozen_mean) / abs(frozen_mean) * 100)
    else:
        drift_pct = 0.0

    drift_rows.append({
        "feature": feat,
        "frozen_mean": float(round(frozen_mean, 2)) if frozen_mean else None,
        "live_mean": float(round(live_mean, 2)) if live_mean else None,
        "drift_pct": float(round(drift_pct, 2)),
        "frozen_null_pct": float(round(frozen_stats[feat]["null_pct"], 1)) if frozen_stats[feat]["null_pct"] else 0.0,
        "live_null_pct": float(round(live_stats[feat]["null_pct"], 1)) if live_stats[feat]["null_pct"] else 0.0,
        "drift_severity": "HIGH" if abs(drift_pct) > 10 else ("MEDIUM" if abs(drift_pct) > 5 else "LOW"),
    })

drift_df = spark.createDataFrame(drift_rows)
display(drift_df.orderBy(F.abs(col("drift_pct")).desc()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Write drift summary table

# COMMAND ----------

drift_df.write.mode("overwrite").saveAsTable(f"{fqn_prefix}.pit_drift_summary")
print(f"✓ Drift summary saved to {fqn_prefix}.pit_drift_summary")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Row-level changes — which policies changed most?

# COMMAND ----------

# Join live and frozen on policy_id to find per-policy changes
comparison = (live_df.alias("live")
    .join(frozen_df.alias("frozen"), "policy_id", "inner")
    .select(
        col("policy_id"),
        col("live.current_premium").alias("live_premium"),
        col("frozen.current_premium").alias("frozen_premium"),
        col("live.composite_location_risk").alias("live_location_risk"),
        col("frozen.composite_location_risk").alias("frozen_location_risk"),
        col("live.credit_score").alias("live_credit"),
        col("frozen.credit_score").alias("frozen_credit"),
        col("live.combined_risk_score").alias("live_risk_score"),
        col("frozen.combined_risk_score").alias("frozen_risk_score"),
    )
    .withColumn("premium_change", col("live_premium") - col("frozen_premium"))
    .withColumn("risk_score_change", col("live_risk_score") - col("frozen_risk_score"))
)

# Show policies with biggest changes
display(
    comparison
    .withColumn("abs_risk_change", spark_abs(col("risk_score_change")))
    .orderBy(col("abs_risk_change").desc())
    .limit(30)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Save frozen snapshot as training set
# MAGIC Materialise the frozen version as an immutable table for model training.

# COMMAND ----------

snapshot_table_name = f"{fqn_prefix}.upt_training_v{snapshot_version}"
frozen_df.write.mode("overwrite").saveAsTable(snapshot_table_name)
print(f"✓ Training snapshot saved: {snapshot_table_name}")
print(f"  Rows: {frozen_count:,}")
print(f"  Columns: {live_cols}")
print(f"  Source version: {snapshot_version}")
print(f"  Source timestamp: {frozen_timestamp}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC **Key compliance statements this enables:**
# MAGIC - "We can cryptographically prove that the 2024 model had zero visibility into 2025 data"
# MAGIC - "Auditors can retrieve the exact feature set for any historical quote in seconds"
# MAGIC - "Feature drift between versions is automatically tracked and quantified"

# COMMAND ----------

print(f"""
Point-in-Time Backtesting Complete
===================================
Frozen Version:    {snapshot_version} ({frozen_timestamp})
Live Version:      latest
Rows Compared:     {frozen_count:,}
Features Analysed: {len(drift_features)}

High Drift Features: {len([r for r in drift_rows if r['drift_severity'] == 'HIGH'])}
Medium Drift:        {len([r for r in drift_rows if r['drift_severity'] == 'MEDIUM'])}
Low Drift:           {len([r for r in drift_rows if r['drift_severity'] == 'LOW'])}

Training Set:        {snapshot_table_name}
Drift Summary:       {fqn_prefix}.pit_drift_summary
""")
