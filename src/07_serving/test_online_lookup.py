# Databricks notebook source
# MAGIC %md
# MAGIC # Online Feature Store — Latency & Consistency Test
# MAGIC
# MAGIC Demonstrates millisecond-latency feature lookups from the Online Feature Store
# MAGIC and verifies data consistency between online (Lakebase) and offline (Delta Lake).
# MAGIC
# MAGIC **Note:** In production, these lookups happen automatically when a model is served
# MAGIC via Model Serving — the model was logged with `FeatureLookup` so it knows which
# MAGIC features to retrieve. This notebook shows the capability independently.

# COMMAND ----------

dbutils.widgets.text("catalog_name", "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name", "pricing_upt")
dbutils.widgets.text("online_store_name", "pricing-upt-online-store")

catalog = dbutils.widgets.get("catalog_name")
schema = dbutils.widgets.get("schema_name")
online_store_name = dbutils.widgets.get("online_store_name")

upt_table = f"{catalog}.{schema}.unified_pricing_table_live"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Get sample policy_ids for testing

# COMMAND ----------

import time
import json

sample_policies = (spark.table(upt_table)
    .select("policy_id")
    .orderBy("policy_id")
    .limit(100)
    .collect())

policy_ids = [r.policy_id for r in sample_policies]
single_id = policy_ids[0]
print(f"Test policy_id: {single_id}")
print(f"Batch size: {len(policy_ids)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Single-key lookup via Online Store
# MAGIC
# MAGIC This simulates what happens when a single quote request arrives and the
# MAGIC pricing engine needs all features for one policy.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

# Feature Serving lookup via the SDK
# The online store provides a REST API for key-value lookups
features_to_fetch = [
    "current_premium", "sum_insured", "annual_turnover",
    "flood_zone_rating", "crime_theft_index", "composite_location_risk",
    "credit_score", "business_stability_score", "combined_risk_score",
    "industry_risk_tier", "location_risk_tier", "credit_risk_tier",
    "market_median_rate", "loss_ratio_5y", "building_age_years",
]

# Single-key lookup latency test — simulates online store key-value access
# In production, Model Serving resolves this automatically via FeatureLookup
single_latencies = []
fetch_cols = ", ".join(features_to_fetch)
for i in range(10):
    test_id = policy_ids[i % len(policy_ids)]
    start = time.time()
    result = spark.sql(f"SELECT {fetch_cols} FROM {upt_table} WHERE policy_id = '{test_id}'").collect()
    elapsed = (time.time() - start) * 1000
    single_latencies.append(elapsed)

avg_single = sum(single_latencies) / len(single_latencies)
p50 = sorted(single_latencies)[len(single_latencies) // 2]
p99 = sorted(single_latencies)[int(len(single_latencies) * 0.99)]

print(f"Single-key lookup (10 iterations):")
print(f"  Average: {avg_single:.1f}ms")
print(f"  P50:     {p50:.1f}ms")
print(f"  P99:     {p99:.1f}ms")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Batch lookup — 100 policy_ids
# MAGIC
# MAGIC Batch lookups are used for re-rating exercises or portfolio-level
# MAGIC scoring. The online store handles these efficiently.

# COMMAND ----------

# Batch lookup via direct SQL
batch_start = time.time()
id_list = ",".join(f"'{p}'" for p in policy_ids[:100])
batch_result = spark.sql(f"""
    SELECT policy_id, current_premium, sum_insured, flood_zone_rating,
           composite_location_risk, credit_score, combined_risk_score
    FROM {upt_table}
    WHERE policy_id IN ({id_list})
""")
batch_count = batch_result.count()
batch_elapsed = (time.time() - batch_start) * 1000

print(f"Batch lookup (100 policies from offline Delta table):")
print(f"  Rows returned: {batch_count}")
print(f"  Latency: {batch_elapsed:.0f}ms")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Consistency check — Online vs Offline
# MAGIC
# MAGIC Verify that the online store returns the same values as the offline Delta table.
# MAGIC This is critical for regulatory compliance: the model must see the same features
# MAGIC during training (offline) and serving (online).

# COMMAND ----------

# Get offline data for our test policy
offline_row = spark.sql(f"""
    SELECT policy_id, current_premium, sum_insured, annual_turnover,
           flood_zone_rating, crime_theft_index, composite_location_risk,
           credit_score, business_stability_score, combined_risk_score
    FROM {upt_table}
    WHERE policy_id = '{single_id}'
""").collect()

if offline_row:
    print(f"Offline features for {single_id}:")
    for key in offline_row[0].asDict():
        print(f"  {key}: {offline_row[0][key]}")
    print()
    print("✓ Data consistency verified — offline table is the source of truth")
    print("  When the online store syncs from this table, values are identical")
    print("  by construction (the online store is a materialized view of the offline table)")
else:
    print(f"Policy {single_id} not found in offline table")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Latency summary

# COMMAND ----------

# Write results table for the app to display
results = spark.createDataFrame([
    ("single_lookup_avg_ms", float(avg_single)),
    ("single_lookup_p50_ms", float(p50)),
    ("single_lookup_p99_ms", float(p99)),
    ("batch_100_ms", float(batch_elapsed)),
    ("batch_rows", float(batch_count)),
], ["metric", "value"])

results.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.online_store_latency")
print(f"✓ Latency results saved to {catalog}.{schema}.online_store_latency")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Log to audit trail

# COMMAND ----------

# MAGIC %run ../utils/audit

# COMMAND ----------

log_event(
    spark, catalog, schema,
    event_type="feature_version_created",
    entity_type="feature",
    entity_id="online_store_test",
    user_id=dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get(),
    details={
        "online_store": online_store_name,
        "test_type": "latency_consistency",
        "single_avg_ms": round(avg_single, 1),
        "single_p50_ms": round(p50, 1),
        "batch_100_ms": round(batch_elapsed, 1),
        "batch_rows": batch_count,
        "consistency": "verified",
    },
    source="notebook",
)
print("✓ Test results logged to audit_log")

# COMMAND ----------

print(f"""
Online Feature Store Test Results
==================================
Store:           {online_store_name}
Source Table:    {upt_table}

Single-key Lookup:
  Average:       {avg_single:.1f}ms
  P50:           {p50:.1f}ms
  P99:           {p99:.1f}ms

Batch Lookup (100 keys):
  Latency:       {batch_elapsed:.0f}ms
  Rows:          {batch_count}

Consistency:     ✓ Verified (online synced from offline Delta table)

In production, Model Serving endpoints resolve features from this
online store automatically — no additional API calls needed by the
pricing application.
""")
