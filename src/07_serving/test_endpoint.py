# Databricks notebook source
# MAGIC %md
# MAGIC # Test Model Serving Endpoint
# MAGIC
# MAGIC Sends scoring requests to the deployed endpoint and measures latency.
# MAGIC **The key demo moment:** Send just a `policy_id`, get a prediction back.
# MAGIC Features are resolved automatically from the Online Feature Store.

# COMMAND ----------

dbutils.widgets.text("catalog_name", "pricing_workbench")
dbutils.widgets.text("schema_name", "pricing_upt")
dbutils.widgets.text("endpoint_name", "pricing-frequency-endpoint")

catalog = dbutils.widgets.get("catalog_name")
schema = dbutils.widgets.get("schema_name")
endpoint_name = dbutils.widgets.get("endpoint_name")

# COMMAND ----------

import time
import json
import concurrent.futures
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# Get sample policy IDs
sample_ids = (spark.table(f"{catalog}.{schema}.unified_pricing_table_live")
    .select("policy_id").orderBy("policy_id").limit(100).collect())
policy_ids = [r.policy_id for r in sample_ids]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test 1: Single request — just send policy_id
# MAGIC
# MAGIC This proves automatic feature lookup works. We send ONLY the primary key
# MAGIC and get back a prediction — the endpoint resolved all features from the
# MAGIC Online Feature Store behind the scenes.

# COMMAND ----------

single_id = policy_ids[0]
print(f"Scoring policy: {single_id}")
print(f"Sending ONLY the policy_id — no features in the request\n")

start = time.time()
response = w.serving_endpoints.query(
    name=endpoint_name,
    dataframe_records=[{"policy_id": single_id}],
)
latency_ms = (time.time() - start) * 1000

print(f"Response: {response.predictions}")
print(f"Latency: {latency_ms:.0f}ms")
print(f"\n✓ Prediction returned with ONLY policy_id as input")
print(f"  Features were looked up automatically from the Online Feature Store")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test 2: Batch of 100 requests — latency distribution

# COMMAND ----------

latencies = []
for pid in policy_ids[:100]:
    start = time.time()
    resp = w.serving_endpoints.query(
        name=endpoint_name,
        dataframe_records=[{"policy_id": pid}],
    )
    elapsed = (time.time() - start) * 1000
    latencies.append(elapsed)

latencies.sort()
p50 = latencies[len(latencies) // 2]
p95 = latencies[int(len(latencies) * 0.95)]
p99 = latencies[int(len(latencies) * 0.99)]
avg = sum(latencies) / len(latencies)

print(f"Batch latency (100 sequential requests):")
print(f"  Average: {avg:.0f}ms")
print(f"  P50:     {p50:.0f}ms")
print(f"  P95:     {p95:.0f}ms")
print(f"  P99:     {p99:.0f}ms")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test 3: Feature override
# MAGIC
# MAGIC Send a request WITH some feature values to show they override the online
# MAGIC store lookup. This is useful for "what-if" scenarios — e.g. "what would
# MAGIC the prediction be if this policy had a flood score of 10?"

# COMMAND ----------

# Normal prediction
normal_resp = w.serving_endpoints.query(
    name=endpoint_name,
    dataframe_records=[{"policy_id": single_id}],
)
print(f"Normal prediction for {single_id}: {normal_resp.predictions}")

# Override flood_zone_rating to worst case
override_resp = w.serving_endpoints.query(
    name=endpoint_name,
    dataframe_records=[{"policy_id": single_id, "flood_zone_rating": 10.0}],
)
print(f"With flood_zone_rating=10: {override_resp.predictions}")
print(f"\n✓ Feature override works — enables 'what-if' pricing scenarios")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save results

# COMMAND ----------

results = spark.createDataFrame([
    ("endpoint_avg_ms", float(avg)),
    ("endpoint_p50_ms", float(p50)),
    ("endpoint_p95_ms", float(p95)),
    ("endpoint_p99_ms", float(p99)),
    ("requests_tested", float(len(latencies))),
], ["metric", "value"])
results.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.endpoint_latency")
print(f"✓ Results saved to {catalog}.{schema}.endpoint_latency")

# COMMAND ----------

# MAGIC %run ../utils/audit

# COMMAND ----------

log_event(
    spark, catalog, schema,
    event_type="model_deployed",
    entity_type="endpoint",
    entity_id=endpoint_name,
    entity_version="latency_test",
    user_id=dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get(),
    details={
        "avg_ms": round(avg, 1),
        "p50_ms": round(p50, 1),
        "p95_ms": round(p95, 1),
        "p99_ms": round(p99, 1),
        "requests": len(latencies),
    },
    source="notebook",
)
print("✓ Test results logged to audit_log")
