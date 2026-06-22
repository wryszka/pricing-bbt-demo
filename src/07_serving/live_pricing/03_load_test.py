# Databricks notebook source
# MAGIC %md
# MAGIC # Live Pricing System — load test
# MAGIC
# MAGIC Async httpx + asyncio load against `pricing_scorer`. Targets a fixed
# MAGIC QPS (default 100) for a fixed duration (default 300s). Per-request
# MAGIC outcome is summarised into per-second rows and written to
# MAGIC `live_pricing_metrics` so the live chart in the app can plot p50/p95/p99
# MAGIC + QPS in real time.
# MAGIC
# MAGIC No new dependencies — `httpx` and `asyncio` ship with the runtime.

# COMMAND ----------

dbutils.widgets.text("catalog_name",     "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name",      "pricing_upt")
dbutils.widgets.text("endpoint_name",    "pricing_scorer")
dbutils.widgets.text("target_qps",       "100")
dbutils.widgets.text("duration_seconds", "60")
dbutils.widgets.text("concurrency",      "50")
dbutils.widgets.text("run_id",           "")

# COMMAND ----------

# MAGIC %pip install httpx databricks-sdk nest_asyncio --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import asyncio, json, time, uuid, statistics
from datetime import datetime, timezone
import httpx, nest_asyncio
from databricks.sdk import WorkspaceClient

# Databricks notebooks run inside an existing IPython event loop, so plain
# asyncio.run() raises 'cannot be called from a running event loop'.
nest_asyncio.apply()

catalog       = dbutils.widgets.get("catalog_name")
schema        = dbutils.widgets.get("schema_name")
endpoint_name = dbutils.widgets.get("endpoint_name")
target_qps    = int(dbutils.widgets.get("target_qps"))
duration_s    = int(dbutils.widgets.get("duration_seconds"))
concurrency   = int(dbutils.widgets.get("concurrency"))
run_id_in     = dbutils.widgets.get("run_id").strip()

run_id        = run_id_in or f"loadtest_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
fqn           = f"{catalog}.{schema}"
upt_table     = f"{fqn}.unified_pricing_table_live"
metrics_table = f"{fqn}.live_pricing_metrics"

print(f"run_id={run_id}  target={target_qps} qps for {duration_s}s  concurrency={concurrency}")

# COMMAND ----------

w     = WorkspaceClient()
host  = w.config.host.rstrip("/")
url   = f"{host}/serving-endpoints/{endpoint_name}/invocations"
token = w.config._header_factory()

# Sample policy_ids — pull a few thousand to randomise across the load test
sample_size = max(1000, target_qps * 10)
policy_ids = [r["policy_id"] for r in spark.sql(
    f"SELECT policy_id FROM {upt_table} ORDER BY rand() LIMIT {sample_size}"
).collect()]
print(f"sampled {len(policy_ids):,} policy_ids")

# COMMAND ----------

# Pacing — fire one request every (1/target_qps) seconds. Cap concurrency so
# a slow tail doesn't blow up open file descriptors. asyncio.Semaphore makes
# the worker throttle naturally; we drop requests if pacing falls more than
# 5s behind (the chart still renders the slowdown via lower QPS).
sem = asyncio.Semaphore(concurrency)
results: list[tuple[float, float, int]] = []  # (ts, latency_ms, status_code)

async def _one(client: httpx.AsyncClient, pid: str):
    async with sem:
        # Bucket by request fire time, not completion time — keeps the QPS
        # curve aligned with target_qps and stops drain-phase responses
        # piling into the final 1-2s wall-clock buckets.
        request_ts = time.time()
        t0 = time.perf_counter()
        try:
            r = await client.post(url,
                headers={**token, "Content-Type": "application/json"},
                json={"dataframe_records": [{"policy_id": pid}]},
                timeout=15.0,
            )
            dt = (time.perf_counter() - t0) * 1000.0
            results.append((request_ts, dt, r.status_code))
        except Exception:
            dt = (time.perf_counter() - t0) * 1000.0
            results.append((request_ts, dt, 0))

async def _run():
    period = 1.0 / target_qps
    deadline = time.monotonic() + duration_s
    n_fired = 0
    next_fire = time.monotonic()
    async with httpx.AsyncClient(http2=False, limits=httpx.Limits(max_connections=concurrency * 2)) as client:
        tasks: list[asyncio.Task] = []
        i = 0
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now < next_fire:
                await asyncio.sleep(min(0.005, next_fire - now))
                continue
            # If we're more than 5s behind, drop the backlog so pacing recovers
            if now - next_fire > 5.0:
                next_fire = now
            pid = policy_ids[i % len(policy_ids)]
            i += 1
            tasks.append(asyncio.create_task(_one(client, pid)))
            n_fired += 1
            next_fire += period
        # Wait for in-flight to drain (cap at duration + 30s)
        if tasks:
            await asyncio.wait(tasks, timeout=30.0)
    return n_fired

test_start_wall = time.time()
n_fired = asyncio.run(_run())
test_end_wall = test_start_wall + duration_s
print(f"fired {n_fired} requests, captured {len(results)} responses")

# COMMAND ----------

# Aggregate per-second, dropping anything outside the nominal test window.
buckets: dict[int, list[tuple[float, int]]] = {}
for ts, dt, status in results:
    if ts < test_start_wall or ts > test_end_wall:
        continue
    sec = int(ts)
    buckets.setdefault(sec, []).append((dt, status))

rows = []
for sec in sorted(buckets):
    bucket = buckets[sec]
    lats = [d for d, _ in bucket]
    statuses = [s for _, s in bucket]
    qps = len(bucket)
    p50 = statistics.median(lats) if lats else 0.0
    p95 = sorted(lats)[max(0, int(len(lats) * 0.95) - 1)] if lats else 0.0
    p99 = sorted(lats)[max(0, int(len(lats) * 0.99) - 1)] if lats else 0.0
    err_rate = sum(1 for s in statuses if s != 200) / max(1, len(statuses))
    rows.append({
        "ts":            datetime.fromtimestamp(sec, tz=timezone.utc),
        "source":        "load_test",
        "policy_id":     "",
        "latency_ms":    round(p50, 2),
        "final_premium": None,
        "status_code":   int(round(err_rate * 100)),
        "run_id":        run_id,
        "qps":           qps,
        "p95_ms":        round(p95, 2),
        "p99_ms":        round(p99, 2),
    })

# COMMAND ----------

# Write per-second summaries. We persist the headline (p50 in `latency_ms`)
# plus extras (qps, p95, p99) in a side table so the metrics-table schema
# stays aligned with the per-quote rows the FastAPI route writes.
import pyspark.sql.functions as F
from pyspark.sql.types import (
    StructType, StructField, TimestampType, StringType, DoubleType, IntegerType,
)

if rows:
    base_rows = [{k: v for k, v in r.items() if k in
                  ("ts", "source", "policy_id", "latency_ms",
                   "final_premium", "status_code", "run_id")}
                 for r in rows]
    base_schema = StructType([
        StructField("ts",            TimestampType(), True),
        StructField("source",        StringType(),    True),
        StructField("policy_id",     StringType(),    True),
        StructField("latency_ms",    DoubleType(),    True),
        StructField("final_premium", DoubleType(),    True),
        StructField("status_code",   IntegerType(),   True),
        StructField("run_id",        StringType(),    True),
    ])
    spark.createDataFrame(base_rows, schema=base_schema) \
         .write.mode("append").saveAsTable(metrics_table)

    # Wider schema with qps/p95/p99 for the chart
    summary_table = f"{fqn}.live_pricing_load_test_summary"
    summary_schema = StructType([
        StructField("ts",          TimestampType(), True),
        StructField("run_id",      StringType(),    True),
        StructField("qps",         IntegerType(),   True),
        StructField("p50_ms",      DoubleType(),    True),
        StructField("p95_ms",      DoubleType(),    True),
        StructField("p99_ms",      DoubleType(),    True),
        StructField("error_pct",   IntegerType(),   True),
    ])
    summary_rows = [{
        "ts": r["ts"], "run_id": r["run_id"], "qps": r["qps"],
        "p50_ms": r["latency_ms"], "p95_ms": r["p95_ms"], "p99_ms": r["p99_ms"],
        "error_pct": r["status_code"],
    } for r in rows]
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {summary_table} (
            ts TIMESTAMP, run_id STRING, qps INT,
            p50_ms DOUBLE, p95_ms DOUBLE, p99_ms DOUBLE,
            error_pct INT
        ) USING DELTA
    """)
    spark.createDataFrame(summary_rows, schema=summary_schema) \
         .write.mode("append").saveAsTable(summary_table)
    print(f"wrote {len(rows)} per-second rows to {metrics_table} and {summary_table}")

# COMMAND ----------

all_lats = [d for _, d, _ in results]
overall = {
    "run_id":          run_id,
    "fired":           n_fired,
    "captured":        len(results),
    "duration_s":      duration_s,
    "target_qps":      target_qps,
    "p50_ms":          round(statistics.median(all_lats), 2) if all_lats else None,
    "p95_ms":          round(sorted(all_lats)[max(0, int(len(all_lats) * 0.95) - 1)], 2) if all_lats else None,
    "p99_ms":          round(sorted(all_lats)[max(0, int(len(all_lats) * 0.99) - 1)], 2) if all_lats else None,
    "error_count":     sum(1 for _, _, s in results if s != 200),
}
print(json.dumps(overall, indent=2))
dbutils.notebook.exit(json.dumps(overall))
