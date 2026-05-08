# Databricks notebook source
# MAGIC %md
# MAGIC # Live Pricing System — provision
# MAGIC
# MAGIC One-shot bring-up of the live pricing demo:
# MAGIC  1. Lakebase online store (CU_2 — smallest performant tier)
# MAGIC  2. Continuous publish of `unified_pricing_table_live` → online store
# MAGIC  3. `pricing_scorer` champion logged + deployed to a route-optimised
# MAGIC     Model Serving endpoint with `scale_to_zero=False`
# MAGIC  4. 5-request warm-up so the first demo quote is sub-second
# MAGIC  5. `live_pricing_metrics` table for the load-test chart
# MAGIC
# MAGIC Idempotent — every step skips work that's already done.
# MAGIC
# MAGIC Notebook exits with JSON describing the live state. Tear-down is the
# MAGIC inverse: `02_teardown.py`.

# COMMAND ----------

dbutils.widgets.text("catalog_name",      "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name",       "pricing_upt")
dbutils.widgets.text("online_store_name", "pricing-upt-online-store-live")
dbutils.widgets.text("endpoint_name",     "pricing_scorer")
dbutils.widgets.text("online_store_capacity", "CU_2")

# COMMAND ----------

# MAGIC %pip install mlflow databricks-feature-engineering databricks-sdk \
# MAGIC   statsmodels lightgbm scikit-learn --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import json, os, time
catalog        = dbutils.widgets.get("catalog_name")
schema         = dbutils.widgets.get("schema_name")
online_store   = dbutils.widgets.get("online_store_name")
endpoint_name  = dbutils.widgets.get("endpoint_name")
capacity       = dbutils.widgets.get("online_store_capacity")
fqn            = f"{catalog}.{schema}"
upt_table      = f"{fqn}.unified_pricing_table_live"
scorer_uc_name = f"{fqn}.pricing_scorer"
metrics_table  = f"{fqn}.live_pricing_metrics"

user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

# COMMAND ----------

# MAGIC %run ../../utils/audit

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.ml import (
    OnlineStore, PublishSpec, PublishSpecPublishMode,
)
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput, ServedEntityInput,
)
from databricks.feature_engineering import FeatureEngineeringClient
from mlflow.tracking import MlflowClient
import mlflow

mlflow.set_registry_uri("databricks-uc")
w  = WorkspaceClient()
mc = MlflowClient()
fe = FeatureEngineeringClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Online store — provision Lakebase

# COMMAND ----------

try:
    store = w.feature_store.get_online_store(online_store)
    print(f"online store exists: {store.name} (state={store.state}, capacity={store.capacity})")
except Exception:
    print(f"creating online store {online_store} at {capacity}…")
    store = w.feature_store.create_online_store(
        online_store=OnlineStore(name=online_store, capacity=capacity)
    )
    print(f"created: {store.name}")

for i in range(60):  # up to 5 min
    store = w.feature_store.get_online_store(online_store)
    if str(store.state).endswith("AVAILABLE"):
        print(f"online store AVAILABLE after {i*5}s")
        break
    print(f"  waiting… state={store.state}")
    time.sleep(5)
else:
    raise RuntimeError(f"online store {online_store} did not become AVAILABLE in 5 min")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Continuous publish UPT → online store
# MAGIC
# MAGIC Continuous mode (per project decision) — every Delta change to UPT is
# MAGIC streamed into Lakebase. Latency from MERGE to online lookup ~5-15s.

# COMMAND ----------

print(f"publishing {upt_table} → {online_store} (CONTINUOUS)…")

# publish_table requires the source to be registered as an FE feature table.
# UPT already has a PK (build_upt.py) but needs explicit fe.create_table to
# satisfy the online publish API. Idempotent — get_table tells us if it's
# already registered.
try:
    fe.get_table(name=upt_table)
    print(f"FE table already registered: {upt_table}")
except Exception:
    print(f"registering {upt_table} as FE table…")
    fe.create_table(
        name         = upt_table,
        primary_keys = "policy_id",
        df           = spark.table(upt_table),
        description  = "Unified Pricing Table — feature table for live pricing FeatureLookup.",
    )
    print(f"registered: {upt_table}")

try:
    result = w.feature_store.publish_table(
        source_table_name=upt_table,
        publish_spec=PublishSpec(
            online_store      = online_store,
            online_table_name = upt_table,
            publish_mode      = PublishSpecPublishMode.CONTINUOUS,
        ),
    )
    print(f"publish_table OK: {getattr(result, 'status', result)}")
except Exception as e:
    err = str(e)
    if "already published" in err.lower() or "already exists" in err.lower():
        print(f"already published — continuous sync continuing")
    else:
        raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. pricing_scorer champion → endpoint
# MAGIC
# MAGIC If no `pricing_scorer` model exists yet (first-run on this catalog),
# MAGIC trigger the production scorer notebook to log + register one. Then
# MAGIC deploy the latest version to a route-optimised endpoint.

# COMMAND ----------

def _latest_version(name: str) -> str | None:
    versions = list(mc.search_model_versions(f"name='{name}'"))
    if not versions:
        return None
    return str(max(int(v.version) for v in versions))

scorer_version = _latest_version(scorer_uc_name)
if scorer_version is None:
    print(f"no version of {scorer_uc_name} found — running 04_models/production/pricing_scorer.py")
    dbutils.notebook.run(
        "../../04_models/production/pricing_scorer",
        timeout_seconds=1800,
        arguments={
            "catalog_name":  catalog,
            "schema_name":   schema,
            "endpoint_name": endpoint_name,
        },
    )
    scorer_version = _latest_version(scorer_uc_name)
    if scorer_version is None:
        raise RuntimeError(f"failed to log {scorer_uc_name}")

print(f"deploying {scorer_uc_name} v{scorer_version} → endpoint {endpoint_name}")

served = [ServedEntityInput(
    entity_name           = scorer_uc_name,
    entity_version        = str(scorer_version),
    scale_to_zero_enabled = False,
    workload_size         = "Small",
)]

try:
    w.serving_endpoints.get(endpoint_name)
    w.serving_endpoints.update_config(name=endpoint_name, served_entities=served)
    print("updated existing endpoint")
except Exception:
    w.serving_endpoints.create(
        name             = endpoint_name,
        config           = EndpointCoreConfigInput(name=endpoint_name, served_entities=served),
        route_optimized  = True,
    )
    print("created endpoint (route_optimized)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Wait for endpoint READY + warm-up

# COMMAND ----------

for i in range(180):  # up to 15 min — first deploy can be slow
    ep = w.serving_endpoints.get(endpoint_name)
    state = getattr(ep.state, "ready", None)
    config_state = getattr(ep.state, "config_update", None)
    print(f"  endpoint state ready={state} config_update={config_state}")
    if str(state).endswith("READY") and not str(config_state).endswith("IN_PROGRESS"):
        break
    time.sleep(5)
else:
    raise RuntimeError(f"endpoint {endpoint_name} not READY in 15 min")

# COMMAND ----------

# Warm up — issue 5 sequential quotes against random policy_ids; discard the
# first 2 latencies (cold path / lazy import). Reports the warm latency.
import requests as _rq

sample_pids = [r["policy_id"] for r in spark.sql(
    f"SELECT policy_id FROM {upt_table} LIMIT 5"
).collect()]
print(f"warm-up policy ids: {sample_pids}")

host  = w.config.host.rstrip("/")
token = w.config._header_factory()
warm_latencies_ms = []
for pid in sample_pids:
    t0 = time.perf_counter()
    resp = _rq.post(
        f"{host}/serving-endpoints/{endpoint_name}/invocations",
        headers={**token, "Content-Type": "application/json"},
        json={"dataframe_records": [{"policy_id": pid}]},
        timeout=120,
    )
    dt = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    warm_latencies_ms.append(round(dt, 1))
    print(f"  {pid} → {dt:.0f} ms")

warm_after_initial = warm_latencies_ms[2:]
warm_p50 = sorted(warm_after_initial)[len(warm_after_initial) // 2] if warm_after_initial else None
print(f"warm latencies: {warm_latencies_ms}  warm-after-initial p50: {warm_p50} ms")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Metrics table for load-test chart

# COMMAND ----------

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {metrics_table} (
        ts             TIMESTAMP,
        source         STRING,
        policy_id      STRING,
        latency_ms     DOUBLE,
        final_premium  DOUBLE,
        status_code    INT,
        run_id         STRING
    ) USING DELTA
""")
print(f"metrics table ready: {metrics_table}")

# COMMAND ----------

log_event(
    spark, catalog, schema,
    event_type    = "live_pricing_started",
    entity_type   = "endpoint",
    entity_id     = endpoint_name,
    entity_version= str(scorer_version),
    user_id       = user,
    details={
        "online_store":     online_store,
        "online_capacity":  capacity,
        "scorer_version":   scorer_version,
        "warm_latencies_ms": warm_latencies_ms,
        "warm_p50_ms":      warm_p50,
        "publish_mode":     "CONTINUOUS",
    },
    source="notebook",
)

# COMMAND ----------

dbutils.notebook.exit(json.dumps({
    "online_store_name": online_store,
    "endpoint_name":     endpoint_name,
    "scorer_version":    scorer_version,
    "warm_p50_ms":       warm_p50,
    "warm_latencies_ms": warm_latencies_ms,
    "metrics_table":     metrics_table,
}))
